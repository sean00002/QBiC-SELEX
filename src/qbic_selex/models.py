"""Loading QBiC-SELEX models and covariance matrices, and resolving which to run."""

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kmers import recover_non_rc_dict

log = logging.getLogger(__name__)

MODEL_SUFFIX = ".weights.qbic"
_COV_NPY_RE = re.compile(r"\.cov_(\d+)\.npy$")


class ModelInputError(ValueError):
    """Raised when model or covariance inputs are missing or inconsistent."""


@dataclass
class Model:
    name: str
    path: Path
    kmer_size: int
    partial: dict[str, float]     # canonical k-mers, in file order (matches the covariance)
    complete: dict[str, float]    # all 4^k k-mers


def model_name_from_path(path) -> str:
    """Model name from a model or covariance file path.

    ``X.weights.qbic`` -> ``X``; ``X.cov_8192.npy`` -> ``X``; otherwise the
    part of the file name before the first dot.
    """
    name = Path(path).name
    if name.endswith(MODEL_SUFFIX):
        return name[:-len(MODEL_SUFFIX)]
    match = _COV_NPY_RE.search(name)
    if match:
        return name[:match.start()]
    return name.split(".")[0]


# ----------------------------------------------------------------------------
# Restricted unpickling: model files are pickled {str: numpy.float32} dicts.
# Only the numpy constructors those need are allowed, so a tampered file cannot
# run arbitrary code.
# ----------------------------------------------------------------------------

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "dtype"),
    # legacy pickled covariance matrices
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"disallowed object in model file: {module}.{name}")


def _safe_unpickle(path: Path):
    with open(path, "rb") as f:
        return _RestrictedUnpickler(f).load()


def load_model(path) -> Model:
    path = Path(path)
    partial = _safe_unpickle(path)
    if not isinstance(partial, dict) or not partial:
        raise ModelInputError(f"{path}: not a QBiC-SELEX model (expected a non-empty k-mer dict)")
    kmer_size = len(next(iter(partial)))
    if any(len(k) != kmer_size for k in partial):
        raise ModelInputError(f"{path}: k-mers of different lengths in one model")
    complete = recover_non_rc_dict(partial)
    if len(complete) != 4 ** kmer_size:
        raise ModelInputError(f"{path}: expected {4 ** kmer_size} k-mers after reverse-complement "
                              f"expansion, got {len(complete)}")
    return Model(model_name_from_path(path), path, kmer_size, partial, complete)


def load_covariance(path, expected_dim: int | None = None) -> np.ndarray:
    """Load a covariance matrix.

    ``*.cov_<n>.npy`` files hold the upper triangle (row-major, ``np.triu_indices``
    order) of an n x n matrix; legacy pickles hold the full matrix.
    """
    path = Path(path)
    match = _COV_NPY_RE.search(path.name)
    if match:
        n = int(match.group(1))
        upper = np.load(path)
        if upper.ndim != 1 or upper.size != n * (n + 1) // 2:
            raise ModelInputError(f"{path}: expected {n * (n + 1) // 2} upper-triangle values "
                                  f"for a {n}x{n} matrix, found shape {upper.shape}")
        cov = np.empty((n, n), dtype=upper.dtype)
        iu, ju = np.triu_indices(n)
        cov[iu, ju] = upper
        cov[ju, iu] = upper
    elif path.suffix == ".npy":
        raise ModelInputError(f"{path}: covariance .npy files must be named <model>.cov_<n>.npy")
    else:
        cov = np.asarray(_safe_unpickle(path))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ModelInputError(f"{path}: covariance matrix is not square: {cov.shape}")
    if expected_dim is not None and cov.shape[0] != expected_dim:
        raise ModelInputError(f"{path}: covariance is {cov.shape[0]}x{cov.shape[0]} but the model "
                              f"has {expected_dim} coefficients (wrong covariance for this model?)")
    return cov


# ----------------------------------------------------------------------------
# Resolving --models / --covs arguments
# ----------------------------------------------------------------------------

def _read_list_file(path: Path) -> list[str]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries


def _resolve_entry(entry: str, list_file: Path) -> Path | None:
    """A list-file entry as given (absolute or relative to the CWD), else relative to the list file."""
    for candidate in (Path(entry), list_file.parent / entry):
        if candidate.exists():
            return candidate
    return None


def _index_dir(directory: Path, pattern: str) -> dict[str, Path]:
    index = {}
    for p in sorted(directory.rglob(pattern)):
        index.setdefault(model_name_from_path(p), p)
    return index


def resolve_models(spec: str, model_dir: str | None = None) -> list[Path]:
    """Model paths from a model file, a directory of models, or a list file.

    List-file entries are paths, or bare model names looked up in ``model_dir``.
    """
    p = Path(spec)
    if not p.exists():
        raise ModelInputError(f"model input not found: {spec}")
    if p.is_dir():
        paths = sorted(p.rglob("*" + MODEL_SUFFIX))
        if not paths:
            raise ModelInputError(f"no *{MODEL_SUFFIX} files under {spec}")
        return paths
    if p.name.endswith(".qbic"):
        return [p]

    dir_index = _index_dir(Path(model_dir), "*" + MODEL_SUFFIX) if model_dir else None
    paths, missing = [], []
    for entry in _read_list_file(p):
        found = _resolve_entry(entry, p)
        if found is None and dir_index is not None:
            found = dir_index.get(model_name_from_path(entry))
        if found is None:
            missing.append(entry)
        else:
            paths.append(found)
    if missing:
        hint = "" if model_dir else " (bare model names need --model-dir)"
        shown = "\n  ".join(missing[:10]) + ("\n  ..." if len(missing) > 10 else "")
        raise ModelInputError(f"{len(missing)} model(s) in {spec} not found{hint}:\n  {shown}")
    if not paths:
        raise ModelInputError(f"no models listed in {spec}")
    return paths


def resolve_covariances(spec: str, model_names: list[str]) -> dict[str, Path]:
    """Map each model name to its covariance file.

    ``spec`` is a single covariance file (only for a single model), a directory
    of ``*.cov_*.npy`` files, or a list file whose lines are either a covariance
    path or ``model_path,cov_path``.
    """
    p = Path(spec)
    if not p.exists():
        raise ModelInputError(f"covariance input not found: {spec}")

    if p.is_dir():
        index = _index_dir(p, "*.cov_*.npy")
    elif p.suffix == ".npy" or (p.suffix == ".qbic" and "cov" in p.name):
        if len(model_names) != 1:
            raise ModelInputError("a single covariance file was given for several models; "
                                  "pass a directory or a list file of covariance matrices")
        return {model_names[0]: p}
    else:
        index, missing_files = {}, []
        for entry in _read_list_file(p):
            if "," in entry:
                model_part, cov_part = (s.strip() for s in entry.split(",", 1))
                name = model_name_from_path(model_part)
            else:
                cov_part = entry
                name = model_name_from_path(cov_part)
            found = _resolve_entry(cov_part, p)
            if found is None:
                missing_files.append(cov_part)
            else:
                index[name] = found
        if missing_files:
            log.warning("%d covariance file(s) listed in %s do not exist", len(missing_files), spec)

    missing = [m for m in model_names if m not in index]
    if missing:
        shown = "\n  ".join(missing[:10]) + ("\n  ..." if len(missing) > 10 else "")
        raise ModelInputError(f"{len(missing)} model(s) have no covariance matrix in {spec}:\n  {shown}")
    return {m: index[m] for m in model_names}
