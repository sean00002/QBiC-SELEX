"""Scoring variants with QBiC-SELEX models and writing the results."""

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import sequences as seqs
from .kmers import sliding_window
from .models import Model, load_covariance, load_model
from .output import apply_short_output
from .stats import compute_statistics

log = logging.getLogger(__name__)


@dataclass
class Scores:
    effect: np.ndarray                 # float32, NaN where not scored
    status: np.ndarray                 # object array of status strings
    z: np.ndarray | None = None     # float64, only when statistics are computed
    p: np.ndarray | None = None


def score_model(model: Model, ref_seqs: list[str], alt_seqs: list[str], status: np.ndarray,
                cov: np.ndarray | None = None, use_gpu: bool = False) -> Scores:
    """Predicted effect (sum of alt k-mer coefficients minus ref), plus z/p if ``cov`` is given.

    ``ref_seqs``/``alt_seqs``/``status`` come from :func:`sequences.prepare_sequences`.
    """
    k = model.kmer_size
    status = status.copy()
    n = len(status)
    ok = np.flatnonzero(status == seqs.OK)
    too_short = [i for i in ok if len(ref_seqs[i]) < k or len(alt_seqs[i]) < k]
    if too_short:
        status[too_short] = seqs.SHORTER_THAN_K
        ok = np.flatnonzero(status == seqs.OK)

    alt_kmers = [sliding_window(alt_seqs[i], k) for i in ok]
    ref_kmers = [sliding_window(ref_seqs[i], k) for i in ok]

    # Sequential float32 accumulation, as in the original implementation.
    values = {kmer: np.float32(v) for kmer, v in model.complete.items()}
    effect = np.full(n, np.nan, dtype=np.float32)
    for j, i in enumerate(ok):
        effect[i] = (sum(values[kmer] for kmer in alt_kmers[j])
                     - sum(values[kmer] for kmer in ref_kmers[j]))

    scores = Scores(effect=effect, status=status)
    if cov is not None:
        z_ok, p_ok = compute_statistics(alt_kmers, ref_kmers, model.partial, cov, use_gpu=use_gpu)
        scores.z = np.full(n, np.nan)
        scores.p = np.full(n, np.nan)
        scores.z[ok] = z_ok
        scores.p[ok] = p_ok
    return scores


@dataclass
class RunConfig:
    with_stats: bool = False
    use_gpu: bool = False
    pval: float | None = None
    short_output: bool = False
    model_ids: dict[str, int] | None = None
    output_dir_mode: bool = False
    wildcard: str | None = None


@dataclass
class RunSummary:
    models_ok: list[str] = field(default_factory=list)
    models_failed: dict[str, str] = field(default_factory=dict)   # name -> traceback
    rows_written: int = 0
    rows_before_filter: int = 0


def results_table(variants: pd.DataFrame, model_name: str, scores: Scores, with_stats: bool) -> pd.DataFrame:
    """Input columns + model + predicted_effect [+ z_score, p_value] + qbic_status."""
    out = variants.drop(columns=[seqs.STATUS_COLUMN], errors="ignore").copy()
    for col in seqs.SEQUENCE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["model"] = model_name
    out["predicted_effect"] = scores.effect
    if with_stats:
        out["z_score"] = scores.z
        out["p_value"] = scores.p
    out[seqs.STATUS_COLUMN] = scores.status
    return out


def _filter_and_format(table: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    if cfg.pval is not None:
        table = table[table["p_value"].notna() & (table["p_value"] <= cfg.pval)]
    if cfg.short_output:
        # The compact format only carries scored rows.
        table = table[table[seqs.STATUS_COLUMN] == seqs.OK]
        table = apply_short_output(table, cfg.model_ids, cfg.with_stats)
    return table


def run_models(variants: pd.DataFrame, model_paths: list[Path], output: str, cfg: RunConfig,
               cov_paths: dict[str, Path] | None = None) -> RunSummary:
    """Score every model on ``variants`` (which must have sequences) and write results.

    Models are processed one at a time and written as they finish, so memory
    stays bounded by one model's results. A model whose prediction or
    statistics fail is recorded in the summary and left out of the output.
    """
    ref_seqs, alt_seqs, status = seqs.prepare_sequences(variants, cfg.wildcard)
    summary = RunSummary()
    out_path = Path(output)
    if cfg.output_dir_mode:
        out_path.mkdir(parents=True, exist_ok=True)
    elif out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False

    for n_done, path in enumerate(model_paths, 1):
        name = path.name
        try:
            model = load_model(path)
            name = model.name
            log.info("[%d/%d] %s", n_done, len(model_paths), name)
            cov = None
            if cfg.with_stats:
                cov = load_covariance(cov_paths[name], expected_dim=len(model.partial))
            scores = score_model(model, ref_seqs, alt_seqs, status, cov=cov, use_gpu=cfg.use_gpu)
            del cov
            table = results_table(variants, name, scores, cfg.with_stats)
            summary.rows_before_filter += len(table)
            table = _filter_and_format(table, cfg)
        except Exception as e:
            log.error("Model %s failed: %s: %s", name, type(e).__name__, e)
            summary.models_failed[name] = traceback.format_exc()
            continue

        if cfg.output_dir_mode:
            table.to_csv(out_path / f"{name}.csv", index=False)
        else:
            table.to_csv(out_path, index=False, mode="a" if header_written else "w",
                         header=not header_written)
            header_written = True
        summary.rows_written += len(table)
        summary.models_ok.append(name)

    return summary


def write_error_report(summary: RunSummary, directory: Path) -> Path | None:
    if not summary.models_failed:
        return None
    path = directory / f"qbic_error_report_{datetime.now():%Y%m%d_%H%M%S}.log"
    with open(path, "w") as f:
        f.write(f"QBiC-SELEX error report ({datetime.now().isoformat(timespec='seconds')})\n")
        f.write(f"{len(summary.models_failed)} model(s) failed\n\n")
        for name, tb in summary.models_failed.items():
            f.write(f"=== {name}\n{tb}\n")
    return path
