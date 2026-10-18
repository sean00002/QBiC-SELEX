"""z-scores and p-values for predicted effects, batched on CPU (BLAS) or GPU (CuPy).

For variant i, let c_i be its k-mer count difference (alt minus ref) over the
model's canonical k-mers, beta the model coefficients and Sigma their
covariance. Then

    estimate_i = c_i . beta
    se_i       = sqrt(c_i^T Sigma c_i)
    z_i        = estimate_i / se_i
    p_i        = 2 * (1 - Phi(|z_i|))

computed for a chunk of variants at once as C @ beta and rowsum((C @ Sigma) * C).
See docs/statistics.md.
"""

import logging

import numpy as np
import scipy.stats

from .kmers import reverse_complement

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    tqdm = None


def gpu_check() -> str | None:
    """None if GPU statistics will work, else the reason they will not.

    Runs a tiny matrix product, so a missing CUDA library (e.g. cuBLAS) shows
    up here rather than in the middle of a run.
    """
    try:
        import cupy as cp
    except ImportError:
        return "cupy is not installed (pip install 'qbic-selex[gpu]')"
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return "no CUDA device visible"
        a = cp.ones((4, 4), dtype=cp.float32)
        float((a @ a).sum())
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def gpu_name() -> str:
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        return f"{props['name'].decode()} ({props['totalGlobalMem'] / 1024 ** 3:.1f} GB)"
    except Exception:
        return "unknown GPU"


def build_c_matrix(alt_kmers: list[list[str]], ref_kmers: list[list[str]],
                   partial: dict, start: int, end: int) -> np.ndarray:
    """Count-difference matrix C for variants [start, end).

    C[i, j] = occurrences of canonical k-mer j (or its reverse complement) in
    the alt sequence minus those in the ref sequence. int16, so long or
    repetitive sequences cannot overflow.
    """
    key_to_idx = {key: idx for idx, key in enumerate(partial)}
    for key, idx in list(key_to_idx.items()):
        key_to_idx.setdefault(reverse_complement(key), idx)
    C = np.zeros((end - start, len(partial)), dtype=np.int16)
    for local_i, i in enumerate(range(start, end)):
        row = C[local_i]
        for kmer in alt_kmers[i]:
            idx = key_to_idx.get(kmer)
            if idx is not None:
                row[idx] += 1
        for kmer in ref_kmers[i]:
            idx = key_to_idx.get(kmer)
            if idx is not None:
                row[idx] -= 1
    return C


def _finish(estimates: np.ndarray, se: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p-values plus the edge cases: a zero effect gives (z=0, p=1); an invalid SE gives NaN."""
    p = 2.0 * scipy.stats.norm.sf(np.abs(z))
    zero = estimates == 0
    bad_se = ~np.isfinite(se) | (se <= 0)
    z[zero], p[zero] = 0.0, 1.0
    z[bad_se & ~zero] = np.nan
    p[bad_se & ~zero] = np.nan
    return z, p


def compute_statistics(alt_kmers: list[list[str]], ref_kmers: list[list[str]], partial: dict,
                       cov: np.ndarray, use_gpu: bool = False,
                       chunk_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """z-scores and two-sided p-values for every variant (float32 linear algebra)."""
    n = len(alt_kmers)
    z_all = np.full(n, np.nan, dtype=np.float64)
    p_all = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return z_all, p_all

    if use_gpu:
        import cupy as xp
        chunk_size = chunk_size or 100_000
    else:
        xp = np
        chunk_size = chunk_size or 20_000

    beta = xp.asarray(np.array(list(partial.values()), dtype=np.float32))
    sigma = xp.asarray(cov, dtype=xp.float32)

    starts = range(0, n, chunk_size)
    if tqdm is not None:
        # disable=None: no bar when stderr is not a terminal (logs, batch jobs)
        starts = tqdm(starts, total=(n + chunk_size - 1) // chunk_size, unit="chunk", leave=False,
                      desc=f"Statistics ({'GPU' if use_gpu else 'CPU'})",
                      disable=None if log.isEnabledFor(logging.INFO) else True)
    for start in starts:
        end = min(start + chunk_size, n)
        C = xp.asarray(build_c_matrix(alt_kmers, ref_kmers, partial, start, end), dtype=xp.float32)
        estimates = C @ beta
        se = xp.sqrt(xp.sum((C @ sigma) * C, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = estimates / se
        if use_gpu:
            estimates, se, z = (xp.asnumpy(a) for a in (estimates, se, z))
        z, p = _finish(estimates, se, z.astype(np.float64))
        z_all[start:end] = z
        p_all[start:end] = p

    if use_gpu:
        xp.get_default_memory_pool().free_all_blocks()
    return z_all, p_all
