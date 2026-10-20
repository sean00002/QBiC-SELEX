"""GPU statistics must agree with CPU. Run on a GPU node with: pytest -m gpu"""

import numpy as np
import pytest

from qbic_selex.kmers import sliding_window
from qbic_selex.models import load_covariance, load_model
from qbic_selex.stats import compute_statistics, gpu_check

from .conftest import EXAMPLE_MODELS
from .test_predict_stats import random_pairs

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    problem = gpu_check()
    if problem:
        pytest.fail(f"GPU tests selected but the GPU is not usable: {problem}")


def kmers(seqs, k):
    return [sliding_window(s, k) for s in seqs]


def assert_stats_close(z_gpu, p_gpu, z_cpu, p_cpu):
    """cuBLAS and OpenBLAS sum in different orders, so float32 results differ slightly.

    p is compared on the -log10 scale: its relative error grows with |z|.
    """
    np.testing.assert_allclose(z_gpu, z_cpu, rtol=1e-4, atol=1e-5)
    with np.errstate(divide="ignore"):   # p underflows to 0 for huge |z| (on both)
        np.testing.assert_allclose(-np.log10(p_gpu), -np.log10(p_cpu), rtol=1e-3, atol=1e-5)


def test_gpu_matches_cpu_toy(toy):
    model = load_model(toy["model"])
    refs, alts = random_pairs(np.random.default_rng(3), 500, 21)
    a, r = kmers(alts, 3), kmers(refs, 3)
    z_cpu, p_cpu = compute_statistics(a, r, model.partial, toy["cov_matrix"], use_gpu=False)
    z_gpu, p_gpu = compute_statistics(a, r, model.partial, toy["cov_matrix"], use_gpu=True, chunk_size=128)
    assert_stats_close(z_gpu, p_gpu, z_cpu, p_cpu)


def test_gpu_matches_cpu_example_model(example_covs):
    name = "ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer"
    model = load_model(EXAMPLE_MODELS / f"{name}.weights.qbic")
    cov = load_covariance(example_covs / f"{name}.cov_8192.npy", expected_dim=len(model.partial))
    refs, alts = random_pairs(np.random.default_rng(4), 2000, 21)
    a, r = kmers(alts, 7), kmers(refs, 7)
    z_cpu, p_cpu = compute_statistics(a, r, model.partial, cov, use_gpu=False)
    z_gpu, p_gpu = compute_statistics(a, r, model.partial, cov, use_gpu=True)
    assert_stats_close(z_gpu, p_gpu, z_cpu, p_cpu)
