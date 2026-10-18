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


def test_gpu_matches_cpu_toy(toy):
    model = load_model(toy["model"])
    refs, alts = random_pairs(np.random.default_rng(3), 500, 21)
    a, r = kmers(alts, 3), kmers(refs, 3)
    z_cpu, p_cpu = compute_statistics(a, r, model.partial, toy["cov_matrix"], use_gpu=False)
    z_gpu, p_gpu = compute_statistics(a, r, model.partial, toy["cov_matrix"], use_gpu=True, chunk_size=128)
    np.testing.assert_allclose(z_gpu, z_cpu, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(p_gpu, p_cpu, rtol=1e-4, atol=1e-10)


def test_gpu_matches_cpu_example_model(example_covs):
    name = "ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer"
    model = load_model(EXAMPLE_MODELS / f"{name}.weights.qbic")
    cov = load_covariance(example_covs / f"{name}.cov_8192.npy", expected_dim=len(model.partial))
    refs, alts = random_pairs(np.random.default_rng(4), 2000, 21)
    a, r = kmers(alts, 7), kmers(refs, 7)
    z_cpu, p_cpu = compute_statistics(a, r, model.partial, cov, use_gpu=False)
    z_gpu, p_gpu = compute_statistics(a, r, model.partial, cov, use_gpu=True)
    np.testing.assert_allclose(z_gpu, z_cpu, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p_gpu, p_cpu, rtol=1e-3, atol=1e-10)
