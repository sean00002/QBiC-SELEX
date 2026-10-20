"""Shared fixtures: a toy 3-mer model with a covariance matrix, and a tiny genome."""

import itertools
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from qbic_selex.kmers import reverse_complement

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_MODELS = REPO / "example_models"


def canonical_kmers(k):
    """One k-mer per reverse-complement pair, stored as the larger of the two (like the real files)."""
    seen, out = set(), []
    for kmer in ("".join(p) for p in itertools.product("ACGT", repeat=k)):
        rc = reverse_complement(kmer)
        if kmer in seen or rc in seen:
            continue
        seen.add(kmer)
        out.append(max(kmer, rc))
    return out


def write_model(path, coefs: dict):
    with open(path, "wb") as f:
        pickle.dump({k: np.float32(v) for k, v in coefs.items()}, f)


def write_cov(path, cov: np.ndarray):
    np.save(path, cov[np.triu_indices(cov.shape[0])].astype(np.float32))


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """A 3-mer model (32 canonical k-mers) with a random SPD covariance."""
    d = tmp_path_factory.mktemp("toy")
    rng = np.random.default_rng(0)
    kmers = canonical_kmers(3)
    coefs = dict(zip(kmers, rng.normal(0, 1, len(kmers)), strict=True))
    a = rng.normal(0, 0.1, (len(kmers), len(kmers)))
    cov = (a @ a.T + 0.01 * np.eye(len(kmers))).astype(np.float32)
    model_path = d / "TOY_A_test_0_1_3mer.weights.qbic"
    cov_path = d / f"TOY_A_test_0_1_3mer.cov_{len(kmers)}.npy"
    write_model(model_path, coefs)
    write_cov(cov_path, cov)

    # a second model so batch mode has something to iterate over
    coefs_b = dict(zip(kmers, rng.normal(0, 1, len(kmers)), strict=True))
    model_b = d / "TOY_B_test_0_1_3mer.weights.qbic"
    cov_b = d / f"TOY_B_test_0_1_3mer.cov_{len(kmers)}.npy"
    write_model(model_b, coefs_b)
    write_cov(cov_b, cov * 2)

    return {
        "dir": d,
        "model": model_path, "cov": cov_path, "coefs": coefs, "cov_matrix": cov,
        "model_b": model_b, "cov_b": cov_b,
        "kmers": kmers,
    }


GENOME_SEQS = {
    "chr1": "ACGTACGTTAGCCGATAGGCTTACGATCGATGCATGCAAGCTTGACCTAGGATCCGATAA",   # 60 bp
    "chr2": "TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAANNNNNACGTACGTACGTACGTACGT",     # 57 bp, with Ns
}


@pytest.fixture(scope="session")
def genome(tmp_path_factory):
    path = tmp_path_factory.mktemp("genome") / "toy.fa"
    with open(path, "w") as f:
        for name, seq in GENOME_SEQS.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 20):
                f.write(seq[i:i + 20] + "\n")
    return path


@pytest.fixture(scope="session")
def example_covs():
    d = os.environ.get("QBIC_SELEX_EXAMPLE_COVS")
    if not d or not Path(d).is_dir():
        pytest.skip("set QBIC_SELEX_EXAMPLE_COVS to the unpacked example_covs/ directory")
    return Path(d)


@pytest.fixture(scope="session")
def hg38():
    g = os.environ.get("QBIC_SELEX_GENOME")
    if not g or not Path(g).exists():
        pytest.skip("set QBIC_SELEX_GENOME to an hg38 FASTA")
    return Path(g)
