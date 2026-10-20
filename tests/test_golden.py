"""Re-run the example scenarios with the real example models and compare with tests/data/golden.

See tests/data/golden/README.md for how the baselines were produced. Scenarios
that need the example covariances or hg38 are skipped unless
QBIC_SELEX_EXAMPLE_COVS / QBIC_SELEX_GENOME are set.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qbic_selex.cli import main

from .conftest import REPO

GOLDEN = REPO / "tests" / "data" / "golden"
MODELS = REPO / "example_models_list.txt"
ETV4 = "ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer"

_spec = importlib.util.spec_from_file_location("compare_outputs", REPO / "scripts" / "compare_outputs.py")
compare_outputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_outputs)


def assert_matches_golden(expected: pd.DataFrame, got: pd.DataFrame, rounded=False):
    """Everything except z/p must match (effects are plain float32 sums, identical everywhere).

    z and p come from BLAS matrix products, whose float32 rounding depends on the
    CPU kernel and BLAS build (measured: z within 4e-7 relative, -log10 p within
    5e-5). With --short-output (4 significant digits) that can flip the last digit.
    """
    problems = compare_outputs.compare_frames(expected, got, rtol=1e-6, atol=1e-9,
                                              ignore={"z_score", "p_value"})
    assert not problems, "\n".join(problems)
    if "z_score" not in expected:
        return
    z_e, z_g = expected.z_score.to_numpy(float), got.z_score.to_numpy(float)
    p_e, p_g = expected.p_value.to_numpy(float), got.p_value.to_numpy(float)
    if rounded:   # one unit in the 4th significant digit
        np.testing.assert_allclose(z_g, z_e, rtol=1.1e-3, equal_nan=True)
        np.testing.assert_allclose(p_g, p_e, rtol=1.1e-3, equal_nan=True)
    else:
        np.testing.assert_allclose(z_g, z_e, rtol=1e-5, atol=1e-5, equal_nan=True)
        with np.errstate(divide="ignore"):
            np.testing.assert_allclose(-np.log10(p_g), -np.log10(p_e), rtol=0, atol=1e-4, equal_nan=True)


def run_and_compare(name, args, tmp_path, rounded=False):
    out = tmp_path / f"{name}.csv"
    assert main(["--quiet", "predict", *map(str, args), "--device", "cpu", "-o", str(out)]) == 0
    assert_matches_golden(pd.read_csv(GOLDEN / f"{name}.csv"), pd.read_csv(out), rounded=rounded)


def test_A_sequences_no_stats(tmp_path):
    run_and_compare("A_seq_nostats", ["-v", REPO / "example_input_sequences.csv", "-m", MODELS], tmp_path)


def test_B_sequences_stats(tmp_path, example_covs):
    run_and_compare("B_seq_stats", ["-v", REPO / "example_input_sequences.csv", "-m", MODELS,
                                    "-c", example_covs, "--compute-stats"], tmp_path)


def test_C_sequences_wildcard_stats(tmp_path, example_covs):
    run_and_compare("C_seq_wildcard_stats", ["-v", REPO / "example_input_sequences.csv", "-m", MODELS,
                                             "-c", example_covs, "--compute-stats", "--wildcard", "A"],
                    tmp_path)


def test_D_variants_stats(tmp_path, example_covs, hg38):
    run_and_compare("D_var_stats", ["-v", REPO / "example_input_variants.csv", "-m", MODELS,
                                    "-c", example_covs, "--compute-stats", "-g", hg38], tmp_path)


def test_D_zero_based_equivalent(tmp_path, example_covs, hg38):
    v = pd.read_csv(REPO / "example_input_variants.csv")
    zero = tmp_path / "zero.csv"
    v.assign(pos=v.start).drop(columns="start").to_csv(zero, index=False)
    out = tmp_path / "E.csv"
    assert main(["--quiet", "predict", "-v", str(zero), "-m", str(MODELS), "-c", str(example_covs),
                 "--compute-stats", "-g", str(hg38), "--zero-based", "--device", "cpu", "-o", str(out)]) == 0
    expected = pd.read_csv(GOLDEN / "D_var_stats.csv").drop(columns="pos").rename(columns={"start": "pos"})
    assert_matches_golden(expected, pd.read_csv(out))


def test_F_short_output_pval(tmp_path, example_covs, hg38):
    run_and_compare("F_var_short_pval", ["-v", REPO / "example_input_variants.csv", "-m", MODELS,
                                         "-c", example_covs, "--compute-stats", "-g", hg38,
                                         "--pval", "0.001", "--short-output",
                                         "--model-map", GOLDEN / "example_model_map.csv"],
                    tmp_path, rounded=True)


def test_H_single_model(tmp_path, example_covs, hg38):
    run_and_compare("H_single_stats", ["-v", REPO / "example_input_variants.csv",
                                       "-m", REPO / "example_models" / f"{ETV4}.weights.qbic",
                                       "-c", Path(example_covs) / f"{ETV4}.cov_8192.npy",
                                       "--compute-stats", "-g", hg38], tmp_path)
