import subprocess
import sys

import pandas as pd
import pytest

from qbic_selex.cli import main

from .conftest import GENOME_SEQS, REPO

CHR1 = GENOME_SEQS["chr1"]


@pytest.fixture
def seq_input(tmp_path):
    p = tmp_path / "seqs.csv"
    pd.DataFrame({
        "id": ["a", "b", "c"],
        "ref_sequence": ["ACGTTGCAAC", "GGCATTACGA", "ACGTNGCAAC"],
        "alt_sequence": ["ACGTAGCAAC", "GGCAGTACGA", "ACGTAGCAAC"],
    }).to_csv(p, index=False)
    return p


@pytest.fixture
def var_input(tmp_path):
    p = tmp_path / "vars.tsv"
    pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "pos": [21, 30, 40],
        "ref": [CHR1[20], CHR1[29], "N" if CHR1[39] != "N" else "A"],   # last one mismatches
        "alt": ["T" if CHR1[20] != "T" else "A", "G" if CHR1[29] != "G" else "A", "C"],
    }).to_csv(p, sep="\t", index=False)
    return p


def run(*args):
    return main(["--quiet", *map(str, args)])


def test_predict_sequences_batch(toy, seq_input, tmp_path):
    out = tmp_path / "out.csv"
    assert run("predict", "-v", seq_input, "-m", toy["dir"], "-o", out) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["id", "ref_sequence", "alt_sequence", "model", "predicted_effect",
                                "qbic_status"]
    assert len(df) == 6
    assert df.qbic_status.tolist().count("contains_N") == 2


def test_predict_with_stats_pval_and_short_output(toy, seq_input, tmp_path):
    full = tmp_path / "full.csv"
    assert run("predict", "-v", seq_input, "-m", toy["dir"], "-c", toy["dir"], "--compute-stats",
               "--device", "cpu", "-o", full) == 0
    df = pd.read_csv(full)
    assert {"z_score", "p_value"} <= set(df.columns)
    threshold = float(df.p_value.dropna().median())

    filtered = tmp_path / "filtered.csv"
    assert run("predict", "-v", seq_input, "-m", toy["dir"], "-c", toy["dir"], "--compute-stats",
               "--device", "cpu", "--pval", threshold, "-o", filtered) == 0
    kept = pd.read_csv(filtered)
    assert len(kept) == (df.p_value <= threshold).sum()
    assert (kept.p_value <= threshold).all()

    mm = tmp_path / "map.csv"
    lst = tmp_path / "models.txt"
    lst.write_text(f"{toy['model']}\n{toy['model_b']}\n")
    assert run("model-map", "-m", lst, "-o", mm) == 0
    short = tmp_path / "short.csv"
    assert run("predict", "-v", seq_input, "-m", lst, "-c", toy["dir"], "--compute-stats",
               "--device", "cpu", "--short-output", "--model-map", mm, "-o", short) == 0
    s = pd.read_csv(short)
    assert list(s.columns) == ["model", "predicted_effect", "z_score", "p_value"]
    assert sorted(s.model.unique()) == [0, 1]
    assert len(s) == 4   # N rows are not in the compact format


def test_predict_variants_and_output_dir(toy, genome, var_input, tmp_path):
    outdir = tmp_path / "per_model"
    assert run("predict", "-v", var_input, "-m", toy["dir"], "-g", genome, "--context-length", 5,
               "--output-dir", "-o", outdir) == 0
    files = sorted(p.name for p in outdir.glob("*.csv"))
    assert files == ["TOY_A_test_0_1_3mer.csv", "TOY_B_test_0_1_3mer.csv"]
    df = pd.read_csv(outdir / files[0])
    assert df.qbic_status.tolist() == ["ok", "ok", "ref_mismatch"]
    assert df.predicted_effect.isna().tolist() == [False, False, True]


def test_stats_failure_fails_the_run(toy, seq_input, tmp_path):
    # TOY_B gets a covariance of the wrong size
    import numpy as np
    covs = tmp_path / "covs"
    covs.mkdir()
    (covs / toy["cov"].name).symlink_to(toy["cov"])
    np.save(covs / "TOY_B_test_0_1_3mer.cov_3.npy", np.ones(6, dtype=np.float32))
    out = tmp_path / "out" / "res.csv"
    args = ["predict", "-v", seq_input, "-m", toy["dir"], "-c", covs, "--compute-stats", "--device", "cpu",
            "-o", out]
    assert run(*args) == 1
    df = pd.read_csv(out)
    assert set(df.model) == {"TOY_A_test_0_1_3mer"}         # failed model left out, columns consistent
    assert {"z_score", "p_value"} <= set(df.columns)
    assert len(list(out.parent.glob("qbic_error_report_*.log"))) == 1
    assert run(*args, "--allow-failures") == 0


@pytest.mark.parametrize("args, message", [
    (["--compute-stats"], "requires --covs"),
    (["--pval", "0.01"], "requires --compute-stats"),
    (["--short-output"], "must be used together"),
])
def test_argument_validation(toy, seq_input, tmp_path, args, message, capsys):
    with pytest.raises(SystemExit, match=message):
        run("predict", "-v", seq_input, "-m", toy["model"], "-o", tmp_path / "x.csv", *args)


def test_missing_models_exit_code(seq_input, tmp_path):
    assert run("predict", "-v", seq_input, "-m", tmp_path / "nope", "-o", tmp_path / "x.csv") == 2


def test_extract(genome, var_input, tmp_path):
    out = tmp_path / "ext.csv"
    assert run("extract", "-v", var_input, "-g", genome, "--context-length", 3, "-o", out) == 0
    df = pd.read_csv(out)
    assert df.ref_sequence.iloc[0] == CHR1[17:24]


def test_legacy_wrapper_and_flags(toy, seq_input, tmp_path):
    """qbic_predict.py with the pre-1.0 flags still works."""
    out = tmp_path / "legacy.csv"
    r = subprocess.run([sys.executable, str(REPO / "qbic_predict.py"), "-v", str(seq_input),
                        "--model", str(toy["model"]), "--cov-file", str(toy["cov"]), "--compute-stats",
                        "--use-cpu", "--n-jobs", "4", "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "no effect" in r.stderr          # --n-jobs is accepted but ignored
    assert len(pd.read_csv(out)) == 3
