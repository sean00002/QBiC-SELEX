import os
import pickle

import numpy as np
import pytest

from qbic_selex.kmers import recover_non_rc_dict, reverse_complement, sliding_window
from qbic_selex.models import (ModelInputError, load_covariance, load_model, model_name_from_path,
                               resolve_covariances, resolve_models)

from .conftest import EXAMPLE_MODELS, write_cov


def test_reverse_complement():
    assert reverse_complement("AACGT") == "ACGTT"
    assert reverse_complement("GATC") == "GATC"


def test_sliding_window():
    assert sliding_window("ACGTA", 3) == ["ACG", "CGT", "GTA"]
    assert sliding_window("AC", 3) == []


def test_recover_non_rc_dict_covers_all_kmers(toy):
    complete = recover_non_rc_dict(toy["coefs"])
    assert len(complete) == 4 ** 3
    for kmer, v in complete.items():
        assert complete[reverse_complement(kmer)] == v


def test_recover_handles_palindromes():
    # even k: palindromes (e.g. ACGT) are stored once
    partial = {"ACGT": 1.0, "TTTT": 2.0}
    complete = recover_non_rc_dict(partial)
    assert complete == {"ACGT": 1.0, "TTTT": 2.0, "AAAA": 2.0}


@pytest.mark.parametrize("path, name", [
    ("dir/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic",
     "ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer"),
    ("COUP-TF2_NA_Lib1_NA_bhimsaria2023_0(2)_2_7mer.cov_8192.npy",
     "COUP-TF2_NA_Lib1_NA_bhimsaria2023_0(2)_2_7mer"),
    ("X.v2_0_1_7mer.weights.qbic", "X.v2_0_1_7mer"),   # dots inside a name survive
])
def test_model_name_from_path(path, name):
    assert model_name_from_path(path) == name


def test_load_example_model():
    m = load_model(EXAMPLE_MODELS / "ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic")
    assert m.kmer_size == 7
    assert len(m.partial) == 8192
    assert len(m.complete) == 4 ** 7


def test_restricted_unpickler_rejects_code(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    bad = tmp_path / "evil.weights.qbic"
    bad.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError):
        load_model(bad)


def test_load_covariance_roundtrip(toy):
    cov = load_covariance(toy["cov"], expected_dim=32)
    np.testing.assert_array_equal(cov, toy["cov_matrix"])


def test_covariance_dimension_mismatch(toy):
    with pytest.raises(ModelInputError, match="32x32 but the model has 8192"):
        load_covariance(toy["cov"], expected_dim=8192)


def test_covariance_truncated_file(tmp_path):
    p = tmp_path / "M.cov_10.npy"
    np.save(p, np.zeros(7, dtype=np.float32))
    with pytest.raises(ModelInputError, match="upper-triangle"):
        load_covariance(p)


def test_resolve_models_file_dir_list_and_names(toy, tmp_path):
    assert resolve_models(str(toy["model"])) == [toy["model"]]
    assert resolve_models(str(toy["dir"])) == sorted([toy["model"], toy["model_b"]])

    # list file with a path relative to the list file
    lst = toy["dir"] / "models.txt"
    lst.write_text(f"# comment\n{toy['model'].name}\n")
    assert resolve_models(str(lst)) == [toy["dir"] / toy["model"].name]

    # bare names need --model-dir
    names = tmp_path / "names.txt"
    names.write_text("TOY_B_test_0_1_3mer\n")
    with pytest.raises(ModelInputError, match="--model-dir"):
        resolve_models(str(names))
    assert resolve_models(str(names), model_dir=str(toy["dir"])) == [toy["model_b"]]


def test_resolve_covariances(toy, tmp_path):
    names = ["TOY_A_test_0_1_3mer", "TOY_B_test_0_1_3mer"]
    by_dir = resolve_covariances(str(toy["dir"]), names)
    assert by_dir == {names[0]: toy["cov"], names[1]: toy["cov_b"]}

    pairs = tmp_path / "covs.txt"
    pairs.write_text(f"{toy['model']},{toy['cov']}\n{toy['cov_b']}\n")
    assert resolve_covariances(str(pairs), names) == by_dir

    with pytest.raises(ModelInputError, match="several models"):
        resolve_covariances(str(toy["cov"]), names)

    with pytest.raises(ModelInputError, match="no covariance"):
        resolve_covariances(str(toy["dir"]), [*names, "MISSING_0_1_3mer"])


def test_cov_file_naming_enforced(tmp_path):
    p = tmp_path / "M.npy"
    write_cov(p, np.eye(3, dtype=np.float32))
    with pytest.raises(ModelInputError, match="cov_<n>"):
        load_covariance(p)
