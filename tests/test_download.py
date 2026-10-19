"""qbic-selex download, against a fake Zenodo record served from local files (no network)."""

import hashlib
import io
import json
import tarfile

import pytest
import zstandard

from qbic_selex import download
from qbic_selex.cli import main


def make_archive(path, members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 1_600_000_000          # 2020: must not survive extraction
            tar.addfile(info, io.BytesIO(data))
    path.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    return path


def fake_record(archive, md5=None):
    data = archive.read_bytes()
    return {"id": "999", "doi": "10.5281/zenodo.999", "files": [{
        "key": archive.name, "size": len(data),
        "checksum": "md5:" + (md5 or hashlib.md5(data).hexdigest()),
        "links": {"self": archive.as_uri()}}]}


@pytest.fixture
def archive(tmp_path, toy):
    src = tmp_path / "src"
    src.mkdir()
    return make_archive(src / "models_primary.tar.zst", {
        "tf_models_primary_collection/ols_models/TOY/TOY_A_test_0_1_3mer.weights.qbic":
            toy["model"].read_bytes(),
    })


def test_download_unpacks_and_verifies(tmp_path, archive):
    dest = tmp_path / "data"
    download.download_dataset("primary", dest, record_json=fake_record(archive))
    model = dest / "tf_models_primary_collection/ols_models/TOY/TOY_A_test_0_1_3mer.weights.qbic"
    assert model.exists()
    assert model.stat().st_mtime > 1_700_000_000          # fresh mtime, not the archive's
    assert not (dest / archive.name).exists()             # archive removed by default
    # second run is a no-op (marker with the same md5)
    model.unlink()
    download.download_dataset("primary", dest, record_json=fake_record(archive))
    assert not model.exists()


def test_checksum_mismatch_is_an_error(tmp_path, archive):
    with pytest.raises(OSError, match="md5"):
        download.download_dataset("primary", tmp_path / "data",
                                  record_json=fake_record(archive, md5="0" * 32))
    assert not list((tmp_path / "data").glob("*.part"))   # bad partial download removed


def test_partial_download_is_completed(tmp_path, archive):
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / (archive.name + ".part")).write_bytes(archive.read_bytes()[:10])
    download.download_dataset("primary", dest, record_json=fake_record(archive), keep_archive=True)
    assert (dest / archive.name).read_bytes() == archive.read_bytes()


def test_unsafe_members_are_skipped(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    bad = make_archive(src / "models_primary.tar.zst", {"../escape.txt": b"x", "ok/fine.txt": b"y"})
    dest = tmp_path / "data"
    download.download_dataset("primary", dest, record_json=fake_record(bad))
    assert (dest / "ok/fine.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_cli_list_and_unknown_dataset(monkeypatch, archive, capsys):
    monkeypatch.setattr(download, "fetch_record", lambda record: fake_record(archive))
    assert main(["download", "--list"]) == 0
    assert "primary" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["download", "nonsense"])


def test_record_files_parses_zenodo_json():
    rec = {"files": [{"key": "a.tar.zst", "size": 3, "checksum": "md5:abc", "links": {"self": "u"}}]}
    f = download.record_files(rec)["a.tar.zst"]
    assert (f.size, f.md5, f.url) == (3, "abc", "u")
    assert json.dumps(rec)
