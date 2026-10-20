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
    download.download_dataset("primary-weights", dest, record_json=fake_record(archive))
    model = dest / "tf_models_primary_collection/ols_models/TOY/TOY_A_test_0_1_3mer.weights.qbic"
    assert model.exists()
    assert model.stat().st_mtime > 1_700_000_000          # fresh mtime, not the archive's
    assert not (dest / archive.name).exists()             # archive removed by default
    # second run is a no-op (marker with the same md5)
    model.unlink()
    download.download_dataset("primary-weights", dest, record_json=fake_record(archive))
    assert not model.exists()


def test_checksum_mismatch_is_an_error(tmp_path, archive):
    with pytest.raises(OSError, match="md5"):
        download.download_dataset("primary-weights", tmp_path / "data",
                                  record_json=fake_record(archive, md5="0" * 32))
    assert not list((tmp_path / "data").glob("*.part"))   # bad partial download removed


def test_partial_download_is_completed(tmp_path, archive):
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / (archive.name + ".part")).write_bytes(archive.read_bytes()[:10])
    download.download_dataset("primary-weights", dest, record_json=fake_record(archive), keep_archive=True)
    assert (dest / archive.name).read_bytes() == archive.read_bytes()


def test_unsafe_members_are_skipped(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    bad = make_archive(src / "models_primary.tar.zst", {"../escape.txt": b"x", "ok/fine.txt": b"y"})
    dest = tmp_path / "data"
    download.download_dataset("primary-weights", dest, record_json=fake_record(bad))
    assert (dest / "ok/fine.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_cli_list_and_unknown_dataset(monkeypatch, archive, capsys):
    monkeypatch.setattr(download, "fetch_record", lambda record: fake_record(archive))
    assert main(["download", "--list"]) == 0
    out = capsys.readouterr().out
    assert "primary-weights" in out and "primary-covs" in out
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["download", "nonsense"])


def test_record_files_parses_zenodo_json():
    rec = {"files": [{"key": "a.tar.zst", "size": 3, "checksum": "md5:abc", "links": {"self": "u"}}]}
    f = download.record_files(rec)["a.tar.zst"]
    assert (f.size, f.md5, f.url) == (3, "abc", "u")
    assert json.dumps(rec)


def test_dataset_split_into_parts(tmp_path, toy):
    src = tmp_path / "src"
    src.mkdir()
    p1 = make_archive(src / "covs_primary.part01.tar.zst", {"covs_primary/A.cov_32.npy": b"a"})
    p2 = make_archive(src / "covs_primary.part02.tar.zst", {"covs_primary/B.cov_32.npy": b"b"})
    rec = fake_record(p1)
    rec["files"] += fake_record(p2)["files"]
    dest = tmp_path / "data"
    download.download_dataset("primary-covs", dest, record_json=rec)
    assert sorted(p.name for p in (dest / "covs_primary").iterdir()) == ["A.cov_32.npy", "B.cov_32.npy"]


# --- archives with one zstd frame per member, and single-model fetches -------------------------

def make_framed_archive(path, members):
    """Like the Zenodo covariance archives: each member is its own zstd frame. Returns index rows."""
    cctx = zstandard.ZstdCompressor(write_content_size=True)
    rows, out = [], io.BytesIO()
    for name, data in members.items():
        info = tarfile.TarInfo(name)
        info.size = len(data)
        frame = cctx.compress(info.tobuf(format=tarfile.PAX_FORMAT) + data + b"\0" * (-len(data) % 512))
        rows.append(dict(model_id=name.split("/")[-1].split(".cov_")[0], archive=path.name, member=name,
                         offset=out.tell(), length=len(frame), size=len(data),
                         md5=hashlib.md5(data).hexdigest()))
        out.write(frame)
    out.write(cctx.compress(b"\0" * 1024))
    path.write_bytes(out.getvalue())
    return rows


@pytest.fixture
def framed_record(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    data = {f"covs_primary/M{i}_0_1_7mer.cov_8.npy": bytes([i]) * (1000 + i) for i in range(5)}
    names = list(data)
    rows = make_framed_archive(src / "covs_primary.part01.tar.zst", {k: data[k] for k in names[:3]})
    rows += make_framed_archive(src / "covs_primary.part02.tar.zst", {k: data[k] for k in names[3:]})
    index = src / download.INDEX_KEY
    with open(index, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rec = fake_record(src / "covs_primary.part01.tar.zst")
    for p in (src / "covs_primary.part02.tar.zst", index):
        rec["files"] += fake_record(p)["files"]
    return rec, data


@pytest.fixture
def range_server(monkeypatch):
    """Serve file:// URLs honouring Range headers, like Zenodo does (HTTP 206)."""
    import urllib.parse

    class Resp(io.BytesIO):
        status = 206

    def fake_urlopen(url, headers=None):
        path = urllib.parse.urlparse(url).path
        data = open(path, "rb").read()
        rng = (headers or {}).get("Range")
        if rng:
            a, b = rng.removeprefix("bytes=").split("-")
            r = Resp(data[int(a):int(b) + 1])
            return r
        r = Resp(data)
        r.status = 200
        return r
    monkeypatch.setattr(download, "_urlopen", fake_urlopen)


def test_framed_archive_unpacks_whole(tmp_path, framed_record, range_server):
    """Regression: archives with several frames must unpack completely (not just the first member)."""
    rec, data = framed_record
    dest = tmp_path / "all"
    download.download_dataset("primary-covs", dest, record_json=rec)
    got = {str(p.relative_to(dest)): p.read_bytes() for p in dest.rglob("*.npy")}
    assert got == data


def test_fetch_individual_models(tmp_path, framed_record, range_server):
    rec, data = framed_record
    dest = tmp_path / "some"
    paths = download.download_models("primary-covs", ["M1_0_1_7mer", "M4_0_1_7mer"], dest, record_json=rec)
    assert sorted(p.name for p in paths) == ["M1_0_1_7mer.cov_8.npy", "M4_0_1_7mer.cov_8.npy"]
    for p in paths:
        assert p.read_bytes() == data["covs_primary/" + p.name]
    assert len(list(dest.rglob("*.npy"))) == 2           # nothing else downloaded
    # second call: already present, nothing re-fetched
    download.download_models("primary-covs", ["M1_0_1_7mer"], dest, record_json=rec)


def test_fetch_model_list_via_cli(tmp_path, framed_record, range_server, monkeypatch):
    rec, data = framed_record
    monkeypatch.setattr(download, "fetch_record", lambda record: rec)
    lst = tmp_path / "models.txt"
    lst.write_text("# my models\nM0_0_1_7mer\nsomewhere/M3_0_1_7mer.weights.qbic\n")
    dest = tmp_path / "cli"
    assert main(["--quiet", "download", "primary-covs", "--model-list", str(lst), "-o", str(dest)]) == 0
    assert sorted(p.name for p in dest.rglob("*.npy")) == ["M0_0_1_7mer.cov_8.npy", "M3_0_1_7mer.cov_8.npy"]


def test_fetch_unknown_model_and_corruption(tmp_path, framed_record, range_server):
    rec, data = framed_record
    with pytest.raises(ValueError, match="not in primary-covs"):
        download.download_models("primary-covs", ["NOPE_0_1_7mer"], tmp_path / "x", record_json=rec)
    # corrupt the md5 in the index -> the fetch must refuse the file
    idx = next(f for f in rec["files"] if f["key"] == download.INDEX_KEY)
    path = idx["links"]["self"].removeprefix("file://")
    rows = open(path).read().splitlines()
    rows[1] = ",".join(rows[1].split(",")[:-1] + ["0" * 32])
    open(path, "w").write("\n".join(rows) + "\n")
    with pytest.raises(OSError, match="md5"):
        download.download_models("primary-covs", ["M0_0_1_7mer"], tmp_path / "y", record_json=rec)
