"""Download and unpack the model collections from Zenodo, verifying checksums."""

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# The Zenodo record this version of the software was released with. `--record latest`
# follows the concept DOI to the newest version instead.
ZENODO_RECORD = "23482275"   # version 2: the paper's 997 primary + 243 secondary models
ZENODO_CONCEPT_RECORD = "15891510"
ZENODO_API = "https://zenodo.org/api/records/"

DATASETS = {
    "primary-weights": ("models_primary.tar.zst", "primary-collection model files (weights)"),
    "secondary-weights": ("models_secondary.tar.zst", "secondary-collection model files (weights)"),
    "examples": ("example_covs.tar.zst", "example models, inputs and covariances, to try QBiC-SELEX"),
    "example-covs": ("example_covs.tar.zst", "covariances for the two example models"),
    "primary-covs": ("covs_primary.tar.zst", "covariances for the primary models"),
    "secondary-covs": ("covs_secondary.tar.zst", "covariances for the secondary models"),
}

# Index of every covariance file inside the covariance archives (archive, byte offset and
# length of the member's own zstd frame, md5), used to fetch single models.
INDEX_KEY = "covs_index.csv"

_CHUNK = 8 * 1024 * 1024


@dataclass
class RemoteFile:
    key: str
    size: int
    md5: str
    url: str


def _urlopen(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "qbic-selex", **(headers or {})})
    return urllib.request.urlopen(req, timeout=60)


def fetch_record(record: str = ZENODO_RECORD) -> dict:
    """Zenodo record metadata. ``latest`` resolves the concept record to its newest version."""
    rec = ZENODO_CONCEPT_RECORD if record == "latest" else record
    with _urlopen(ZENODO_API + rec) as r:
        return json.load(r)


def record_files(record_json: dict) -> dict[str, RemoteFile]:
    files = {}
    for f in record_json["files"]:
        algo, _, digest = f["checksum"].partition(":")
        files[f["key"]] = RemoteFile(f["key"], int(f["size"]), digest if algo == "md5" else "",
                                     f["links"]["self"])
    return files


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_file(remote: RemoteFile, dest: Path) -> None:
    """Download to ``dest``, resuming a partial ``dest.part``, then check size and md5."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > remote.size:
        part.unlink()
        have = 0
    if have < remote.size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with _urlopen(remote.url, headers) as r:
            if have and r.status != 206:      # server ignored the range: start over
                have = 0
            mode = "ab" if have else "wb"
            last = time.time()
            with open(part, mode) as out:
                while chunk := r.read(_CHUNK):
                    out.write(chunk)
                    have += len(chunk)
                    if time.time() - last > 30:
                        log.info("  %s: %.2f / %.2f GB", remote.key, have / 1e9, remote.size / 1e9)
                        last = time.time()
    if part.stat().st_size != remote.size:
        raise OSError(f"{remote.key}: downloaded {part.stat().st_size} bytes, expected {remote.size}")
    if remote.md5:
        got = md5sum(part)
        if got != remote.md5:
            part.unlink()
            raise OSError(f"{remote.key}: md5 {got} does not match Zenodo's {remote.md5}; "
                          f"the partial download was removed, please retry")
    part.rename(dest)


def _safe_members(tar: tarfile.TarFile, dest: Path):
    """Yield regular files and directories that stay inside ``dest``, with a fresh mtime.

    A fresh mtime matters on scratch filesystems that purge files by age: the
    archives keep their 2025 timestamps otherwise.
    """
    root = dest.resolve()
    now = time.time()
    for m in tar:
        target = (dest / m.name).resolve()
        if not (m.isfile() or m.isdir()) or (target != root and root not in target.parents):
            log.warning("skipping unexpected archive member %s", m.name)
            continue
        m.mtime = now
        yield m


def extract_tar_zst(archive: Path, dest: Path) -> int:
    import zstandard

    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    # read_across_frames: archives may hold one zstd frame per member (for single-model fetches)
    dctx = zstandard.ZstdDecompressor()
    with open(archive, "rb") as fh, dctx.stream_reader(fh, read_across_frames=True) as zr, \
            tarfile.open(fileobj=zr, mode="r|") as tar:
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        for member in _safe_members(tar, dest):
            tar.extract(member, dest, **kwargs)
            n += member.isfile()
    return n


def _marker(dest: Path, key: str) -> Path:
    return dest / f".{key}.qbic-selex-download.json"


def dataset_files(name: str, files: dict[str, RemoteFile]) -> list[RemoteFile]:
    """The record files making up a dataset: ``stem.tar.zst``, or its parts ``stem.partNN.tar.zst``."""
    key = DATASETS[name][0]
    if key in files:
        return [files[key]]
    stem = key.removesuffix(".tar.zst")
    parts = sorted(k for k in files if re.fullmatch(re.escape(stem) + r"\.part\d+\.tar\.zst", k))
    return [files[k] for k in parts]


def _download_one(remote: RemoteFile, dest: Path, keep_archive: bool, record_id) -> None:
    marker = _marker(dest, remote.key)
    if marker.exists() and json.loads(marker.read_text()).get("md5") == remote.md5:
        log.info("%s already downloaded and unpacked in %s", remote.key, dest)
        return
    need = remote.size * (2.05 if keep_archive else 2)
    free = shutil.disk_usage(dest).free
    if free < need:
        log.warning("%s needs about %.1f GB (archive + unpacked); %.1f GB free in %s",
                    remote.key, need / 1e9, free / 1e9, dest)
    archive = dest / remote.key
    if archive.exists() and remote.md5 and md5sum(archive) == remote.md5:
        log.info("Using existing %s", archive)
    else:
        log.info("Downloading %s (%.2f GB) from Zenodo record %s", remote.key, remote.size / 1e9, record_id)
        download_file(remote, archive)
    log.info("Checksum OK; unpacking into %s", dest)
    n = extract_tar_zst(archive, dest)
    marker.write_text(json.dumps({"key": remote.key, "md5": remote.md5, "record": record_id,
                                  "files": n, "unpacked": time.strftime("%Y-%m-%d %H:%M:%S")}))
    if not keep_archive:
        os.remove(archive)
    log.info("Unpacked %d files from %s", n, remote.key)


def copy_example_files(dest: Path) -> list[Path]:
    """Copy the example models and inputs bundled with the package into ``dest``."""
    from importlib.resources import files

    written = []

    def walk(node, rel):
        for child in node.iterdir():
            if child.is_dir():
                walk(child, rel / child.name)
            elif not child.name.startswith((".", "__")):
                target = dest / rel / child.name
                target.parent.mkdir(parents=True, exist_ok=True)
                data = child.read_bytes()
                if not (target.exists() and target.read_bytes() == data):
                    target.write_bytes(data)
                written.append(target)

    walk(files("qbic_selex") / "examples", Path())
    return written


def download_dataset(name: str, dest: Path, record: str = ZENODO_RECORD, keep_archive: bool = False,
                     record_json: dict | None = None) -> Path:
    """Download one dataset (see DATASETS) into ``dest`` and unpack it. Returns ``dest``.

    ``examples`` = the example models and inputs shipped with the package, plus the
    example covariances from Zenodo.
    """
    if name == "examples":
        dest.mkdir(parents=True, exist_ok=True)
        n = len(copy_example_files(dest))
        log.info("Wrote %d example files (models, inputs, lists) into %s", n, dest)
        name = "example-covs"
    record_json = record_json or fetch_record(record)
    remotes = dataset_files(name, record_files(record_json))
    if not remotes:
        raise ValueError(f"{DATASETS[name][0]} is not in Zenodo record {record_json.get('id', record)}")
    dest.mkdir(parents=True, exist_ok=True)
    for remote in remotes:
        _download_one(remote, dest, keep_archive, record_json.get("id", record))
    return dest


# ----------------------------------------------------------------------------
# Single models from the covariance archives
# ----------------------------------------------------------------------------

def _dataset_stem(name: str) -> str:
    return DATASETS[name][0].removesuffix(".tar.zst")


def fetch_index(files: dict[str, RemoteFile]) -> list[dict]:
    if INDEX_KEY not in files:
        raise ValueError(f"this Zenodo record has no {INDEX_KEY}, so single models cannot be fetched; "
                         f"download the whole dataset, or use a newer record (--record latest)")
    with _urlopen(files[INDEX_KEY].url) as r:
        return list(csv.DictReader(io.TextIOWrapper(r, encoding="utf-8")))


def _fetch_member(remote: RemoteFile, offset: int, length: int) -> bytes:
    """One archive member: an HTTP range request for its zstd frame, decompressed."""
    import zstandard

    with _urlopen(remote.url, {"Range": f"bytes={offset}-{offset + length - 1}"}) as r:
        if getattr(r, "status", 206) != 206:
            raise OSError(f"{remote.key}: server ignored the byte-range request (HTTP {r.status})")
        frame = r.read()
    if len(frame) != length:
        raise OSError(f"{remote.key}: got {len(frame)} bytes, expected {length}")
    return zstandard.ZstdDecompressor().decompress(frame)


def download_models(name: str, model_ids: list[str], dest: Path, record: str = ZENODO_RECORD,
                    record_json: dict | None = None) -> list[Path]:
    """Fetch the covariance files of selected models from a covariance dataset.

    Each file is read from its own frame inside the archive (HTTP range request),
    so only the requested models are downloaded. Files are checked against the
    md5 in the index; files already present with the right md5 are skipped.
    """
    if not name.endswith("-covs"):
        raise ValueError("single models can be fetched only from covariance datasets "
                         "(the weights archives are small; download them whole)")
    record_json = record_json or fetch_record(record)
    files = record_files(record_json)
    stem = _dataset_stem(name)
    rows = {}
    for row in fetch_index(files):
        if row["archive"].startswith(stem + "."):
            rows.setdefault(row["model_id"], row)
    wanted = list(dict.fromkeys(model_ids))
    missing = [m for m in wanted if m not in rows]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ValueError(f"{len(missing)} model(s) not in {name}: {shown}")

    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written = []
    for i, m in enumerate(wanted, 1):
        row = rows[m]
        target = (dest / row["member"]).resolve()
        if root not in target.parents:
            raise OSError(f"unsafe path in index: {row['member']}")
        if target.exists() and md5sum(target) == row["md5"]:
            log.info("[%d/%d] %s already present", i, len(wanted), m)
            written.append(target)
            continue
        log.info("[%d/%d] %s (%.0f MB)", i, len(wanted), m, int(row["length"]) / 1e6)
        member = _fetch_member(files[row["archive"]], int(row["offset"]), int(row["length"]))
        with tarfile.open(fileobj=io.BytesIO(member)) as tar:
            info = tar.next()
            data = tar.extractfile(info).read()
        if info.name != row["member"] or hashlib.md5(data).hexdigest() != row["md5"]:
            raise OSError(f"{m}: downloaded file does not match the index (name or md5)")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        tmp.rename(target)
        written.append(target)
    return written
