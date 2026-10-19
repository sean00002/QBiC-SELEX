"""Download and unpack the model collections from Zenodo, verifying checksums."""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# The Zenodo record this version of the software was released with. `--record latest`
# follows the concept DOI to the newest version instead.
ZENODO_RECORD = "15891511"
ZENODO_CONCEPT_RECORD = "15891510"
ZENODO_API = "https://zenodo.org/api/records/"

DATASETS = {
    "primary": ("models_primary.tar.zst", "primary-collection model files"),
    "secondary": ("models_secondary.tar.zst", "secondary-collection model files"),
    "example-covs": ("example_covs.tar.zst", "covariances for the two example models"),
    "primary-covs": ("covs_primary.tar.zst", "covariances for the primary models"),
    "secondary-covs": ("covs_secondary.tar.zst", "covariances for the secondary models"),
}

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
    with open(archive, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr, \
            tarfile.open(fileobj=zr, mode="r|") as tar:
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        for member in _safe_members(tar, dest):
            tar.extract(member, dest, **kwargs)
            n += member.isfile()
    return n


def _marker(dest: Path, key: str) -> Path:
    return dest / f".{key}.qbic-selex-download.json"


def download_dataset(name: str, dest: Path, record: str = ZENODO_RECORD, keep_archive: bool = False,
                     record_json: dict | None = None) -> Path:
    """Download one dataset (see DATASETS) into ``dest`` and unpack it. Returns ``dest``."""
    key = DATASETS[name][0]
    record_json = record_json or fetch_record(record)
    files = record_files(record_json)
    if key not in files:
        raise ValueError(f"{key} is not in Zenodo record {record_json.get('id', record)}")
    remote = files[key]

    dest.mkdir(parents=True, exist_ok=True)
    marker = _marker(dest, key)
    if marker.exists() and json.loads(marker.read_text()).get("md5") == remote.md5:
        log.info("%s already downloaded and unpacked in %s", key, dest)
        return dest

    need = remote.size * (2 if not keep_archive else 2.05)
    free = shutil.disk_usage(dest).free
    if free < need:
        log.warning("%s needs about %.1f GB (archive + unpacked); %.1f GB free in %s",
                    key, need / 1e9, free / 1e9, dest)

    archive = dest / key
    if archive.exists() and remote.md5 and md5sum(archive) == remote.md5:
        log.info("Using existing %s", archive)
    else:
        log.info("Downloading %s (%.2f GB) from Zenodo record %s", key, remote.size / 1e9,
                 record_json.get("id", record))
        download_file(remote, archive)
    log.info("Checksum OK; unpacking into %s", dest)
    n = extract_tar_zst(archive, dest)
    marker.write_text(json.dumps({"key": key, "md5": remote.md5, "record": record_json.get("id"),
                                  "files": n, "unpacked": time.strftime("%Y-%m-%d %H:%M:%S")}))
    if not keep_archive:
        os.remove(archive)
    log.info("Unpacked %d files from %s", n, key)
    return dest
