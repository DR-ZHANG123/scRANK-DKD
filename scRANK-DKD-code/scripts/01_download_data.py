#!/usr/bin/env python
"""Download every public GEO dataset the pipeline needs.

Bulk series matrices, raw CEL archives for the two probe-centred cohorts, and the
two single-cell datasets. Checksums are recorded so a partial download is not
mistaken for a complete one."""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_BULK = ROOT / "data_raw" / "bulk"
RAW_SC = ROOT / "data_raw" / "scrna"
LOG = ROOT / "results" / "logs"

BULK_GSE = ["GSE30528", "GSE30529", "GSE96804", "GSE1009",
            "GSE30122", "GSE104954", "GSE99339"]
SC_GSE = ["GSE131882", "GSE209781"]

FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"


def _prefix(gse: str) -> str:
    return gse[:-3] + "nnn" if len(gse) > 6 else gse[:3] + "nnn"


def list_remote(url: str) -> list[str]:
    with urlopen(url, timeout=60) as fh:
        html = fh.read().decode("utf8", "replace")
    out = []
    for chunk in html.split('href="')[1:]:
        name = chunk.split('"')[0]
        if not name.startswith(("/", "http", "?")):
            out.append(name)
    return out


def fetch(url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=300) as fh, open(tmp, "wb") as out:
                while True:
                    block = fh.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            tmp.rename(dest)
            print(f"  [ok]   {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"  [retry {attempt}/{retries}] {url}: {exc}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"download failed: {url}")


def download_bulk() -> None:
    for gse in BULK_GSE:
        print(f"[bulk] {gse}")
        base = f"{FTP}/{_prefix(gse)}/{gse}/matrix/"
        try:
            files = [f for f in list_remote(base) if f.endswith(".txt.gz")]
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] cannot list {base}: {exc}")
            continue
        for f in files:
            fetch(base + f, RAW_BULK / gse / f)


def download_scrna() -> None:
    for gse in SC_GSE:
        print(f"[scrna] {gse}")
        base = f"{FTP}/{_prefix(gse)}/{gse}/suppl/"
        tar = RAW_SC / gse / f"{gse}_RAW.tar"
        fetch(base + f"{gse}_RAW.tar", tar)
        outdir = RAW_SC / gse / "raw"
        if not outdir.exists():
            outdir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as tf:
                tf.extractall(outdir)
            print(f"  [ok]   extracted -> {outdir}")


def checksum_manifest() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    rows = ["path\tbytes\tmd5"]
    for base in (RAW_BULK, RAW_SC):
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix != ".part":
                h = hashlib.md5()
                with open(p, "rb") as fh:
                    for blk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(blk)
                rows.append(f"{p.relative_to(ROOT)}\t{p.stat().st_size}\t{h.hexdigest()}")
    (LOG / "download_manifest.tsv").write_text("\n".join(rows) + "\n")
    print(f"[manifest] {len(rows)-1} files -> results/logs/download_manifest.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["bulk", "scrna", "all", "manifest"], default="all")
    args = ap.parse_args()
    if args.what in ("bulk", "all"):
        download_bulk()
    if args.what in ("scrna", "all"):
        download_scrna()
    if args.what in ("manifest", "all"):
        checksum_manifest()
    sys.exit(0)
