#!/usr/bin/env python3
"""Download a pinned, verified copy of the official CGnet source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


CGNET_COMMIT = "a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481"
ARCHIVE_URL = f"https://github.com/coarse-graining/cgnet/archive/{CGNET_COMMIT}.zip"
ARCHIVE_SHA256 = "98992c79b2c670e86b70c5edae7fad0a39a4631f70a9697dd71d78eaa4924e23"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            destination = (root / member.filename).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(root)
    candidates = [path for path in root.iterdir() if (path / "cgnet" / "__init__.py").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one CGnet source root, found {len(candidates)}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"CGnet source directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix="cgnet-official-", suffix=".zip", dir=output_dir.parent
    )
    os.close(fd)
    archive = Path(temporary_name)
    try:
        request = urllib.request.Request(
            ARCHIVE_URL, headers={"User-Agent": "MLCG-Framework-CGnet-Comparator/1"}
        )
        print(f"[INFO] Downloading pinned official CGnet source: {ARCHIVE_URL}")
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        actual_sha256 = sha256_file(archive)
        if actual_sha256 != ARCHIVE_SHA256:
            raise ValueError(
                f"CGnet archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual_sha256}"
            )
        source_root = safe_extract(archive, output_dir)
    finally:
        archive.unlink(missing_ok=True)

    report = {
        "schema_version": 1,
        "status": "pass",
        "repository": "https://github.com/coarse-graining/cgnet",
        "commit": CGNET_COMMIT,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "source_root": str(source_root),
        "source_files_modified": False,
        "compatibility_strategy": "Runtime NumPy alias only; official source bytes are unchanged.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[PASS] Official CGnet source verified at commit {CGNET_COMMIT}.")
    print(source_root)


if __name__ == "__main__":
    main()
