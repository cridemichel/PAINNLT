#!/usr/bin/env python3
"""Download and validate the public five-bead alanine-dipeptide arrays."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np


CGNET_COMMIT = "a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481"
BASE_URL = (
    "https://raw.githubusercontent.com/coarse-graining/cgnet/"
    f"{CGNET_COMMIT}/examples/data"
)
FILES = {
    "ala2_coordinates.npy": "00f1f6b70fbc9473157511d53a73b6f629d284d3e08e79155b9d2bf546d6dc81",
    "ala2_forces.npy": "de1936e1a431b789cb3366b6d5c0208913d2b516d81d199f09f5c914bb536f56",
}
EXPECTED_SHAPE = (10000, 5, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_array(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    array = np.load(path, allow_pickle=False)
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected shape for {path}: expected {EXPECTED_SHAPE}, got {array.shape}"
        )
    if array.dtype != np.float32:
        raise ValueError(f"Unexpected dtype for {path}: expected float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {path}")


def download_one(destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        validate_array(destination, expected_sha256)
        print(f"[PASS] Existing verified file: {destination}")
        return

    url = f"{BASE_URL}/{destination.name}"
    request = urllib.request.Request(url, headers={"User-Agent": "MLCG-Framework-Ala2/1"})
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f"[INFO] Downloading {url}")
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        validate_array(temporary, expected_sha256)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[PASS] Downloaded and verified: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--check-only", action="store_true", help="Validate existing files without downloading"
    )
    args = parser.parse_args()

    for name, digest in FILES.items():
        path = args.output_dir.resolve() / name
        if args.check_only:
            if not path.is_file():
                raise FileNotFoundError(path)
            validate_array(path, digest)
            print(f"[PASS] Verified: {path}")
        else:
            download_one(path, digest)

    print(f"[PASS] Official CGnet Ala2 arrays verified at commit {CGNET_COMMIT}.")


if __name__ == "__main__":
    main()
