#!/usr/bin/env python3
"""Archive terminal TEL22_IBI historical outputs without touching live or GROMACS artifacts.

Dry-run is the default.  ``--run`` moves only a small reviewed allowlist of
historical output directories into ``tutorials/tel22_IBI/diagnostics/historical/phase3_archive``.  Historical
artifacts that are still referenced by the current configured workflows remain
in place.  GROMACS-generated files are protected explicitly and are never part
of the move plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NamedTuple

SCHEMA_VERSION = 1

# Reviewed terminal historical outputs.  These are development evidence, not
# inputs required by the current conservative production/certification path.
ARCHIVE_PLAN = {
    "historical_ibi": (
        "ibi_dbi_preview",
        "ibi_run",
    ),
    "ml_residual_experiments": (
        "training_multiseed_benchmark",
        "ibi_ml_ab_validation",
    ),
}

# Historical artifacts intentionally retained at the tutorial root because the
# current model-dependent configuration or downstream diagnostics still refer
# to them.  Archiving these requires a separate atomic reference migration.
LIVE_HISTORICAL_DEPENDENCIES = (
    "ibi_run_16ps",
    "ibi_run_16ps_continue",
    "diagnostics/ibi/ibi_validation_best",
    "diagnostics/ml/postibi_runtime_validation",
    "tel22_dataset_ibi_residual.bin",
    "tel22_model_ibi.pt",
    "tel22_model_ibi.pt.manifest.json",
    "tel22_model_ibi_conservative.pt",
    "tel22_model_ibi_conservative.pt.manifest.json",
    "ibi_residual_build_manifest.json",
)

# GROMACS products are explicitly protected by policy.  The phase-3 archive
# operation must never move or remove these even when they are large or
# byte-identical between tel22 and tel22_IBI.
PROTECTED_GROMACS_TOPLEVEL = (
    "143D.pdb",
    "pdb143d.ent.gz",
    "tel22_clean.pdb",
    "tel22_processed.gro",
    "topol.top",
    "posre.itp",
    "box_10.gro",
    "box_ions.gro",
    "box_solvated.gro",
    "ions.tpr",
    "em.edr",
    "em.gro",
    "em.log",
    "em.tpr",
    "em.trr",
    "nvt.cpt",
    "nvt.edr",
    "nvt.gro",
    "nvt.log",
    "nvt.tpr",
    "npt.cpt",
    "npt.edr",
    "npt.gro",
    "npt.log",
    "npt.tpr",
    "md.cpt",
    "md.edr",
    "md.gro",
    "md.log",
    "md.tpr",
    "md.trr",
    "md_whole.trr",
    "mdout.mdp",
    "md_sanity_100.log",
)


class MoveItem(NamedTuple):
    category: str
    name: str
    source: Path
    destination: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file() or path.is_symlink():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() or candidate.is_symlink():
            yield candidate


def describe_path(path: Path) -> dict:
    """Return a deterministic compact provenance description for ``path``."""
    if path.is_symlink():
        target = os.readlink(path)
        digest = hashlib.sha256(("symlink\0" + target).encode("utf-8")).hexdigest()
        return {
            "kind": "symlink",
            "file_count": 1,
            "bytes": 0,
            "tree_sha256": digest,
            "symlink_target": target,
        }
    if path.is_file():
        return {
            "kind": "file",
            "file_count": 1,
            "bytes": path.stat().st_size,
            "tree_sha256": _sha256_file(path),
        }

    h = hashlib.sha256()
    total = 0
    count = 0
    for file_path in _iter_files(path):
        rel = file_path.relative_to(path).as_posix()
        if file_path.is_symlink():
            payload = "symlink:" + os.readlink(file_path)
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            size = 0
        else:
            digest = _sha256_file(file_path)
            size = file_path.stat().st_size
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
        total += size
        count += 1
    return {
        "kind": "directory",
        "file_count": count,
        "bytes": total,
        "tree_sha256": h.hexdigest(),
    }


def build_move_plan(tutorial_dir: Path) -> list[MoveItem]:
    archive_root = tutorial_dir / "diagnostics" / "historical" / "phase3_archive"
    plan: list[MoveItem] = []
    protected = set(PROTECTED_GROMACS_TOPLEVEL)
    live = set(LIVE_HISTORICAL_DEPENDENCIES)
    for category, names in ARCHIVE_PLAN.items():
        for name in names:
            if name in protected:
                raise RuntimeError(f"Internal archive policy error: {name} is protected GROMACS data")
            if name in live:
                raise RuntimeError(f"Internal archive policy error: {name} is a live dependency")
            plan.append(
                MoveItem(
                    category=category,
                    name=name,
                    source=tutorial_dir / name,
                    destination=archive_root / category / name,
                )
            )
    return plan


def preflight(plan: list[MoveItem]) -> None:
    """Fail before moving anything if a destination collision is possible."""
    collisions = [item for item in plan if item.source.exists() and item.destination.exists()]
    if collisions:
        paths = ", ".join(str(item.destination) for item in collisions)
        raise RuntimeError(f"Archive destination already exists for live source(s): {paths}")


def _manifest_paths(tutorial_dir: Path) -> tuple[Path, Path]:
    root = tutorial_dir / "diagnostics" / "historical" / "phase3_archive"
    return root / "archive_manifest.json", root / "archive_manifest.md"


def make_manifest(
    tutorial_dir: Path,
    plan: list[MoveItem],
    *,
    executed: bool,
    moved_names: set[str] | None = None,
) -> dict:
    moved_names = moved_names or set()
    items = []
    for item in plan:
        source_exists = item.source.exists()
        dest_exists = item.destination.exists()
        if source_exists:
            description = describe_path(item.source)
            status = "planned"
        elif dest_exists:
            description = describe_path(item.destination)
            status = "archived_this_run" if item.name in moved_names else "already_archived"
        else:
            description = None
            status = "absent"
        items.append(
            {
                "category": item.category,
                "name": item.name,
                "source": str(item.source),
                "destination": str(item.destination),
                "status": status,
                "description": description,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "tel22_ibi_historical_archive_manifest",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(executed),
        "policy": {
            "gromacs_generated_preserved": True,
            "protected_gromacs_toplevel": list(PROTECTED_GROMACS_TOPLEVEL),
            "live_historical_dependencies_preserved": list(LIVE_HISTORICAL_DEPENDENCIES),
            "note": "Only reviewed terminal historical outputs are archived; GROMACS products are never moved.",
        },
        "items": items,
    }


def write_manifest(tutorial_dir: Path, manifest: dict) -> None:
    json_path, md_path = _manifest_paths(tutorial_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# TEL22_IBI historical archive manifest",
        "",
        f"- executed: **{manifest['executed']}**",
        "- GROMACS generated artifacts preserved: **yes**",
        "- live historical dependencies preserved at tutorial root: **yes**",
        "",
        "| category | artifact | status | files | bytes | tree SHA256 |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in manifest["items"]:
        desc = item["description"] or {}
        lines.append(
            "| {category} | `{name}` | {status} | {count} | {bytes_} | `{sha}` |".format(
                category=item["category"],
                name=item["name"],
                status=item["status"],
                count=desc.get("file_count", 0),
                bytes_=desc.get("bytes", 0),
                sha=desc.get("tree_sha256", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Preserved live historical dependencies",
            "",
            *[f"- `{name}`" for name in manifest["policy"]["live_historical_dependencies_preserved"]],
            "",
            "## GROMACS preservation policy",
            "",
            "The archive operation does not move or remove AA/GROMACS products. This includes",
            "`md.trr`, `md_whole.trr`, `md.gro`, TPR/CPT/EDR/LOG files, EM/NVT/NPT outputs,",
            "solvated/ionized GRO files, topology/position-restraint files, and source PDB files.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def execute(tutorial_dir: Path, *, run: bool) -> dict:
    tutorial_dir = tutorial_dir.resolve()
    plan = build_move_plan(tutorial_dir)
    preflight(plan)

    for item in plan:
        if item.source.exists():
            label = "MOVE" if run else "DRY-RUN"
            print(f"[{label}] {item.name} -> archive/{item.category}/{item.name}")
        elif item.destination.exists():
            print(f"[REUSE] {item.name} already archived")
        else:
            print(f"[SKIP] {item.name} not present")

    moved_names: set[str] = set()
    if run:
        for item in plan:
            if not item.source.exists():
                continue
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.source), str(item.destination))
            moved_names.add(item.name)

    manifest = make_manifest(tutorial_dir, plan, executed=run, moved_names=moved_names)
    if run:
        write_manifest(tutorial_dir, manifest)
        print(f"[DONE] manifest: {_manifest_paths(tutorial_dir)[0]}")
        print("[PASS] Historical terminal outputs archived; GROMACS generated files preserved in place.")
    else:
        print("[NOTE] Dry-run only. No file was moved or removed.")
        print("[NOTE] GROMACS generated files are explicitly outside the archive plan.")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tutorial-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tel22_IBI",
        help="TEL22_IBI tutorial directory (default: tutorials/tel22_IBI)",
    )
    parser.add_argument("--run", action="store_true", help="perform the reviewed archive moves")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        execute(args.tutorial_dir, run=args.run)
    except (OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
