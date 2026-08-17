#!/usr/bin/env python3
"""Non-destructive TEL22/TEL22_IBI deduplication and reference audit.

The audit never removes, moves, rewrites, or symlinks tutorial artifacts. It
computes SHA256 for files that share the same relative path across the two
Tutorial trees, classifies candidates conservatively, and records textual path
references that must be considered before any future deduplication pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# Files produced by the normal AA/CG tutorial execution. These are regenerable
# artifacts, not immutable inputs. Directory names below are intentionally
# tutorial-output classes rather than scientific acceptance policy.
GENERATED_NAMES = {
    "143D.pdb",
    "box_10.gro",
    "box_ions.gro",
    "box_solvated.gro",
    "cg_training_log.csv",
    "cg_trajectory.vtf",
    "em.edr",
    "em.gro",
    "em.log",
    "em.tpr",
    "em.trr",
    "energy.csv",
    "equilibrated.npz",
    "ions.tpr",
    "md.cpt",
    "md.edr",
    "md.gro",
    "md.log",
    "md.tpr",
    "md.trr",
    "md_whole.trr",
    "mdout.mdp",
    "md_sanity_100.log",
    "npt.cpt",
    "npt.edr",
    "npt.gro",
    "npt.log",
    "npt.tpr",
    "nvt.cpt",
    "nvt.edr",
    "nvt.gro",
    "nvt.log",
    "nvt.tpr",
    "pdb143d.ent.gz",
    "posre.itp",
    "smoke_equilibrated.npz",
    "tel22_clean.pdb",
    "tel22_dataset.bin",
    "tel22_dataset_ibi_residual.bin",
    "tel22_model.pt",
    "tel22_model_ibi.pt",
    "tel22_model_ibi_conservative.pt",
    "tel22_processed.gro",
    "topol.top",
}

# Only source-like immutable inputs are proposed for physical sharing. The audit
# deliberately excludes dataset/model/prior/config files even when hashes match:
# those are tutorial-local semantic artifacts that are allowed to diverge.
SHARED_INPUT_NAMES = {
    "143D.pdb",
    "pdb143d.ent.gz",
}
SHARED_INPUT_PREFIXES = (
    "mdp/",
)

HISTORICAL_IBI_PREFIXES = (
    "ibi_dbi_preview/",
    "ibi_ml_ab_validation/",
    "ibi_run/",
    "ibi_run_16ps/",
    "ibi_run_16ps_continue/",
    "ibi_validation_best/",
    "postibi_runtime_validation/",
    "training_multiseed_benchmark/",
)
HISTORICAL_IBI_NAMES = {
    "ibi_residual_build_manifest.json",
    "tel22_dataset_ibi_residual.bin",
    "tel22_model_ibi.pt",
    "tel22_model_ibi.pt.manifest.json",
    "tel22_model_ibi_conservative.pt",
    "tel22_model_ibi_conservative.pt.manifest.json",
}

DIAGNOSTIC_IBI_PREFIXES = (
    "conservative_ibi_energy_localization/",
    "ibi_angle_final_candidate_validation/",
    "ibi_angle_regularization_diagnostic/",
    "ibi_angle_regularization_validation/",
    "ibi_angle_smoothing_sweep/",
    "ibi_dihedral_candidate_test/",
    "ibi_dihedral_conservative_in_loop_test/",
    "ibi_dihedral_conservative_replica_matrix/",
    "ibi_dihedral_legacy_conservativity_diagnostic/",
    "ibi_dihedral_update_localization/",
    "ibi_timestep_range_diagnostic/",
    "ibi_conservative_pre_smooth_0p0075/",
    "nve_certification/",
    "nve_certification_conservative_ibi_only/",
    "nve_certification_fp32/",
    "nve_certification_fp32_new_morse/",
    "nve_certification_fp64/",
    "nve_certification_fp64_1ps/",
    "nve_certification_fp64_2ps/",
    "nve_certification_fp64_new_morse/",
    "nve_certification_wca_v3_dt10_stress/",
    "nve_diagnostic_conservative_ibi_only/",
    "nve_equilibration_conservative_ibi_only/",
    "nve_final_certification_conservative_ibi_only/",
    "nve_state_convergence_conservative_ibi_only/",
    "sigma_energy_replica_window_diagnostic/",
    "wca12_selective_ab/",
)

CANONICAL_IBI_PREFIXES = (
    "ibi_conservative/",
    "ibi_promoted_final_certification/",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path) -> dict[str, Path]:
    inventory: dict[str, Path] = {}
    if not root.is_dir():
        raise FileNotFoundError(f"tutorial directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            inventory[path.relative_to(root).as_posix()] = path
    return inventory


def is_generated(rel: str) -> bool:
    name = Path(rel).name
    if name in GENERATED_NAMES:
        return True
    if rel.startswith(".short_mdp/"):
        return True
    return False


def is_shared_input_candidate(rel: str) -> bool:
    if Path(rel).name in SHARED_INPUT_NAMES:
        return True
    return rel.startswith(SHARED_INPUT_PREFIXES)


def classify_ibi_only(rel: str) -> str:
    if rel in HISTORICAL_IBI_NAMES or rel.startswith(HISTORICAL_IBI_PREFIXES):
        return "HISTORICAL"
    if is_generated(rel):
        return "GENERATED"
    if rel.startswith(CANONICAL_IBI_PREFIXES):
        return "KEEP"
    if rel.startswith(DIAGNOSTIC_IBI_PREFIXES):
        return "DIAGNOSTIC"
    return "KEEP"


def classify_tel22_only(rel: str) -> str:
    if is_generated(rel):
        return "GENERATED"
    if rel.startswith("nve_certification") or rel.startswith("wca12_selective_ab/"):
        return "DIAGNOSTIC"
    return "KEEP"


def readable_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Makefile"}:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def collect_text_documents(repo_root: Path, tutorial_roots: Iterable[Path]) -> dict[Path, str]:
    roots = [repo_root / "README.md", repo_root / "HOWTO.md", repo_root / "HOWTO_EN.md"]
    files: set[Path] = {p for p in roots if p.is_file()}
    for root in tutorial_roots:
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                files.add(path)
    documents: dict[Path, str] = {}
    for path in sorted(files):
        text = readable_text(path)
        if text is not None:
            documents[path] = text
    return documents


def reference_hits(repo_root: Path, documents: dict[Path, str], rel: str) -> list[dict[str, object]]:
    basename = Path(rel).name
    needles = {
        rel,
        f"tel22/{rel}",
        f"tel22_IBI/{rel}",
        f"tutorials/tel22/{rel}",
        f"tutorials/tel22_IBI/{rel}",
    }
    # For root-level files, local wrapper scripts generally reference only the
    # basename. Include that form. For nested files, basename-only matches are
    # too ambiguous to be useful.
    if "/" not in rel:
        needles.add(basename)

    hits: list[dict[str, object]] = []
    for path, text in documents.items():
        for lineno, line in enumerate(text.splitlines(), start=1):
            matching = sorted(needle for needle in needles if needle and needle in line)
            if matching:
                hits.append(
                    {
                        "file": path.relative_to(repo_root).as_posix(),
                        "line": lineno,
                        "matched": matching,
                        "text": line.strip()[:240],
                    }
                )
    return hits


def duplicate_class(rel: str, identical: bool) -> str:
    if not identical:
        return "KEEP_SEPARATE_DIFFERENT"
    if is_shared_input_candidate(rel):
        return "SHARED_CANDIDATE"
    if is_generated(rel):
        return "GENERATED_DUPLICATE"
    return "KEEP_SEPARATE_DUPLICATE"


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def render_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# TEL22 phase-2 deduplication audit",
        "",
        "> Non-destructive audit. No file was removed, moved, rewritten or linked.",
        "",
        "## Summary",
        "",
        f"- TEL22 files: **{summary['tel22_files']}**",
        f"- TEL22_IBI files: **{summary['tel22_ibi_files']}**",
        f"- same-relative-path files: **{summary['same_relative_path_files']}**",
        f"- byte-identical duplicates: **{summary['identical_duplicates']}**",
        f"- generated duplicate bytes: **{human_bytes(summary['generated_duplicate_bytes'])}**",
        f"- immutable shared-input candidate bytes: **{human_bytes(summary['shared_candidate_bytes'])}**",
        f"- same-path but byte-different files: **{summary['different_same_path']}**",
        "",
        "The generated-duplicate byte count is disk-space opportunity, not a",
        "recommendation to symlink generated products. The shared-input count is",
        "the only class proposed for possible physical sharing in a later phase.",
        "",
        "## Shared immutable-input candidates",
        "",
        "| relative path | bytes | SHA256 | reference hits |",
        "|---|---:|---|---:|",
    ]
    shared = [d for d in report["duplicates"] if d["classification"] == "SHARED_CANDIDATE"]
    if shared:
        for item in shared:
            lines.append(
                f"| `{item['relative_path']}` | {item['size_bytes']} | `{item['sha256_tel22']}` | {len(item['reference_hits'])} |"
            )
    else:
        lines.append("| _none_ |  |  |  |")

    lines.extend(
        [
            "",
            "## Generated byte-identical duplicates",
            "",
            "These are candidates for regeneration rather than sharing. They remain untouched.",
            "",
            "| relative path | bytes | reference hits |",
            "|---|---:|---:|",
        ]
    )
    generated = [d for d in report["duplicates"] if d["classification"] == "GENERATED_DUPLICATE"]
    if generated:
        for item in generated:
            lines.append(
                f"| `{item['relative_path']}` | {item['size_bytes']} | {len(item['reference_hits'])} |"
            )
    else:
        lines.append("| _none_ |  |  |")

    lines.extend(
        [
            "",
            "## Same path but different content",
            "",
            "These files must remain tutorial-local unless a separate semantic migration is designed.",
            "",
            "| relative path | TEL22 bytes | TEL22_IBI bytes |",
            "|---|---:|---:|",
        ]
    )
    different = [d for d in report["duplicates"] if d["classification"] == "KEEP_SEPARATE_DIFFERENT"]
    if different:
        for item in different:
            lines.append(
                f"| `{item['relative_path']}` | {item['tel22_size_bytes']} | {item['tel22_ibi_size_bytes']} |"
            )
    else:
        lines.append("| _none_ |  |  |")

    lines.extend(
        [
            "",
            "## TEL22_IBI-only top-level classification",
            "",
            "The classification is conservative and advisory. `HISTORICAL` means",
            "candidate for a future archive pass, not permission to delete it now.",
            "",
            "| path | class |",
            "|---|---|",
        ]
    )
    top_level = report["tel22_ibi_top_level"]
    for item in top_level:
        lines.append(f"| `{item['path']}` | **{item['classification']}** |")

    lines.extend(
        [
            "",
            "## Reference audit",
            "",
            "Reference hits for dedup candidates are stored in the JSON report.",
            "A future phase must migrate those references atomically before moving",
            "or sharing any immutable input. No `../tel22/...` dependency is introduced",
            "by this audit.",
            "",
            "## Next action",
            "",
            "Physical sharing of `SHARED_CANDIDATE` inputs remains deferred: the measured",
            "space saving is small compared with the coupling/reference-migration cost.",
            "`GENERATED_DUPLICATE` entries should remain local/regenerable rather than",
            "symlinked; preserving GROMACS working products is a valid local policy.",
            "Phase 3 is a separate conservative archive pass for reviewed terminal",
            "`HISTORICAL` outputs only; live historical dependencies remain in place.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(repo_root: Path, tel22: Path, tel22_ibi: Path) -> dict[str, object]:
    inv_a = file_inventory(tel22)
    inv_b = file_inventory(tel22_ibi)
    documents = collect_text_documents(repo_root, (tel22, tel22_ibi))

    common = sorted(set(inv_a) & set(inv_b))
    duplicates: list[dict[str, object]] = []
    generated_duplicate_bytes = 0
    shared_candidate_bytes = 0
    identical_duplicates = 0
    different_same_path = 0

    for rel in common:
        pa = inv_a[rel]
        pb = inv_b[rel]
        size_a = pa.stat().st_size
        size_b = pb.stat().st_size
        sha_a = sha256_file(pa)
        sha_b = sha256_file(pb)
        identical = size_a == size_b and sha_a == sha_b
        classification = duplicate_class(rel, identical)
        if identical:
            identical_duplicates += 1
            if classification == "GENERATED_DUPLICATE":
                generated_duplicate_bytes += size_a
            elif classification == "SHARED_CANDIDATE":
                shared_candidate_bytes += size_a
        else:
            different_same_path += 1
        duplicates.append(
            {
                "relative_path": rel,
                "classification": classification,
                "identical": identical,
                "tel22_size_bytes": size_a,
                "tel22_ibi_size_bytes": size_b,
                "size_bytes": size_a if identical else None,
                "sha256_tel22": sha_a,
                "sha256_tel22_ibi": sha_b,
                "reference_hits": reference_hits(repo_root, documents, rel),
            }
        )

    tel22_only = []
    for rel in sorted(set(inv_a) - set(inv_b)):
        tel22_only.append(
            {
                "relative_path": rel,
                "classification": classify_tel22_only(rel),
                "size_bytes": inv_a[rel].stat().st_size,
            }
        )

    ibi_only = []
    for rel in sorted(set(inv_b) - set(inv_a)):
        ibi_only.append(
            {
                "relative_path": rel,
                "classification": classify_ibi_only(rel),
                "size_bytes": inv_b[rel].stat().st_size,
            }
        )

    # Directory/file top-level view: this is easier to review than thousands of
    # generated files inside diagnostic directories.
    top_level_names = sorted({p.name for p in tel22_ibi.iterdir()})
    top_level: list[dict[str, str]] = []
    for name in top_level_names:
        rel = f"{name}/" if (tel22_ibi / name).is_dir() else name
        if rel.rstrip("/") in inv_a and not rel.endswith("/"):
            # Same-name root file is covered by duplicate classification.
            continue
        top_level.append({"path": rel, "classification": classify_ibi_only(rel)})

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "tel22_tel22_ibi_phase2_dedup_audit",
        "non_destructive": True,
        "repo_root": str(repo_root),
        "tel22_root": str(tel22),
        "tel22_ibi_root": str(tel22_ibi),
        "summary": {
            "tel22_files": len(inv_a),
            "tel22_ibi_files": len(inv_b),
            "same_relative_path_files": len(common),
            "identical_duplicates": identical_duplicates,
            "different_same_path": different_same_path,
            "generated_duplicate_bytes": generated_duplicate_bytes,
            "shared_candidate_bytes": shared_candidate_bytes,
        },
        "duplicates": duplicates,
        "tel22_only": tel22_only,
        "tel22_ibi_only": ibi_only,
        "tel22_ibi_top_level": top_level,
        "classification_policy": {
            "SHARED_CANDIDATE": "byte-identical immutable source input; may be physically shared only after reference migration review",
            "GENERATED_DUPLICATE": "byte-identical regenerable output; prefer regeneration/ignore rather than symlink",
            "KEEP_SEPARATE_DUPLICATE": "byte-identical today but tutorial-local semantic artifact; do not deduplicate automatically",
            "KEEP_SEPARATE_DIFFERENT": "same relative path, different bytes; keep separate",
            "HISTORICAL": "development/ML/old-IBI evidence; future archive candidate, not deletion authorization",
            "DIAGNOSTIC": "diagnostic/certification evidence; keep until an explicit archive policy exists",
            "KEEP": "canonical or otherwise not classified for cleanup",
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=script_repo)
    parser.add_argument("--tel22", type=Path, default=None)
    parser.add_argument("--tel22-ibi", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <repo>/tutorials/.tel22_cleanup_audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    tel22 = (args.tel22 or repo_root / "tutorials" / "tel22").resolve()
    tel22_ibi = (args.tel22_ibi or repo_root / "tutorials" / "tel22_IBI").resolve()
    output_dir = (args.output_dir or repo_root / "tutorials" / ".tel22_cleanup_audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(repo_root, tel22, tel22_ibi)
    json_path = output_dir / "phase2_dedup_audit.json"
    md_path = output_dir / "phase2_dedup_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    summary = report["summary"]
    print("[TEL22 PHASE-2 DEDUP AUDIT]")
    print(f"tel22 files              : {summary['tel22_files']}")
    print(f"tel22_IBI files          : {summary['tel22_ibi_files']}")
    print(f"same relative path       : {summary['same_relative_path_files']}")
    print(f"byte-identical duplicates: {summary['identical_duplicates']}")
    print(f"different same-path      : {summary['different_same_path']}")
    print(
        "generated duplicate bytes: "
        f"{summary['generated_duplicate_bytes']} ({human_bytes(summary['generated_duplicate_bytes'])})"
    )
    print(
        "shared candidate bytes    : "
        f"{summary['shared_candidate_bytes']} ({human_bytes(summary['shared_candidate_bytes'])})"
    )
    print("[NOTE] Non-destructive: no file was moved, removed, rewritten or linked.")
    print(f"[DONE] JSON report: {json_path}")
    print(f"[DONE] Markdown   : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
