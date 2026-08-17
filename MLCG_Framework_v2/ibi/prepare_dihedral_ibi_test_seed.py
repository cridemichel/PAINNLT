#!/usr/bin/env python3
"""Create a non-production IBI seed containing periodic backbone dihedrals.

The seed is derived from the already selected bonded-angle chain in a base
``cg_priors.json``.  Consecutive angle entries i-j-k and j-k-l define one
signed ESPResSo dihedral i-j-k-l.  Existing file-backed priors are rewritten to
absolute paths so the normal IBI driver can write its temporary iteration
folders without breaking the fixed conservative bond/angle references.

This tool never modifies the base priors and deliberately marks the generated
artifact as test-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = 1
KIND = "periodic_dihedral_ibi_test_seed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node(entry: dict, mol_key: str, site_key: str) -> tuple[int, int]:
    return int(entry[mol_key]), int(entry.get(site_key, -1))


def _angle_nodes(entry: dict) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return (
        _node(entry, "mol_i", "site_i"),
        _node(entry, "mol_j", "site_j"),
        _node(entry, "mol_k", "site_k"),
    )


def _dihedral_name(first: dict, second: dict, nodes) -> str:
    def tokens(entry):
        name = str(entry.get("name", ""))
        if name.startswith("ang_"):
            values = name[4:].split("_")
            if len(values) == 3 and all(values):
                return values
        return None

    a = tokens(first)
    b = tokens(second)
    if a is not None and b is not None and a[1:] == b[:2]:
        return "dih_" + "_".join(a + [b[-1]])
    return "dih_" + "_".join(str(mol) for mol, _site in nodes)


def derive_dihedrals(priors: dict) -> list[dict]:
    angles = list(priors.get("angles", []))
    if not angles:
        raise ValueError("Base priors contain no angles from which to derive backbone dihedrals")

    by_prefix: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, dict]]] = {}
    for idx, entry in enumerate(angles):
        nodes = _angle_nodes(entry)
        by_prefix.setdefault((nodes[0], nodes[1]), []).append((idx, entry))

    result = []
    seen = set()
    for idx, first in enumerate(angles):
        a = _angle_nodes(first)
        successors = by_prefix.get((a[1], a[2]), [])
        if len(successors) > 1:
            raise ValueError(
                f"Angle chain branches after angles[{idx}] {a}; cannot define an unambiguous dihedral"
            )
        if not successors:
            continue
        _next_idx, second = successors[0]
        b = _angle_nodes(second)
        if (a[1], a[2]) != (b[0], b[1]):
            raise RuntimeError("Internal angle-chain matching failure")
        nodes = (a[0], a[1], a[2], b[2])
        if len(set(nodes)) != 4:
            raise ValueError(f"Degenerate/repeated-node dihedral derived from angles[{idx}]: {nodes}")
        if nodes in seen:
            raise ValueError(f"Duplicate dihedral derived from angle chain: {nodes}")
        seen.add(nodes)
        entry = {
            "type": "ibi",
            "ibi_mode": "ibi",
            "name": _dihedral_name(first, second, nodes),
            "mol_i": nodes[0][0],
            "mol_j": nodes[1][0],
            "mol_k": nodes[2][0],
            "mol_l": nodes[3][0],
            "site_i": nodes[0][1],
            "site_j": nodes[1][1],
            "site_k": nodes[2][1],
            "site_l": nodes[3][1],
            "test_only": True,
        }
        result.append(entry)

    if not result:
        raise ValueError("No consecutive angle pairs produced a dihedral")
    return result


def _absolutize_file_references(priors: dict, base_priors: Path) -> None:
    for key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(priors.get(key, [])):
            if "file" not in entry:
                continue
            path = Path(str(entry["file"])).expanduser()
            if not path.is_absolute():
                path = base_priors.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing base prior table {key}[{idx}]: {path}")
            entry["file"] = str(path)


def prepare(base_priors: Path, output: Path, report_path: Path | None = None, *, grouping_strategy: str) -> dict:
    base_priors = base_priors.expanduser().resolve()
    output = output.expanduser().resolve()
    if not base_priors.is_file():
        raise FileNotFoundError(base_priors)
    source = json.loads(base_priors.read_text())
    if grouping_strategy != "consecutive_angle_types":
        raise ValueError(f"Unsupported configured dihedral grouping strategy: {grouping_strategy!r}")
    if source.get("dihedrals"):
        raise ValueError(
            "Base production priors already contain dihedrals; this test seed refuses to replace them"
        )

    generated = derive_dihedrals(source)
    seed = copy.deepcopy(source)

    # The production bond/angle priors are fixed background interactions for
    # this diagnostic.  Promoted conservative IBI priors intentionally retain
    # ``ibi_mode`` as provenance, but the generic IBI driver interprets that
    # field as an instruction to rebuild/update the group.  Strip only the
    # inherited update marker here so step 35 isolates the newly generated
    # torsions while leaving the actual production interaction definitions
    # untouched.
    frozen_inherited = 0
    for key in ("bonds", "angles"):
        for idx, entry in enumerate(seed.get(key, [])):
            direct = str(entry.get("type", "")).lower()
            if direct in {"ibi", "dbi"}:
                raise ValueError(
                    f"Base production {key}[{idx}] is an unevaluated {direct!r} prior; "
                    "step 35 requires fixed evaluated bond/angle interactions"
                )
            entry.pop("ibi_mode", None)
            frozen_inherited += 1

    _absolutize_file_references(seed, base_priors)
    seed["dihedrals"] = generated
    groups = Counter(entry["name"] for entry in generated)
    base_promotion_metadata = {}
    for key in ("promotion", "regularization_candidate"):
        if key in seed:
            base_promotion_metadata[key] = seed.pop(key)
    seed["dihedral_ibi_test"] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "test_only": True,
        "base_priors": str(base_priors),
        "base_priors_sha256": sha256_file(base_priors),
        "dihedral_occurrences": len(generated),
        "grouping_strategy": grouping_strategy,
        "dihedral_groups": dict(sorted(groups.items())),
        "frozen_inherited_bond_angle_entries": frozen_inherited,
        "base_production_metadata": base_promotion_metadata,
        "note": "Diagnostic seed only; inherited production-promotion metadata is nested here and does not apply to the torsional candidate.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(seed, indent=2, sort_keys=False) + "\n")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "test_only": True,
        "base_priors": str(base_priors),
        "base_priors_sha256": sha256_file(base_priors),
        "output_seed": str(output),
        "output_seed_sha256": sha256_file(output),
        "dihedral_occurrences": len(generated),
        "grouping_strategy": grouping_strategy,
        "unique_groups": len(groups),
        "frozen_inherited_bond_angle_entries": frozen_inherited,
        "groups": dict(sorted(groups.items())),
        "pass": True,
    }
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[PERIODIC DIHEDRAL IBI TEST SEED]")
    print(f"base        : {base_priors}")
    print(f"output      : {output}")
    print(f"occurrences : {len(generated)}")
    print(f"groups      : {len(groups)}")
    for name, count in sorted(groups.items()):
        print(f"  {name}: {count}")
    print(
        f"[PASS] Test-only periodic backbone dihedrals derived with grouping={grouping_strategy}; "
        f"frozen inherited bond/angle entries={frozen_inherited}."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-priors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--grouping-strategy", required=True)
    args = parser.parse_args()
    prepare(args.base_priors, args.output, args.report, grouping_strategy=args.grouping_strategy)


if __name__ == "__main__":
    main()
