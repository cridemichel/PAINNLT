#!/usr/bin/env python3
"""Create a fail-closed IBI seed from an already generated priors JSON.

This tutorial helper deliberately leaves the certified TEL22 priors untouched.
Only explicitly selected named bonded groups are converted to ``type=ibi`` or
``type=dbi``.  Existing WCA, Morse and all unselected bonded priors are copied
verbatim.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

SUPPORTED_CATEGORIES = {"bonds", "angles", "dihedrals"}
SUPPORTED_MODES = {"ibi", "dbi"}
STALE_KEYS = {
    "bonds": {"k", "r0", "r_max", "D", "a", "r_switch", "r_cut", "file", "min", "max", "ibi_mode"},
    "angles": {"k", "theta0", "file", "min", "max", "ibi_mode"},
    "dihedrals": {"k", "phi0", "n", "periodicity", "file", "min", "max", "ibi_mode"},
}


def prepare_seed(source_path: Path, selection_path: Path, output_path: Path) -> dict:
    source_path = source_path.resolve()
    selection_path = selection_path.resolve()
    output_path = output_path.resolve()
    priors = json.loads(source_path.read_text())
    selection = json.loads(selection_path.read_text())
    if int(selection.get("schema_version", 0)) != 1:
        raise ValueError("ibi_selection.json requires schema_version=1")
    groups = selection.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("ibi_selection.json requires a non-empty 'groups' list")

    output = copy.deepcopy(priors)
    seen = set()
    converted = Counter()
    report_groups = []

    for spec in groups:
        category = str(spec.get("category", ""))
        name = str(spec.get("name", ""))
        mode = str(spec.get("mode", "")).lower()
        expected_source_type = str(spec.get("expected_source_type", "")).lower()
        expected_count = int(spec.get("expected_count", -1))
        key = (category, name)

        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"Unsupported IBI category {category!r}")
        if not name:
            raise ValueError("Every IBI selection group requires a non-empty name")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"IBI group {category}:{name} has invalid mode {mode!r}")
        if key in seen:
            raise ValueError(f"Duplicate IBI selection for {category}:{name}")
        seen.add(key)

        entries = output.get(category, [])
        indices = [idx for idx, entry in enumerate(entries) if str(entry.get("name", "")) == name]
        if not indices:
            raise ValueError(f"IBI selection {category}:{name} matched no priors")
        if expected_count >= 0 and len(indices) != expected_count:
            raise ValueError(
                f"IBI selection {category}:{name} expected {expected_count} entries, found {len(indices)}"
            )

        source_types = Counter(str(entries[idx].get("type", "")).lower() for idx in indices)
        if expected_source_type and source_types != Counter({expected_source_type: len(indices)}):
            raise ValueError(
                f"IBI selection {category}:{name} expected only type={expected_source_type!r}, "
                f"found {dict(source_types)}"
            )

        for idx in indices:
            entry = entries[idx]
            for stale in STALE_KEYS[category]:
                entry.pop(stale, None)
            entry["type"] = mode
        converted[category] += len(indices)
        report_groups.append(
            {
                "category": category,
                "name": name,
                "mode": mode,
                "count": len(indices),
                "source_types": dict(source_types),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return {
        "source": str(source_path),
        "selection": str(selection_path),
        "output": str(output_path),
        "converted": dict(converted),
        "groups": report_groups,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare explicitly selected bonded IBI/DBI seed priors")
    parser.add_argument("--priors", default="cg_priors.json")
    parser.add_argument("--selection", default="ibi_selection.json")
    parser.add_argument("--output", default="cg_priors_ibi_seed.json")
    parser.add_argument("--report", default="ibi_seed_report.json")
    args = parser.parse_args()

    source = Path(args.priors)
    selection = Path(args.selection)
    output = Path(args.output)
    report = prepare_seed(source, selection, output)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print("[IBI SEED]")
    for group in report["groups"]:
        print(
            f"  {group['category']:<9} {group['name']:<14} "
            f"mode={group['mode']} count={group['count']}"
        )
    print(f"[SUCCESS] Seed priors: {output}")
    print(f"[SUCCESS] Report: {args.report}")


if __name__ == "__main__":
    main()
