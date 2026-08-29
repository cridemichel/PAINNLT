#!/usr/bin/env python3
"""Build or validate the TEL22 Variant-A prior topology.

Variant A keeps WCA, harmonic backbone bonds and harmonic backbone angles, and
removes every pair-specific or type-pair Morse interaction.  The source is
first validated against the corrected antiparallel PDB-143D contact graph so
that this diagnostic cannot accidentally be generated from the legacy graph.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from validate_antiparallel_topology import validate_topology_data


EXPECTED_COPIES = 10
BACKBONE_BONDS_PER_COPY = 21
BACKBONE_ANGLES_PER_COPY = 20
EXPECTED_SOURCE_MORSE_PER_COPY = 18


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_variant_a_topology_data(
    data: dict[str, Any], *, copies: int = EXPECTED_COPIES
) -> dict[str, Any]:
    bonds = data.get("bonds", [])
    angles = data.get("angles", [])
    _require(isinstance(bonds, list), "bonds must be a list")
    _require(isinstance(angles, list), "angles must be a list")

    morse_bonds = [
        item for item in bonds if str(item.get("type", "")).lower() == "morse"
    ]
    harmonic_bonds = [
        item for item in bonds if str(item.get("type", "")).lower() == "harmonic"
    ]
    other_bonds = [
        item
        for item in bonds
        if str(item.get("type", "")).lower() not in {"harmonic", "morse"}
    ]
    harmonic_angles = [
        item for item in angles if str(item.get("type", "")).lower() == "harmonic"
    ]
    other_angles = [
        item for item in angles if str(item.get("type", "")).lower() != "harmonic"
    ]

    _require(not morse_bonds, f"Variant A must contain zero Morse bonds, found {len(morse_bonds)}")
    _require(not data.get("morse_type_pairs"), "Variant A must contain zero Morse type-pair interactions")
    _require(not other_bonds, f"Variant A contains unsupported bond types: {other_bonds[:3]}")
    _require(not other_angles, f"Variant A contains non-harmonic angles: {other_angles[:3]}")
    _require(
        len(harmonic_bonds) == copies * BACKBONE_BONDS_PER_COPY,
        f"Expected {copies * BACKBONE_BONDS_PER_COPY} harmonic bonds, found {len(harmonic_bonds)}",
    )
    _require(
        len(harmonic_angles) == copies * BACKBONE_ANGLES_PER_COPY,
        f"Expected {copies * BACKBONE_ANGLES_PER_COPY} harmonic angles, found {len(harmonic_angles)}",
    )

    has_wca = bool(data.get("wca_pairs")) or "wca_epsilon" in data or isinstance(data.get("wca"), dict)
    _require(has_wca, "Variant A must retain a WCA/excluded-volume prior")

    return {
        "variant": "A",
        "copies": copies,
        "harmonic_bonds": len(harmonic_bonds),
        "harmonic_angles": len(harmonic_angles),
        "morse_contacts": 0,
        "morse_type_pairs": 0,
        "wca_pairs": len(data.get("wca_pairs", {})),
    }


def validate_variant_a_topology_file(
    path: Path, *, copies: int = EXPECTED_COPIES
) -> dict[str, Any]:
    return validate_variant_a_topology_data(
        json.loads(path.read_text(encoding="utf-8")), copies=copies
    )


def build_variant_a_data(
    source: dict[str, Any], *, copies: int = EXPECTED_COPIES
) -> tuple[dict[str, Any], int]:
    validate_topology_data(
        source,
        copies=copies,
        r0_mode="auto",
        require_reference_metadata=True,
    )
    result = copy.deepcopy(source)
    source_bonds = result.get("bonds", [])
    kept_bonds = [
        item for item in source_bonds if str(item.get("type", "")).lower() != "morse"
    ]
    removed = len(source_bonds) - len(kept_bonds)
    expected_removed = copies * EXPECTED_SOURCE_MORSE_PER_COPY
    _require(
        removed == expected_removed,
        f"Expected to remove {expected_removed} Morse bonds, removed {removed}",
    )
    result["bonds"] = kept_bonds
    result["morse_type_pairs"] = []
    result["diagnostic_variant"] = {
        "name": "A",
        "description": "WCA plus harmonic backbone bonds and angles; all Morse interactions removed",
        "source_topology": "PDB 143D antiparallel TEL22 topology",
        "removed_morse_contacts": removed,
    }
    validate_variant_a_topology_data(result, copies=copies)
    return result, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Variant A derived from the corrected source; omit to validate an existing Variant-A file",
    )
    parser.add_argument("--copies", type=int, default=EXPECTED_COPIES)
    args = parser.parse_args()

    if args.output is None:
        summary = validate_variant_a_topology_file(args.input, copies=args.copies)
    else:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        variant, removed = build_variant_a_data(source, copies=args.copies)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(variant, indent=4) + "\n", encoding="utf-8")
        summary = validate_variant_a_topology_data(variant, copies=args.copies)
        summary["removed_morse_contacts"] = removed
        summary["output"] = str(args.output)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[PASS] TEL22 Variant A contains WCA plus harmonic bonds/angles and no Morse interactions.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
