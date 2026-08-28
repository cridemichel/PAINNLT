#!/usr/bin/env python3
"""Validate the TEL22 143D antiparallel G-tetrad contact graph.

The topology uses zero-based molecule indices, while the structural residue
numbers reported here are one-based and match PDB 143D.  Each tetrad is
represented by the complete graph K4 (six pair-specific Morse contacts).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any


RESIDUES_PER_COPY = 22
EXPECTED_TETRADS_1BASED = (
    (2, 10, 14, 22),
    (3, 9, 15, 21),
    (4, 8, 16, 20),
)
LEGACY_TETRADS_1BASED = (
    (2, 8, 14, 20),
    (3, 9, 15, 21),
    (4, 10, 16, 22),
)


def complete_graph_edges(groups: tuple[tuple[int, ...], ...]) -> set[tuple[int, int]]:
    return {
        tuple(sorted(pair))
        for group in groups
        for pair in itertools.combinations(group, 2)
    }


EXPECTED_EDGES_1BASED = complete_graph_edges(EXPECTED_TETRADS_1BASED)
LEGACY_EDGES_1BASED = complete_graph_edges(LEGACY_TETRADS_1BASED)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_topology_data(
    data: dict[str, Any],
    *,
    copies: int = 10,
    r0_mode: str = "any",
    require_reference_metadata: bool = False,
) -> dict[str, Any]:
    """Validate all pair-specific Morse contacts and return a short summary."""
    _require(r0_mode in {"any", "auto", "numeric"}, f"Unknown r0 mode: {r0_mode}")
    _require(not data.get("morse_type_pairs"), "TEL22 must not add broad Morse type pairs")

    reference = data.get("tel22_g4_topology")
    if require_reference_metadata:
        _require(isinstance(reference, dict), "Missing tel22_g4_topology metadata")
        recorded = tuple(tuple(int(x) for x in group) for group in reference.get("tetrads_1based", []))
        _require(recorded == EXPECTED_TETRADS_1BASED, "Incorrect tetrads in tel22_g4_topology metadata")
        _require(reference.get("pdb") == "143D", "Topology reference must be PDB 143D")
        _require(reference.get("model") == 1, "Topology reference must select MODEL 1")

    morse = [bond for bond in data.get("bonds", []) if str(bond.get("type", "")).lower() == "morse"]
    _require(len(morse) == copies * 18, f"Expected {copies * 18} Morse contacts, found {len(morse)}")

    for copy_index in range(copies):
        start = copy_index * RESIDUES_PER_COPY
        stop = start + RESIDUES_PER_COPY
        contacts = [
            bond
            for bond in morse
            if start <= int(bond["mol_i"]) < stop or start <= int(bond["mol_j"]) < stop
        ]
        _require(len(contacts) == 18, f"Copy {copy_index + 1}: expected 18 contacts, found {len(contacts)}")

        edges: set[tuple[int, int]] = set()
        for bond in contacts:
            i = int(bond["mol_i"])
            j = int(bond["mol_j"])
            _require(start <= i < stop and start <= j < stop, f"Copy {copy_index + 1}: cross-copy Morse {i}-{j}")
            _require(int(bond.get("site_i", -1)) == -1, f"Copy {copy_index + 1}: site_i must be explicit COM (-1)")
            _require(int(bond.get("site_j", -1)) == -1, f"Copy {copy_index + 1}: site_j must be explicit COM (-1)")
            edge = tuple(sorted((i - start + 1, j - start + 1)))
            _require(edge not in edges, f"Copy {copy_index + 1}: duplicate Morse edge {edge}")
            edges.add(edge)

            r0 = bond.get("r0")
            if r0_mode == "auto":
                _require(r0 == "auto", f"Copy {copy_index + 1}, edge {edge}: r0 must be 'auto'")
            elif r0_mode == "numeric":
                _require(
                    isinstance(r0, (int, float)) and math.isfinite(float(r0)) and float(r0) > 0.0,
                    f"Copy {copy_index + 1}, edge {edge}: r0 must be finite and positive",
                )

        if edges == LEGACY_EDGES_1BASED:
            raise ValueError(
                f"Copy {copy_index + 1}: legacy parallel-register Morse graph detected; "
                "outer antiparallel tetrads are mixed"
            )
        _require(
            edges == EXPECTED_EDGES_1BASED,
            f"Copy {copy_index + 1}: Morse graph differs from PDB 143D; "
            f"missing={sorted(EXPECTED_EDGES_1BASED - edges)}, extra={sorted(edges - EXPECTED_EDGES_1BASED)}",
        )

    return {
        "copies": copies,
        "morse_contacts": len(morse),
        "contacts_per_copy": 18,
        "tetrads_1based": [list(group) for group in EXPECTED_TETRADS_1BASED],
        "r0_mode": r0_mode,
    }


def validate_topology_file(
    path: Path,
    *,
    copies: int = 10,
    r0_mode: str = "any",
    require_reference_metadata: bool = False,
) -> dict[str, Any]:
    return validate_topology_data(
        json.loads(path.read_text(encoding="utf-8")),
        copies=copies,
        r0_mode=r0_mode,
        require_reference_metadata=require_reference_metadata,
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def validate_pdb_model1(path: Path, *, max_hbond_angstrom: float = 2.30) -> dict[str, Any]:
    """Confirm the eight Hoogsteen H...acceptor distances in each MODEL-1 tetrad."""
    atoms: dict[tuple[int, str], tuple[float, float, float]] = {}
    active = True
    saw_model = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MODEL"):
            saw_model = True
            active = int(line.split()[1]) == 1
            continue
        if line.startswith("ENDMDL") and active:
            break
        if active and line.startswith(("ATOM  ", "HETATM")):
            residue = int(line[22:26])
            atom_name = line[12:16].strip()
            atoms[(residue, atom_name)] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
    _require(bool(atoms), f"No atoms read from {'MODEL 1 of ' if saw_model else ''}{path}")

    # Directed donor->acceptor cycles.  The central tetrad has opposite polarity.
    cycles = (
        (2, 10, 22, 14),
        (15, 21, 9, 3),
        (4, 8, 20, 16),
    )
    distances: list[float] = []
    for cycle in cycles:
        for donor, acceptor in zip(cycle, cycle[1:] + cycle[:1]):
            required = ((donor, "H1"), (donor, "H21"), (donor, "H22"), (acceptor, "O6"), (acceptor, "N7"))
            missing = [key for key in required if key not in atoms]
            _require(not missing, f"PDB MODEL 1 lacks atoms required for Hoogsteen check: {missing}")
            distances.append(_distance(atoms[(donor, "H1")], atoms[(acceptor, "O6")]))
            distances.append(
                min(
                    _distance(atoms[(donor, "H21")], atoms[(acceptor, "N7")]),
                    _distance(atoms[(donor, "H22")], atoms[(acceptor, "N7")]),
                )
            )

    maximum = max(distances)
    _require(
        maximum <= max_hbond_angstrom,
        f"PDB MODEL 1 does not satisfy the expected 143D Hoogsteen graph: max H...A={maximum:.3f} A",
    )
    return {
        "pdb": str(path),
        "model": 1,
        "hoogsteen_distances": len(distances),
        "mean_hbond_angstrom": sum(distances) / len(distances),
        "max_hbond_angstrom": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--pdb", type=Path)
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--r0-mode", choices=("any", "auto", "numeric"), default="any")
    parser.add_argument("--require-reference-metadata", action="store_true")
    args = parser.parse_args()

    result = validate_topology_file(
        args.topology,
        copies=args.copies,
        r0_mode=args.r0_mode,
        require_reference_metadata=args.require_reference_metadata,
    )
    if args.pdb is not None:
        result["pdb_geometry"] = validate_pdb_model1(args.pdb)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[PASS] TEL22 Morse graph matches the antiparallel basket topology of PDB 143D MODEL 1.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
