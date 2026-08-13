#!/usr/bin/env python3
"""Diagnose whether a short-range CG pair is protected by the runtime WCA prior.

The script is deliberately post-processing only: it does not import ESPResSo and
never changes a trajectory.  It reconstructs the particle-ID layout used by
run_cg_md.py from the first dataset frame, combines it with the explicit
site-aware WCA exclusions stored in cg_priors.json, and analyzes the
``min_pids``/``min_pair`` columns written to an NVE energy CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
from typing import Any


def _read_exact(handle, nbytes: int) -> bytes:
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"Unexpected end of dataset while reading {nbytes} bytes")
    return data


def _unpack(handle, fmt: str):
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))


def read_dataset_particle_map(path: str | Path) -> dict[int, dict[str, Any]]:
    """Reconstruct the particle IDs created by run_cg_md.py from frame zero."""
    path = Path(path)
    records: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        (num_frames,) = _unpack(handle, "i")
        (num_molecules,) = _unpack(handle, "i")
        (num_total_sites,) = _unpack(handle, "i")
        box = _unpack(handle, "3f")

        pid = 0
        site_count = 0
        for mol_idx in range(num_molecules):
            (dataset_mol_id,) = _unpack(handle, "i")
            (num_sites,) = _unpack(handle, "i")
            center = _unpack(handle, "3f")
            _unpack(handle, "3f")  # mapped force
            _unpack(handle, "3f")  # mapped torque

            site_types: list[int] = []
            site_positions: list[tuple[float, float, float]] = []
            for _site_idx in range(num_sites):
                (site_type,) = _unpack(handle, "i")
                site_pos = _unpack(handle, "3f")
                site_types.append(int(site_type))
                site_positions.append(tuple(float(x) for x in site_pos))

            # run_cg_md.py adds one COM particle first, then all virtual sites.
            records[pid] = {
                "pid": pid,
                "kind": "com",
                "runtime_mol_idx": mol_idx,
                "dataset_mol_id": int(dataset_mol_id),
                "site_index": None,
                "site_type": None,
                "position_frame0_nm": [float(x) for x in center],
                "molecule_site_types": site_types,
            }
            pid += 1

            for site_idx, (site_type, site_pos) in enumerate(zip(site_types, site_positions)):
                records[pid] = {
                    "pid": pid,
                    "kind": "virtual_site",
                    "runtime_mol_idx": mol_idx,
                    "dataset_mol_id": int(dataset_mol_id),
                    "site_index": site_idx,
                    "site_type": int(site_type),
                    "position_frame0_nm": list(site_pos),
                    "molecule_site_types": site_types,
                }
                pid += 1
                site_count += 1

    if site_count != num_total_sites:
        raise ValueError(
            f"Dataset header reports {num_total_sites} sites, parsed {site_count}"
        )
    if num_frames < 1:
        raise ValueError("Dataset contains no frames")

    records[-1] = {
        "metadata": {
            "num_frames": int(num_frames),
            "num_molecules": int(num_molecules),
            "num_total_sites": int(num_total_sites),
            "box_nm": [float(x) for x in box],
            "num_runtime_particles": pid,
        }
    }
    return records


def load_type_names(topology_path: str | Path | None) -> tuple[dict[int, str], dict[tuple[int, ...], str]]:
    if topology_path is None:
        return {}, {}
    data = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    mapping = data.get("mapping", {})
    site_types = mapping.get("site_types", {})
    names = {int(value): str(name) for name, value in site_types.items()}

    residue_signatures: dict[tuple[int, ...], str] = {}
    for residue, site_map in mapping.get("residues", {}).items():
        if not isinstance(site_map, dict):
            continue
        try:
            signature = tuple(int(site_types[name]) for name in site_map)
        except KeyError:
            continue
        residue_signatures[signature] = str(residue)
    return names, residue_signatures


def parse_pair(value: str) -> tuple[int, int]:
    left, right = value.split(":", 1)
    return int(left), int(right)


def read_energy_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Step", "Time_ps", "min_dist", "min_pair", "min_pids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Energy CSV is missing columns: {sorted(missing)}")
        for raw in reader:
            if not raw.get("min_pids") or not raw.get("min_pair"):
                continue
            try:
                distance = float(raw["min_dist"])
                step = int(raw["Step"])
                time_ps = float(raw["Time_ps"])
                type_pair = parse_pair(raw["min_pair"])
                pid_pair = parse_pair(raw["min_pids"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance):
                continue
            rows.append({
                "step": step,
                "time_ps": time_ps,
                "distance_nm": distance,
                "type_pair": type_pair,
                "pid_pair": pid_pair,
            })
    if not rows:
        raise ValueError("Energy CSV contains no usable min-distance rows")
    return rows


def _pair_set(raw_pairs: Any) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for raw in raw_pairs or []:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        a, b = int(raw[0]), int(raw[1])
        result.add((min(a, b), max(a, b)))
    return result


def find_wca_pair(priors: dict[str, Any], type_i: int, type_j: int) -> dict[str, Any] | None:
    lo, hi = sorted((int(type_i), int(type_j)))
    direct_key = f"{lo}_{hi}"
    candidate = priors.get("wca_pairs", {}).get(direct_key)
    if isinstance(candidate, dict):
        return candidate
    for value in priors.get("wca_pairs", {}).values():
        if not isinstance(value, dict):
            continue
        vi, vj = sorted((int(value.get("type_i", -1)), int(value.get("type_j", -1))))
        if (vi, vj) == (lo, hi):
            return value
    return None


def wca_energy_force(distance_nm: float, params: dict[str, Any] | None) -> dict[str, Any] | None:
    if params is None:
        return None
    sigma = float(params["sigma_nm"])
    epsilon = float(params["epsilon_kjmol"])
    cutoff = float(params["cutoff_nm"])
    if distance_nm <= 0.0:
        return {
            "inside_cutoff": True,
            "energy_kjmol": math.inf,
            "force_magnitude_kjmol_nm": math.inf,
        }
    if distance_nm >= cutoff:
        return {
            "inside_cutoff": False,
            "energy_kjmol": 0.0,
            "force_magnitude_kjmol_nm": 0.0,
        }
    sr6 = (sigma / distance_nm) ** 6
    sr12 = sr6 * sr6
    # ESPResSo shift="auto" shifts LJ to zero at cutoff.  For a standard
    # WCA cutoff at 2^(1/6)*sigma the shift is +epsilon.  Use the general
    # cutoff expression so this diagnostic also remains correct for custom cutoffs.
    src6 = (sigma / cutoff) ** 6
    src12 = src6 * src6
    shift = -4.0 * epsilon * (src12 - src6)
    energy = 4.0 * epsilon * (sr12 - sr6) + shift
    force = abs(24.0 * epsilon / distance_nm * (2.0 * sr12 - sr6))
    return {
        "inside_cutoff": True,
        "energy_kjmol": energy,
        "force_magnitude_kjmol_nm": force,
    }


def topology_context(
    priors: dict[str, Any], mol_i: int, site_i: int, mol_j: int, site_j: int
) -> dict[str, Any]:
    pair = (min(mol_i, mol_j), max(mol_i, mol_j))
    meta = priors.get("wca_exclusions", {})
    direct = _pair_set(meta.get("direct_pairs"))
    one_three = _pair_set(meta.get("one_three_pairs"))

    direct_site_records = set()
    for raw in meta.get("direct_site_pairs", []):
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        mi, mj, si, sj = map(int, raw)
        direct_site_records.add(
            (mi, mj, si, sj) if mi < mj else (mj, mi, sj, si)
        )
    site_record = (mol_i, mol_j, site_i, site_j) if mol_i < mol_j else (mol_j, mol_i, site_j, site_i)

    if mol_i == mol_j:
        classification = "same_molecule"
        excluded = True
        exclusion_reason = "intra-rigid-body virtual-site exclusion"
    elif pair in direct:
        classification = "1-2"
        if meta.get("pair_source") == "explicit_topology_pairs_v2":
            # Historical policy: every cross-site pair of a 1-2 molecule pair was excluded.
            excluded = True
            exclusion_reason = "legacy v2 molecule-pair 1-2 all-sites WCA exclusion"
        else:
            excluded = site_record in direct_site_records
            exclusion_reason = (
                "explicit bonded-site 1-2 WCA exclusion"
                if excluded else
                "1-2 molecule pair, but this virtual-site pair retains WCA under policy v3"
            )
    elif pair in one_three:
        classification = "1-3"
        excluded = True
        exclusion_reason = "explicit molecule-pair 1-3 all-sites WCA exclusion"
    else:
        classification = "nonbonded"
        excluded = False
        exclusion_reason = None

    direct_bonds = []
    for bond in priors.get("bonds", []):
        try:
            bp = tuple(sorted((int(bond["mol_i"]), int(bond["mol_j"]))))
        except (KeyError, TypeError, ValueError):
            continue
        if bp == pair:
            direct_bonds.append({
                key: bond.get(key)
                for key in ("name", "type", "mol_i", "mol_j", "site_i", "site_j", "exclude_wca")
            })

    endpoint_angles = []
    for angle in priors.get("angles", []):
        try:
            ap = tuple(sorted((int(angle["mol_i"]), int(angle["mol_k"]))))
        except (KeyError, TypeError, ValueError):
            continue
        if ap == pair:
            endpoint_angles.append({
                key: angle.get(key)
                for key in (
                    "name", "type", "mol_i", "mol_j", "mol_k",
                    "site_i", "site_j", "site_k", "exclude_wca",
                )
            })

    return {
        "molecule_pair": list(pair),
        "site_pair": [int(site_i), int(site_j)],
        "classification": classification,
        "wca_excluded_by_topology": excluded,
        "exclusion_reason": exclusion_reason,
        "direct_bonds": direct_bonds,
        "endpoint_angles": endpoint_angles,
    }


def annotate_particle(record: dict[str, Any], type_names: dict[int, str], residue_signatures: dict[tuple[int, ...], str]) -> dict[str, Any]:
    out = dict(record)
    site_type = out.get("site_type")
    if site_type is not None:
        out["site_type_name"] = type_names.get(int(site_type), f"type_{site_type}")
    signature = tuple(int(x) for x in out.get("molecule_site_types", []))
    out["molecule_residue_name"] = residue_signatures.get(signature)
    return out


def diagnose_row(
    row: dict[str, Any],
    particle_map: dict[int, dict[str, Any]],
    priors: dict[str, Any],
    type_names: dict[int, str],
    residue_signatures: dict[tuple[int, ...], str],
) -> dict[str, Any]:
    pid_i, pid_j = row["pid_pair"]
    if pid_i not in particle_map or pid_j not in particle_map:
        raise ValueError(f"PID pair {row['pid_pair']} is not present in the reconstructed dataset map")
    part_i = particle_map[pid_i]
    part_j = particle_map[pid_j]
    if part_i.get("kind") != "virtual_site" or part_j.get("kind") != "virtual_site":
        raise ValueError(f"Expected virtual-site PIDs, got {part_i.get('kind')} and {part_j.get('kind')}")

    logged_types = tuple(sorted(int(x) for x in row["type_pair"]))
    mapped_types = tuple(sorted((int(part_i["site_type"]), int(part_j["site_type"]))))
    if logged_types != mapped_types:
        raise ValueError(
            f"Logged type pair {logged_types} disagrees with dataset PID map {mapped_types}"
        )

    top = topology_context(
        priors,
        int(part_i["runtime_mol_idx"]), int(part_i["site_index"]),
        int(part_j["runtime_mol_idx"]), int(part_j["site_index"]),
    )
    wca = find_wca_pair(priors, *mapped_types)
    nominal = wca_energy_force(float(row["distance_nm"]), wca)
    runtime_active = bool(wca is not None and not top["wca_excluded_by_topology"])

    result = {
        "step": int(row["step"]),
        "time_ps": float(row["time_ps"]),
        "distance_nm": float(row["distance_nm"]),
        "logged_type_pair": list(logged_types),
        "logged_type_names": [type_names.get(t, f"type_{t}") for t in logged_types],
        "pid_pair": [int(pid_i), int(pid_j)],
        "particle_i": annotate_particle(part_i, type_names, residue_signatures),
        "particle_j": annotate_particle(part_j, type_names, residue_signatures),
        "topology": top,
        "wca_pair_present": wca is not None,
        "wca_runtime_expected_active": runtime_active,
        "wca_parameters": wca,
        "wca_nominal_at_observed_distance": nominal,
    }
    if nominal is not None and top["wca_excluded_by_topology"]:
        result["wca_nominal_note"] = (
            "Counterfactual only: this is the WCA energy/force that the type pair would have "
            "if this site-level topology exclusion were absent."
        )
    elif nominal is not None:
        result["wca_nominal_note"] = (
            "This WCA type pair should be active at runtime unless an additional ESPResSo exclusion exists."
        )
    return result


def build_report(
    *,
    dataset: str | Path,
    priors_path: str | Path,
    energy_csv: str | Path,
    topology_path: str | Path | None = None,
    threshold_nm: float = 0.20,
) -> dict[str, Any]:
    if threshold_nm <= 0.0:
        raise ValueError("threshold_nm must be positive")
    particle_map = read_dataset_particle_map(dataset)
    metadata = particle_map.pop(-1)["metadata"]
    priors = json.loads(Path(priors_path).read_text(encoding="utf-8"))
    type_names, residue_signatures = load_type_names(topology_path)
    rows = read_energy_rows(energy_csv)

    minimum = min(rows, key=lambda row: row["distance_nm"])
    below = [row for row in rows if row["distance_nm"] <= threshold_nm]

    primary = diagnose_row(minimum, particle_map, priors, type_names, residue_signatures)
    first_below = (
        diagnose_row(below[0], particle_map, priors, type_names, residue_signatures)
        if below else None
    )

    episodes: dict[tuple[int, int], dict[str, Any]] = {}
    for row in below:
        key = tuple(sorted(row["pid_pair"]))
        episode = episodes.setdefault(key, {
            "pid_pair": list(key),
            "samples": 0,
            "first_time_ps": row["time_ps"],
            "last_time_ps": row["time_ps"],
            "minimum_distance_nm": row["distance_nm"],
        })
        episode["samples"] += 1
        episode["last_time_ps"] = row["time_ps"]
        episode["minimum_distance_nm"] = min(episode["minimum_distance_nm"], row["distance_nm"])

    for key, episode in episodes.items():
        representative = min(
            (row for row in below if tuple(sorted(row["pid_pair"])) == key),
            key=lambda row: row["distance_nm"],
        )
        detail = diagnose_row(representative, particle_map, priors, type_names, residue_signatures)
        episode["type_pair"] = detail["logged_type_pair"]
        episode["type_names"] = detail["logged_type_names"]
        episode["molecule_pair"] = detail["topology"]["molecule_pair"]
        episode["classification"] = detail["topology"]["classification"]
        episode["wca_excluded_by_topology"] = detail["topology"]["wca_excluded_by_topology"]
        episode["wca_runtime_expected_active"] = detail["wca_runtime_expected_active"]

    conclusion: str
    if primary["topology"]["wca_excluded_by_topology"]:
        conclusion = (
            "The closest pair is covered by a WCA type prior but this specific site pair is explicitly "
            "excluded by topology, so the nominal WCA repulsion is not applied to this contact."
            if primary["wca_pair_present"] else
            "The closest site pair is explicitly excluded by topology and no WCA type prior is present."
        )
    elif primary["wca_pair_present"]:
        conclusion = (
            "The closest pair is nonbonded and has a WCA type prior that should be active. "
            "A very short distance in this class requires checking the runtime ESPResSo interaction/exclusion state."
        )
    else:
        conclusion = (
            "The closest pair is nonbonded but no pair-specific WCA prior exists for its site types."
        )

    return {
        "inputs": {
            "dataset": str(Path(dataset).resolve()),
            "priors": str(Path(priors_path).resolve()),
            "energy_csv": str(Path(energy_csv).resolve()),
            "topology": str(Path(topology_path).resolve()) if topology_path else None,
        },
        "dataset_metadata": metadata,
        "threshold_nm": threshold_nm,
        "energy_rows": len(rows),
        "rows_at_or_below_threshold": len(below),
        "minimum_observed": primary,
        "first_at_or_below_threshold": first_below,
        "short_range_episodes": sorted(episodes.values(), key=lambda item: item["minimum_distance_nm"]),
        "conclusion": conclusion,
    }


def print_summary(report: dict[str, Any]) -> None:
    item = report["minimum_observed"]
    top = item["topology"]
    pi = item["particle_i"]
    pj = item["particle_j"]
    nominal = item["wca_nominal_at_observed_distance"]

    print("[SHORT-RANGE DIAGNOSTIC]")
    print(
        f"  closest: t={item['time_ps']:.6g} ps step={item['step']} "
        f"r={item['distance_nm']:.6g} nm"
    )
    print(
        f"  types:   {item['logged_type_pair'][0]}:{item['logged_type_pair'][1]} "
        f"({item['logged_type_names'][0]} / {item['logged_type_names'][1]})"
    )
    print(
        f"  PIDs:    {item['pid_pair'][0]}:{item['pid_pair'][1]} -> "
        f"mol {pi['runtime_mol_idx']} site {pi['site_index']} / "
        f"mol {pj['runtime_mol_idx']} site {pj['site_index']}"
    )
    if pi.get("molecule_residue_name") or pj.get("molecule_residue_name"):
        print(
            f"  residues: {pi.get('molecule_residue_name') or '?'} / "
            f"{pj.get('molecule_residue_name') or '?'}"
        )
    print(
        f"  topology: {top['classification']} | "
        f"WCA excluded={top['wca_excluded_by_topology']} | "
        f"runtime WCA expected active={item['wca_runtime_expected_active']}"
    )
    if item["wca_parameters"] is not None:
        p = item["wca_parameters"]
        print(
            f"  WCA: sigma={float(p['sigma_nm']):.6g} nm "
            f"epsilon={float(p['epsilon_kjmol']):.6g} kJ/mol "
            f"cutoff={float(p['cutoff_nm']):.6g} nm"
        )
    if nominal is not None:
        print(
            f"  nominal WCA at r: U={nominal['energy_kjmol']:.6g} kJ/mol "
            f"|F|={nominal['force_magnitude_kjmol_nm']:.6g} kJ/(mol nm)"
        )
        if top["wca_excluded_by_topology"]:
            print("  note: nominal WCA values above are counterfactual because topology excludes this pair.")
    print(f"  conclusion: {report['conclusion']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="CG dataset binary used by run_cg_md.py")
    parser.add_argument("--priors", required=True, help="cg_priors.json")
    parser.add_argument("--energy-csv", required=True, help="NVE energy.csv written by run_cg_md.py")
    parser.add_argument(
        "--topology",
        default=None,
        help="Optional topology/config JSON containing mapping.site_types for human-readable names",
    )
    parser.add_argument(
        "--threshold-nm",
        type=float,
        default=0.20,
        help="Short-range threshold used to summarize recurring PID pairs (default: 0.20 nm)",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = build_report(
        dataset=args.dataset,
        priors_path=args.priors,
        energy_csv=args.energy_csv,
        topology_path=args.topology,
        threshold_nm=args.threshold_nm,
    )
    print_summary(report)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"  report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
