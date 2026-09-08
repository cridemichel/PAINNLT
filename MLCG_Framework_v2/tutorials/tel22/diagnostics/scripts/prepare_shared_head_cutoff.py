#!/usr/bin/env python3
"""Scan TEL22 shared-head primitive edges and write a dataset-bound config."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


CONTACTS = (
    (1, 9, 2, 4), (1, 13, 4, 2), (1, 21, 4, 2),
    (9, 13, 2, 4), (9, 21, 2, 4), (13, 21, 4, 2),
    (2, 8, 4, 2), (2, 14, 2, 4), (2, 20, 2, 4),
    (8, 14, 4, 2), (8, 20, 4, 2), (14, 20, 2, 4),
    (3, 7, 2, 4), (3, 15, 4, 2), (3, 19, 4, 2),
    (7, 15, 2, 4), (7, 19, 2, 4), (15, 19, 4, 2),
)
STACKING = ((1, 2), (2, 3), (9, 8), (8, 7),
            (13, 14), (14, 15), (21, 20), (20, 19))
SITE_NAMES = ("S", "B1", "B2", "B3", "B4", "B5")


def read_exact(handle, size: int, label: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"Truncated dataset while reading {label}")
    return value


def read_int(handle, label: str) -> int:
    return struct.unpack("<i", read_exact(handle, 4, label))[0]


def read_vec3(handle, label: str) -> tuple[float, float, float]:
    return struct.unpack("<3f", read_exact(handle, 12, label))


def expected_layout(residue: int) -> tuple[int, tuple[int, ...]]:
    if residue in {0, 6, 12, 18}:
        return 1, (0,)
    if residue in {4, 5, 10, 11, 16, 17}:
        return 1, (1,)
    return 6, (2, 3, 4, 5, 6, 7)


def scan_dataset(path: Path, physical_cutoff: float) -> dict:
    maxima: dict[str, dict] = {}
    global_maximum: dict | None = None
    observations: dict[str, int] = {}
    outside_cutoff: dict[str, int] = {}
    with path.open("rb") as handle:
        frames = read_int(handle, "frame count")
        if frames <= 0:
            raise ValueError("Dataset has no frames")
        for frame_index in range(frames):
            molecules = read_int(handle, "molecule count")
            sites_total = read_int(handle, "site count")
            box = read_vec3(handle, "box")
            if molecules != 220 or sites_total != 820:
                raise ValueError(
                    f"Frame {frame_index}: expected 220 molecules/820 sites, "
                    f"found {molecules}/{sites_total}"
                )
            frame_sites: list[list[tuple[int, tuple[float, float, float]]]] = []
            for molecule in range(molecules):
                molecule_id = read_int(handle, "molecule id")
                site_count = read_int(handle, "molecule site count")
                read_vec3(handle, "molecule center")
                read_vec3(handle, "target force")
                read_vec3(handle, "target torque")
                if molecule_id != molecule:
                    raise ValueError(f"Frame {frame_index}: nonsequential molecule id")
                expected_count, expected_types = expected_layout(molecule % 22)
                if site_count != expected_count:
                    raise ValueError(
                        f"Frame {frame_index}, molecule {molecule}: site-count mismatch"
                    )
                sites = []
                for site in range(site_count):
                    site_type = read_int(handle, "site type")
                    position = read_vec3(handle, "site position")
                    if site_type != expected_types[site]:
                        raise ValueError(
                            f"Frame {frame_index}, molecule {molecule}, site {site}: "
                            "site-type mismatch"
                        )
                    sites.append((site_type, position))
                frame_sites.append(sites)

            def observe(category: str, copy: int, residue_i: int, site_i: int,
                        residue_j: int, site_j: int) -> None:
                nonlocal global_maximum
                position_i = frame_sites[copy * 22 + residue_i][site_i][1]
                position_j = frame_sites[copy * 22 + residue_j][site_j][1]
                delta = []
                for axis in range(3):
                    raw = position_i[axis] - position_j[axis]
                    delta.append(raw - box[axis] * round(raw / box[axis]))
                distance = math.sqrt(sum(value * value for value in delta))
                record = {
                    "distance_nm": distance,
                    "frame_zero_based": frame_index,
                    "copy_zero_based": copy,
                    "residue_i_one_based": residue_i + 1,
                    "site_i": SITE_NAMES[site_i],
                    "residue_j_one_based": residue_j + 1,
                    "site_j": SITE_NAMES[site_j],
                    "category": category,
                }
                observations[category] = observations.get(category, 0) + 1
                if distance > physical_cutoff:
                    outside_cutoff[category] = outside_cutoff.get(category, 0) + 1
                if category not in maxima or distance > maxima[category]["distance_nm"]:
                    maxima[category] = record
                if global_maximum is None or distance > global_maximum["distance_nm"]:
                    global_maximum = record

            for copy in range(10):
                for residue in range(21):
                    observe("backbone_anchor", copy, residue, 0, residue + 1, 0)
                for residue_i, residue_j, site_i, site_j in CONTACTS:
                    observe("tetrad_b3", copy, residue_i, 3, residue_j, 3)
                    observe("tetrad_oriented", copy, residue_i, site_i, residue_j, site_j)
                for residue_i, residue_j in STACKING:
                    observe("stacking_b3", copy, residue_i, 3, residue_j, 3)
                    observe("stacking_b5", copy, residue_i, 5, residue_j, 5)
        if handle.read(1):
            raise ValueError("Dataset contains trailing bytes")
    assert global_maximum is not None
    return {
        "frames": frames,
        "global_maximum": global_maximum,
        "category_maxima": maxima,
        "observations": observations,
        "outside_cutoff": outside_cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config-in", required=True, type=Path)
    parser.add_argument("--config-out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config_in.read_text(encoding="utf-8"))
    configured_cutoff = float(config["cutoff"])
    if configured_cutoff <= 0.0:
        raise ValueError("Configured physical cutoff must be positive")
    scan = scan_dataset(args.dataset, configured_cutoff)
    observed_maximum = float(scan["global_maximum"]["distance_nm"])
    config["ordered_geometry_edge_policy"] = "augment_required_topology_edges_v1"
    args.config_out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    total_observations = sum(scan["observations"].values())
    total_outside_cutoff = sum(scan["outside_cutoff"].values())
    report = {
        "schema_version": 2,
        "kind": "tel22_shared_head_cutoff_preflight",
        "dataset": str(args.dataset.resolve()),
        "frames": scan["frames"],
        "configured_cutoff_nm": configured_cutoff,
        "selected_cutoff_nm": configured_cutoff,
        "edge_policy": "augment_required_topology_edges_v1",
        "observed_maximum": scan["global_maximum"],
        "category_maxima": scan["category_maxima"],
        "mandatory_edge_observations": total_observations,
        "mandatory_edge_observations_outside_cutoff": total_outside_cutoff,
        "outside_cutoff_by_category": scan["outside_cutoff"],
        "structural_excursion_above_1_5_nm": observed_maximum > 1.5,
        "status": "PASS",
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"[PASS] TEL22 physical cutoff retained at {configured_cutoff:.6f} nm; "
        f"{total_outside_cutoff}/{total_observations} required descriptor observations "
        "will use explicit topology-edge augmentation "
        f"(observed max {observed_maximum:.6f} nm)."
    )


if __name__ == "__main__":
    main()
