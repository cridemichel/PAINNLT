#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

import analyze_conditional_noise as cn
import build_dna_self_isolated_dataset as bsi
import prepare_temporal_self_training_targets as pts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build isolated-copy PaiNN dataset from temporally averaged DNA-self targets")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--copy-manifest", type=Path, required=True)
    ap.add_argument("--target-cache", type=Path, required=True)
    ap.add_argument("--target-cache-report", type=Path, required=True)
    ap.add_argument("--window-ps", type=float, required=True)
    ap.add_argument("--sample-count", type=int, default=None)
    ap.add_argument("--validation-stride", type=int, default=5)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.50)
    ap.add_argument("--layout-cols", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    for path in (args.dataset, args.config, args.copy_manifest, args.target_cache, args.target_cache_report):
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")
    if args.validation_stride < 2:
        raise SystemExit("validation-stride must be >= 2 for the controlled comparison")
    if args.margin <= 0.0:
        raise SystemExit("margin must be > 0")
    if args.layout_cols <= 0:
        raise SystemExit("layout-cols must be > 0")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cutoff = float(cfg["cutoff"] if args.cutoff is None else args.cutoff)
    if cutoff <= 0.0:
        raise SystemExit("cutoff must be > 0")

    manifest: Dict = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])
    cache_report = json.loads(args.target_cache_report.read_text(encoding="utf-8"))
    key = pts.window_key(args.window_ps)

    with np.load(args.target_cache, allow_pickle=False) as z:
        force_key = f"force_{key}"
        torque_key = f"torque_{key}"
        if force_key not in z.files or torque_key not in z.files:
            available = sorted(k for k in z.files if k.startswith("force_"))
            raise SystemExit(f"window {args.window_ps:g} ps not in target cache; available={available}")
        center_raw = np.asarray(z["center_raw_dataset_indices"], dtype=np.int64)
        center_times = np.asarray(z["center_times_ps"], dtype=np.float64)
        center_rerun_positions = np.asarray(z["center_rerun_positions"], dtype=np.int64)
        all_f = np.asarray(z[force_key], dtype=np.float32)
        all_t = np.asarray(z[torque_key], dtype=np.float32)

    n_available = len(center_raw)
    expected_shape = (n_available, copies, period, 3)
    if all_f.shape != expected_shape or all_t.shape != expected_shape:
        raise RuntimeError(
            f"cached target shape mismatch: F={all_f.shape}, T={all_t.shape}, expected={expected_shape}"
        )
    if len(center_times) != n_available or len(center_rerun_positions) != n_available:
        raise RuntimeError("target cache center metadata length mismatch")

    sample_count = n_available if args.sample_count is None else int(args.sample_count)
    selected = bsi.evenly_spaced_indices(n_available, sample_count)
    train_pos, val_pos, dataset_order = bsi.stratified_tail_order(sample_count, args.validation_stride)
    selected_centers = selected[dataset_order]

    raw_indices = center_raw[selected_centers]
    source_frames = bsi.read_selected_frames(args.dataset, raw_indices)
    if not source_frames:
        raise RuntimeError("no source frames selected")
    detected_period = cn.detect_repeat_period(source_frames[0].molecules)
    if detected_period != period or len(source_frames[0].molecules) != copies * period:
        raise RuntimeError("dataset repeated-copy topology does not match copy manifest")

    force = all_f[selected_centers].reshape(sample_count, copies * period, 3)
    torque = all_t[selected_centers].reshape(sample_count, copies * period, 3)
    output_frames, report = bsi.build_frames(
        source_frames,
        force,
        torque,
        period=period,
        copies=copies,
        cutoff=cutoff,
        margin=float(args.margin),
        layout_cols=int(args.layout_cols),
    )

    report["definition"] = {
        "purpose": "controlled PaiNN ablation with temporally averaged DNA-self generalized targets",
        "target": f"single-copy GROMACS DNA-self generalized force/torque averaged over a centered {args.window_ps:g}-ps boxcar in a co-moving Kabsch copy frame and transported back to the central physical-frame orientation",
        "geometry": "retained CG geometry of the central frame; each TEL22 copy is translated rigidly to a sparse periodic lattice",
        "intercopy_model_edges": "forbidden geometrically: every cross-copy site distance is greater than the PaiNN cutoff",
        "comparison_guardrail": "all temporal windows are drawn from the same common center-frame pool and use the same deterministic train/validation split",
        "production_guardrail": "diagnostic target only; temporal averaging defines a variance-reduced force-matching target and is not yet a production force estimator prescription",
    }
    report["temporal_target"] = {
        "window_ps": float(args.window_ps),
        "window_key": key,
        "target_cache": str(args.target_cache),
        "target_cache_report": str(args.target_cache_report),
        "support": cache_report.get("support", {}).get(key),
        "cache_target_scale": cache_report.get("target_scale", {}).get(key),
    }
    report["sampling"] = {
        "available_common_center_frames": int(n_available),
        "selected_frames": int(sample_count),
        "selection": "rounded linspace across the common temporal-center pool",
        "selected_common_center_positions_in_chronological_order": [int(x) for x in selected],
    }
    report["split"] = {
        "mode": "stratified_temporal_tail_v1",
        "validation_stride": int(args.validation_stride),
        "train_frames": int(len(train_pos)),
        "validation_frames": int(len(val_pos)),
        "dataset_binary_order": "all training frames first, then validation frames",
        "trainer_config": {
            "validation_split_mode": "tail",
            "validation_tail_frames": int(len(val_pos)),
        },
        "train_common_center_positions": [int(x) for x in selected[train_pos]],
        "validation_common_center_positions": [int(x) for x in selected[val_pos]],
        "train_rerun_positions": [int(x) for x in center_rerun_positions[selected[train_pos]]],
        "validation_rerun_positions": [int(x) for x in center_rerun_positions[selected[val_pos]]],
    }
    report["inputs"] = {
        "dataset": str(args.dataset),
        "copy_manifest": str(args.copy_manifest),
        "target_mode": f"temporal_self_{key}",
        "window_ps": float(args.window_ps),
        "center_times_ps_in_dataset_order": [float(x) for x in center_times[selected_centers]],
        "center_raw_dataset_indices_in_dataset_order": [int(x) for x in raw_indices],
        "center_rerun_positions_in_dataset_order": [int(x) for x in center_rerun_positions[selected_centers]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bsi.write_dataset(args.output, output_frames)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("======================================================")
    print(" TEL22 TEMPORAL SELF TRAINING DATASET")
    print("======================================================")
    print(
        f"window={args.window_ps:g} ps | frames={sample_count} | "
        f"train={len(train_pos)} | val={len(val_pos)}"
    )
    print(
        f"RMS F={report['target_scale']['force_component_rms_kj_mol_nm']:.3f} | "
        f"T={report['target_scale']['torque_component_rms_kj_mol_multisite_only']:.3f}"
    )
    print(
        f"cutoff={cutoff:.4f} nm | min inter-copy distance="
        f"{report['isolation']['minimum_intercopy_site_distance_nm']:.4f} nm"
    )
    print(f"dataset: {args.output}")
    print(f"report:  {args.report}")


if __name__ == "__main__":
    main()
