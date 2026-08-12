#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_conditional_noise as cn
import build_dna_self_isolated_dataset as bsi


def main() -> None:
    ap = argparse.ArgumentParser(description="Build isolated-copy PaiNN dataset from aggforce-mapped instantaneous self forces")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--copy-manifest", type=Path, required=True)
    ap.add_argument("--target-cache", type=Path, required=True)
    ap.add_argument("--aggforce-report", type=Path, required=True)
    ap.add_argument("--variant", choices=["current", "constraint_aware", "optimized"], required=True)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.50)
    ap.add_argument("--layout-cols", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    for p in (args.dataset, args.config, args.copy_manifest, args.target_cache, args.aggforce_report):
        if not p.exists():
            raise SystemExit(f"required file not found: {p}")
    if args.margin <= 0.0 or args.layout_cols <= 0:
        raise SystemExit("margin/layout-cols must be > 0")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cutoff = float(cfg["cutoff"] if args.cutoff is None else args.cutoff)
    manifest = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    agg = json.loads(args.aggforce_report.read_text(encoding="utf-8"))
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])

    with np.load(args.target_cache, allow_pickle=False) as z:
        centers = np.asarray(z["center_rerun_positions"], dtype=np.int64)
        center_raw = np.asarray(z["center_raw_dataset_indices"], dtype=np.int64)
        center_times = np.asarray(z["center_times_ps"], dtype=np.float64)
        train_pos = np.asarray(z["train_center_positions"], dtype=np.int64)
        val_pos = np.asarray(z["validation_center_positions"], dtype=np.int64)
        order = np.asarray(z["dataset_center_order"], dtype=np.int64)
        force_all = np.asarray(z[f"force_{args.variant}"], dtype=np.float32)
        torque_all = np.asarray(z["torque_current"], dtype=np.float32)

    ntime = force_all.shape[0]
    if force_all.shape != (ntime, copies, period, 3) or torque_all.shape != force_all.shape:
        raise RuntimeError(f"target cache shape mismatch F={force_all.shape} T={torque_all.shape}")
    if len(order) != len(centers) or set(order.tolist()) != set(range(len(centers))):
        raise RuntimeError("dataset center order is not a permutation of common centers")
    if not np.array_equal(order, np.concatenate([train_pos, val_pos])):
        raise RuntimeError("cache split order is not train then validation")

    raw_ordered = center_raw[order]
    source_frames = bsi.read_selected_frames(args.dataset, raw_ordered)
    if not source_frames:
        raise RuntimeError("no source frames selected")
    detected_period = cn.detect_repeat_period(source_frames[0].molecules)
    if detected_period != period or len(source_frames[0].molecules) != copies * period:
        raise RuntimeError("dataset repeated-copy topology does not match copy manifest")

    rerun_ordered = centers[order]
    force = force_all[rerun_ordered].reshape(len(order), copies * period, 3)
    torque = torque_all[rerun_ordered].reshape(len(order), copies * period, 3)
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
        "purpose": "PaiNN ablation using instantaneous DNA-self forces with aggforce variance reduction on exact single-site COM residues",
        "force_target": f"{args.variant} instantaneous self-force target; aggforce changes only exact single-site COM DA/DT rows and leaves multi-site rigid DG at the original atom-sum target",
        "torque_target": "original per-residue instantaneous self torque; one-site DA/DT are masked by the trainer and 03r keeps torque_weight=0.5 exactly as 03q",
        "geometry": "unchanged retained CG geometry at the same common center frames; copies are translated to a sparse lattice exactly as in 03m/03q",
        "force_map_fit_guardrail": "optimized/constraint-aware DA/DT maps were constructed only from training-center atomistic samples; validation center frames were excluded and multi-site DG was not remapped",
    }
    report["aggforce_target"] = {
        "variant": args.variant,
        "software": agg.get("software"),
        "diagnostic": agg.get("force_noise_diagnostic", {}).get(args.variant),
        "mapping_guardrail": agg.get("mapping_guardrails", {}).get(args.variant),
        "source_report": str(args.aggforce_report),
    }
    report["sampling"] = {
        "selected_frames": int(len(order)),
        "common_center_frames": int(len(centers)),
        "center_time_start_ps": float(center_times[0]),
        "center_time_end_ps": float(center_times[-1]),
    }
    report["split"] = {
        "mode": "stratified_temporal_tail_v1",
        "train_frames": int(len(train_pos)),
        "validation_frames": int(len(val_pos)),
        "dataset_binary_order": "all training frames first, then validation frames",
        "trainer_config": {
            "validation_split_mode": "tail",
            "validation_tail_frames": int(len(val_pos)),
        },
        "train_common_center_positions": [int(x) for x in train_pos],
        "validation_common_center_positions": [int(x) for x in val_pos],
        "train_raw_dataset_indices": [int(x) for x in center_raw[train_pos]],
        "validation_raw_dataset_indices": [int(x) for x in center_raw[val_pos]],
    }
    report["inputs"] = {
        "dataset": str(args.dataset),
        "target_cache": str(args.target_cache),
        "variant": args.variant,
        "center_raw_dataset_indices_in_binary_order": [int(x) for x in raw_ordered],
        "center_times_ps_in_binary_order": [float(x) for x in center_times[order]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bsi.write_dataset(args.output, output_frames)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("======================================================")
    print(" TEL22 AGGFORCE SELF TRAINING DATASET")
    print("======================================================")
    print(
        f"variant={args.variant} | frames={len(order)} | train={len(train_pos)} | val={len(val_pos)} | "
        f"F RMS={report['target_scale']['force_component_rms_kj_mol_nm']:.3f}"
    )
    print(f"dataset: {args.output}")
    print(f"report:  {args.report}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_conditional_noise as cn
import build_dna_self_isolated_dataset as bsi


def main() -> None:
    ap = argparse.ArgumentParser(description="Build isolated-copy PaiNN dataset from aggforce-mapped instantaneous self forces")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--copy-manifest", type=Path, required=True)
    ap.add_argument("--target-cache", type=Path, required=True)
    ap.add_argument("--aggforce-report", type=Path, required=True)
    ap.add_argument("--variant", choices=["current", "constraint_aware", "optimized"], required=True)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.50)
    ap.add_argument("--layout-cols", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    for p in (args.dataset, args.config, args.copy_manifest, args.target_cache, args.aggforce_report):
        if not p.exists():
            raise SystemExit(f"required file not found: {p}")
    if args.margin <= 0.0 or args.layout_cols <= 0:
        raise SystemExit("margin/layout-cols must be > 0")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cutoff = float(cfg["cutoff"] if args.cutoff is None else args.cutoff)
    manifest = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    agg = json.loads(args.aggforce_report.read_text(encoding="utf-8"))
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])

    with np.load(args.target_cache, allow_pickle=False) as z:
        centers = np.asarray(z["center_rerun_positions"], dtype=np.int64)
        center_raw = np.asarray(z["center_raw_dataset_indices"], dtype=np.int64)
        center_times = np.asarray(z["center_times_ps"], dtype=np.float64)
        train_pos = np.asarray(z["train_center_positions"], dtype=np.int64)
        val_pos = np.asarray(z["validation_center_positions"], dtype=np.int64)
        order = np.asarray(z["dataset_center_order"], dtype=np.int64)
        force_all = np.asarray(z[f"force_{args.variant}"], dtype=np.float32)
        torque_all = np.asarray(z["torque_current"], dtype=np.float32)

    ntime = force_all.shape[0]
    if force_all.shape != (ntime, copies, period, 3) or torque_all.shape != force_all.shape:
        raise RuntimeError(f"target cache shape mismatch F={force_all.shape} T={torque_all.shape}")
    if len(order) != len(centers) or set(order.tolist()) != set(range(len(centers))):
        raise RuntimeError("dataset center order is not a permutation of common centers")
    if not np.array_equal(order, np.concatenate([train_pos, val_pos])):
        raise RuntimeError("cache split order is not train then validation")

    raw_ordered = center_raw[order]
    source_frames = bsi.read_selected_frames(args.dataset, raw_ordered)
    if not source_frames:
        raise RuntimeError("no source frames selected")
    detected_period = cn.detect_repeat_period(source_frames[0].molecules)
    if detected_period != period or len(source_frames[0].molecules) != copies * period:
        raise RuntimeError("dataset repeated-copy topology does not match copy manifest")

    rerun_ordered = centers[order]
    force = force_all[rerun_ordered].reshape(len(order), copies * period, 3)
    torque = torque_all[rerun_ordered].reshape(len(order), copies * period, 3)
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
        "purpose": "PaiNN ablation using instantaneous DNA-self forces with aggforce variance reduction on exact single-site COM residues",
        "force_target": f"{args.variant} instantaneous self-force target; aggforce changes only exact single-site COM DA/DT rows and leaves multi-site rigid DG at the original atom-sum target",
        "torque_target": "original per-residue instantaneous self torque; one-site DA/DT are masked by the trainer and 03r keeps torque_weight=0.5 exactly as 03q",
        "geometry": "unchanged retained CG geometry at the same common center frames; copies are translated to a sparse lattice exactly as in 03m/03q",
        "force_map_fit_guardrail": "optimized/constraint-aware DA/DT maps were constructed only from training-center atomistic samples; validation center frames were excluded and multi-site DG was not remapped",
    }
    report["aggforce_target"] = {
        "variant": args.variant,
        "software": agg.get("software"),
        "diagnostic": agg.get("force_noise_diagnostic", {}).get(args.variant),
        "mapping_guardrail": agg.get("mapping_guardrails", {}).get(args.variant),
        "source_report": str(args.aggforce_report),
    }
    report["sampling"] = {
        "selected_frames": int(len(order)),
        "common_center_frames": int(len(centers)),
        "center_time_start_ps": float(center_times[0]),
        "center_time_end_ps": float(center_times[-1]),
    }
    report["split"] = {
        "mode": "stratified_temporal_tail_v1",
        "train_frames": int(len(train_pos)),
        "validation_frames": int(len(val_pos)),
        "dataset_binary_order": "all training frames first, then validation frames",
        "trainer_config": {
            "validation_split_mode": "tail",
            "validation_tail_frames": int(len(val_pos)),
        },
        "train_common_center_positions": [int(x) for x in train_pos],
        "validation_common_center_positions": [int(x) for x in val_pos],
        "train_raw_dataset_indices": [int(x) for x in center_raw[train_pos]],
        "validation_raw_dataset_indices": [int(x) for x in center_raw[val_pos]],
    }
    report["inputs"] = {
        "dataset": str(args.dataset),
        "target_cache": str(args.target_cache),
        "variant": args.variant,
        "center_raw_dataset_indices_in_binary_order": [int(x) for x in raw_ordered],
        "center_times_ps_in_binary_order": [float(x) for x in center_times[order]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bsi.write_dataset(args.output, output_frames)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("======================================================")
    print(" TEL22 AGGFORCE SELF TRAINING DATASET")
    print("======================================================")
    print(
        f"variant={args.variant} | frames={len(order)} | train={len(train_pos)} | val={len(val_pos)} | "
        f"F RMS={report['target_scale']['force_component_rms_kj_mol_nm']:.3f}"
    )
    print(f"dataset: {args.output}")
    print(f"report:  {args.report}")


if __name__ == "__main__":
    main()
