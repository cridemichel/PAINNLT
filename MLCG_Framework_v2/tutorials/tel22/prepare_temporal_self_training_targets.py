#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import analyze_conditional_noise as cn
import analyze_dna_self_vs_intercopy as dsi
import analyze_force_source_decomposition as fs
import analyze_temporal_force_averaging as tfa
import build_dna_self_isolated_dataset as bsi


def window_key(width_ps: float) -> str:
    text = f"{float(width_ps):g}".replace("-", "m").replace(".", "p")
    return f"{text}ps"


def target_scale(force: np.ndarray, torque: np.ndarray, rigid_mask: np.ndarray) -> Dict[str, float]:
    force = np.asarray(force, dtype=np.float64)
    torque = np.asarray(torque, dtype=np.float64)
    rigid_mask = np.asarray(rigid_mask, dtype=bool)
    return {
        "force_component_rms_kj_mol_nm": float(np.sqrt(np.mean(np.square(force, dtype=np.float64)))),
        "torque_component_rms_kj_mol_multisite_only": (
            float(np.sqrt(np.mean(np.square(torque[:, :, rigid_mask, :], dtype=np.float64))))
            if np.any(rigid_mask)
            else math.nan
        ),
    }


def compute_copy_rotations(
    frames: Sequence[cn.Frame],
    period: int,
    copies: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("no frames available for rotation construction")
    ref_xyz = cn.unwrap_copy_geometry(frames[0].molecules[:period], frames[0].box)
    rotations = np.empty((len(frames), copies, 3, 3), dtype=np.float64)
    orth_err = np.empty((len(frames), copies), dtype=np.float64)
    dets = np.empty((len(frames), copies), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for ti, frame in enumerate(frames):
        if len(frame.molecules) != period * copies:
            raise ValueError(
                f"frame {ti}: molecule count {len(frame.molecules)} != {period * copies}"
            )
        for ci in range(copies):
            lo = ci * period
            hi = lo + period
            xyz = cn.unwrap_copy_geometry(frame.molecules[lo:hi], frame.box)
            r = np.asarray(cn.kabsch_row(xyz, ref_xyz), dtype=np.float64)
            rotations[ti, ci] = r
            orth_err[ti, ci] = float(np.max(np.abs(r @ r.T - eye)))
            dets[ti, ci] = float(np.linalg.det(r))
    return rotations, orth_err, dets


def average_targets_in_central_lab_frame(
    force_lab: np.ndarray,
    torque_lab: np.ndarray,
    rotations_to_ref: np.ndarray,
    times: np.ndarray,
    centers: np.ndarray,
    widths: Sequence[float],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Dict[str, float]]]:
    force_lab = np.asarray(force_lab, dtype=np.float64)
    torque_lab = np.asarray(torque_lab, dtype=np.float64)
    rotations_to_ref = np.asarray(rotations_to_ref, dtype=np.float64)
    if force_lab.shape != torque_lab.shape:
        raise ValueError("force/torque shapes differ")
    if force_lab.ndim != 4 or force_lab.shape[-1] != 3:
        raise ValueError(f"expected [time,copy,residue,3] targets, got {force_lab.shape}")
    if rotations_to_ref.shape != force_lab.shape[:2] + (3, 3):
        raise ValueError(
            f"rotation shape {rotations_to_ref.shape} incompatible with targets {force_lab.shape}"
        )

    edges = tfa.sample_cell_edges(times)
    # Row-vector convention: v_ref = v_lab @ R.
    force_ref = np.einsum("tcpi,tcij->tcpj", force_lab, rotations_to_ref, optimize=True)
    torque_ref = np.einsum("tcpi,tcij->tcpj", torque_lab, rotations_to_ref, optimize=True)

    output: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    support_report: Dict[str, Dict[str, float]] = {}
    r_center = rotations_to_ref[centers]

    for width in widths:
        key = window_key(width)
        avg_f_ref = np.empty((len(centers),) + force_ref.shape[1:], dtype=np.float64)
        avg_t_ref = np.empty((len(centers),) + torque_ref.shape[1:], dtype=np.float64)
        support_counts: List[int] = []
        effective_n: List[float] = []
        span_ps: List[float] = []
        for oi, ci in enumerate(centers):
            idx, weights = tfa.boxcar_support(times, edges, int(ci), float(width))
            support_counts.append(int(len(idx)))
            effective_n.append(float(1.0 / np.sum(np.square(weights))))
            span_ps.append(float(times[idx[-1]] - times[idx[0]]) if len(idx) > 1 else 0.0)
            avg_f_ref[oi] = np.tensordot(weights, force_ref[idx], axes=(0, 0))
            avg_t_ref[oi] = np.tensordot(weights, torque_ref[idx], axes=(0, 0))

        # Back to the orientation of the central physical frame so the targets remain
        # equivariant with the unrotated central-frame geometry later written to the dataset.
        avg_f_lab = np.einsum("acpj,acij->acpi", avg_f_ref, r_center, optimize=True)
        avg_t_lab = np.einsum("acpj,acij->acpi", avg_t_ref, r_center, optimize=True)
        output[key] = (avg_f_lab.astype(np.float32), avg_t_lab.astype(np.float32))
        support_report[key] = {
            "window_ps": float(width),
            "support_samples_min": int(min(support_counts)),
            "support_samples_max": int(max(support_counts)),
            "support_samples_mean": float(np.mean(support_counts)),
            "effective_independent_samples_mean_if_uncorrelated": float(np.mean(effective_n)),
            "sample_center_span_ps_min": float(min(span_ps)),
            "sample_center_span_ps_max": float(max(span_ps)),
        }
    return output, support_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare temporally averaged DNA-self targets for PaiNN training")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--raw-topology", type=Path, required=True)
    ap.add_argument("--raw-trr", type=Path, required=True)
    ap.add_argument("--copy-dir", type=Path, required=True)
    ap.add_argument("--copy-manifest", type=Path, required=True)
    ap.add_argument("--window-ps", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--common-max-window-ps", type=float, default=None,
                    help="optional larger window used only to define the common center-frame pool")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    widths = sorted({float(x) for x in args.window_ps})
    if not widths or any((not np.isfinite(x) or x <= 0.0) for x in widths):
        raise SystemExit("--window-ps values must be finite and > 0")
    common_max_width = max(widths) if args.common_max_window_ps is None else float(args.common_max_window_ps)
    if not np.isfinite(common_max_width) or common_max_width <= 0.0:
        raise SystemExit("--common-max-window-ps must be finite and > 0")
    if common_max_width + 1.0e-12 < max(widths):
        raise SystemExit("--common-max-window-ps cannot be smaller than a requested target window")
    for path in (args.dataset, args.raw_topology, args.raw_trr, args.copy_manifest):
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")

    manifest = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])
    first_gro = args.copy_dir / "copy_00.gro"
    first_trr = args.copy_dir / "copy_00_rerun.trr"
    if not first_gro.exists() or not first_trr.exists():
        raise SystemExit(
            "full single-copy reruns are missing; run 03n_prepare_full_self_reruns.sh first"
        )

    times, _f0, _t0, sig0 = dsi.load_targets(first_gro, first_trr)
    times = np.asarray(times, dtype=np.float64)
    self_f, self_t, self_sig = dsi.load_self_targets(args.copy_dir, manifest, times)
    if sig0 != self_sig or len(self_sig) != period:
        raise RuntimeError("single-copy rerun residue signature mismatch")
    if len(times) < 3:
        raise RuntimeError("need at least three rerun frames")

    raw_indices = np.asarray(
        fs.raw_time_to_frame_indices(args.raw_topology, args.raw_trr, times), dtype=np.int64
    )
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise RuntimeError("raw time mapping produced duplicate dataset frame indices")
    frames = bsi.read_selected_frames(args.dataset, raw_indices)
    detected_period = cn.detect_repeat_period(frames[0].molecules)
    if detected_period != period or len(frames[0].molecules) != copies * period:
        raise RuntimeError("dataset repeated-copy topology does not match rerun manifest")
    rigid_mask = np.asarray([m.nsites > 1 for m in frames[0].molecules[:period]], dtype=bool)

    rotations, orth_err, dets = compute_copy_rotations(frames, period=period, copies=copies)
    edges = tfa.sample_cell_edges(times)
    centers = tfa.common_centers(times, edges, [common_max_width])

    self_f4 = np.asarray(self_f, dtype=np.float64).reshape(len(times), copies, period, 3)
    self_t4 = np.asarray(self_t, dtype=np.float64).reshape(len(times), copies, period, 3)
    averaged, support_report = average_targets_in_central_lab_frame(
        self_f4, self_t4, rotations, times, centers, widths
    )

    arrays: Dict[str, np.ndarray] = {
        "center_rerun_positions": centers.astype(np.int32),
        "center_times_ps": times[centers].astype(np.float64),
        "center_raw_dataset_indices": raw_indices[centers].astype(np.int32),
        "window_ps": np.asarray(widths, dtype=np.float64),
    }
    scales: Dict[str, Dict[str, float]] = {}
    one_ps_guardrail = None
    for width in widths:
        key = window_key(width)
        f, t = averaged[key]
        arrays[f"force_{key}"] = f
        arrays[f"torque_{key}"] = t
        scales[key] = target_scale(f, t, rigid_mask)
        if np.isclose(width, 1.0, atol=1.0e-12, rtol=0.0):
            ref_f = self_f4[centers]
            ref_t = self_t4[centers]
            max_f = float(np.max(np.abs(np.asarray(f, dtype=np.float64) - ref_f)))
            max_t = float(np.max(np.abs(np.asarray(t, dtype=np.float64) - ref_t)))
            denom_f = max(float(np.max(np.abs(ref_f))), 1.0)
            denom_t = max(float(np.max(np.abs(ref_t))), 1.0)
            one_ps_guardrail = {
                "force_max_abs_kj_mol_nm": max_f,
                "torque_max_abs_kj_mol": max_t,
                "force_max_abs_relative_to_global_max": max_f / denom_f,
                "torque_max_abs_relative_to_global_max": max_t / denom_t,
            }
            if max_f / denom_f > 2.0e-6 or max_t / denom_t > 2.0e-6:
                raise RuntimeError(
                    "1-ps target did not reproduce the instantaneous self target after round-trip rotation"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    dt = np.diff(times)
    report = {
        "definition": {
            "purpose": "prepare controlled instantaneous/temporally averaged DNA-self targets for PaiNN training",
            "target_transport": "each support-frame force/torque is rotated into a common Kabsch copy reference, boxcar-averaged there, then inverse-rotated into the central physical frame orientation",
            "geometry": "training geometry is always the central retained-CG frame; no geometry averaging is performed",
            "comparison_guardrail": "all requested windows use the same common center frames; downstream builder uses the same deterministic temporal split for every window",
            "common_pool_guardrail": "the common center pool may be restricted by a larger diagnostic window so training can be compared to an existing matched conditional-noise report without changing pair identities",
        },
        "inputs": {
            "target_frames": int(len(times)),
            "copies_per_frame": copies,
            "residues_per_copy": period,
            "time_start_ps": float(times[0]),
            "time_end_ps": float(times[-1]),
            "sampling_dt_ps_median": float(np.median(dt)),
            "windows_ps": [float(x) for x in widths],
            "common_max_window_ps": float(common_max_width),
            "common_center_frames": int(len(centers)),
            "common_center_rerun_positions": [int(x) for x in centers],
            "common_center_time_start_ps": float(times[centers[0]]),
            "common_center_time_end_ps": float(times[centers[-1]]),
        },
        "rotation_guardrails": {
            "max_orthogonality_error": float(np.max(orth_err)),
            "determinant_min": float(np.min(dets)),
            "determinant_max": float(np.max(dets)),
            "one_ps_round_trip": one_ps_guardrail,
        },
        "support": support_report,
        "target_scale": scales,
        "cache": {
            "npz": str(args.output),
            "force_key_pattern": "force_<window-key>",
            "torque_key_pattern": "torque_<window-key>",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("======================================================")
    print(" TEL22 TEMPORAL SELF TRAINING TARGET CACHE")
    print("======================================================")
    print(
        f"source frames={len(times)} | common centers={len(centers)} "
        f"({times[centers[0]]:g}..{times[centers[-1]]:g} ps) | common max window={common_max_width:g} ps"
    )
    for width in widths:
        key = window_key(width)
        s = scales[key]
        print(
            f"{key:>6s}: RMS F={s['force_component_rms_kj_mol_nm']:.3f} | "
            f"T={s['torque_component_rms_kj_mol_multisite_only']:.3f}"
        )
    if one_ps_guardrail:
        print(
            "1ps round-trip max relative error: "
            f"F={one_ps_guardrail['force_max_abs_relative_to_global_max']:.3e}, "
            f"T={one_ps_guardrail['torque_max_abs_relative_to_global_max']:.3e}"
        )
    print(f"cache:  {args.output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
