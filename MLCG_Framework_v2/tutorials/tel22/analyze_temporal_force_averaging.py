#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import analyze_conditional_noise as cn
import analyze_dna_self_vs_intercopy as dsi
import analyze_force_source_decomposition as fs


def sample_cell_edges(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("need at least two target times")
    dt = np.diff(times)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("target times must be finite and strictly increasing")
    edges = np.empty(len(times) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (times[:-1] + times[1:])
    edges[0] = times[0] - 0.5 * dt[0]
    edges[-1] = times[-1] + 0.5 * dt[-1]
    return edges


def boxcar_support(times: np.ndarray, edges: np.ndarray, center_i: int, width_ps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact centered boxcar weights using sample-centered time cells.

    Each stored force sample represents the Voronoi time cell around its timestamp. The
    overlap of that cell with [t-W/2, t+W/2] gives a symmetric, exactly normalized boxcar.
    For 1-ps sampling this gives, e.g., W=2 ps -> [0.25, 0.5, 0.25] and W=10 ps ->
    half-weighted endpoints plus nine full interior 1-ps cells.
    """
    width_ps = float(width_ps)
    if not np.isfinite(width_ps) or width_ps <= 0.0:
        raise ValueError(f"invalid averaging width {width_ps}")
    tc = float(times[int(center_i)])
    lo = tc - 0.5 * width_ps
    hi = tc + 0.5 * width_ps
    tol = 1.0e-10 * max(1.0, abs(tc), width_ps)
    if lo < edges[0] - tol or hi > edges[-1] + tol:
        raise ValueError("averaging window extends beyond available trajectory")
    overlap = np.maximum(0.0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
    idx = np.flatnonzero(overlap > tol)
    weights = overlap[idx] / width_ps
    s = float(np.sum(weights))
    if not np.isclose(s, 1.0, atol=1.0e-10, rtol=1.0e-10):
        raise RuntimeError(f"boxcar weights do not sum to one: width={width_ps}, sum={s}")
    return idx.astype(np.int64), weights.astype(np.float64)


def common_centers(times: np.ndarray, edges: np.ndarray, widths: Sequence[float]) -> np.ndarray:
    max_width = float(max(widths))
    half = 0.5 * max_width
    tol = 1.0e-10 * max(1.0, max_width, float(np.max(np.abs(times))))
    valid = np.flatnonzero(
        (times - half >= edges[0] - tol) &
        (times + half <= edges[-1] + tol)
    )
    if valid.size < 3:
        raise RuntimeError(f"only {valid.size} common centers remain for max window {max_width} ps")
    return valid.astype(np.int64)


def temporal_average_aligned(
    values: np.ndarray,
    descriptors: np.ndarray,
    times: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    width_ps: float,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """Average same-copy aligned targets along time; return [center,copy,...]."""
    values = np.asarray(values, dtype=np.float64)
    descriptors = np.asarray(descriptors, dtype=np.float64)
    if values.shape[:2] != descriptors.shape[:2] or values.shape[0] != len(times):
        raise ValueError("time/copy dimensions do not match for temporal averaging")

    out = np.empty((len(centers),) + values.shape[1:], dtype=np.float64)
    drift = np.empty((len(centers), values.shape[1]), dtype=np.float64)
    support_counts: List[int] = []
    effective_n: List[float] = []
    span_ps: List[float] = []

    for oi, ci in enumerate(centers):
        idx, w = boxcar_support(times, edges, int(ci), float(width_ps))
        support_counts.append(int(len(idx)))
        effective_n.append(float(1.0 / np.sum(np.square(w))))
        span_ps.append(float(times[idx[-1]] - times[idx[0]]) if len(idx) > 1 else 0.0)
        out[oi] = np.tensordot(w, values[idx], axes=(0, 0))

        delta = descriptors[idx] - descriptors[int(ci)][None, :, :]
        # For every copy: weighted RMS displacement of the retained CG coordinates from
        # the central state, after each frame has already been Kabsch-aligned to ref.
        mse_per_support_copy = np.mean(np.square(delta, dtype=np.float64), axis=2)
        drift[oi] = np.sqrt(np.tensordot(w, mse_per_support_copy, axes=(0, 0)))

    info = {
        "support_samples_min": int(min(support_counts)),
        "support_samples_max": int(max(support_counts)),
        "support_samples_mean": float(np.mean(support_counts)),
        "effective_independent_samples_mean_if_uncorrelated": float(np.mean(effective_n)),
        "sample_center_span_ps_min": float(min(span_ps)),
        "sample_center_span_ps_max": float(max(span_ps)),
    }
    return out.astype(np.float32), info, drift.reshape(-1)


def pstats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"p50": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return cn.percentile_dict(x)


def write_pair_rows(
    rows: List[Dict],
    width_ps: float,
    pair_set: str,
    control: str,
    pairs: np.ndarray,
    descriptors: np.ndarray,
    forces: np.ndarray,
    torques: np.ndarray,
    rigid_mask: np.ndarray,
    frame_ids: np.ndarray,
    frame_times: np.ndarray,
    copy_ids: np.ndarray,
) -> None:
    if len(pairs) == 0:
        return
    for i, j in pairs:
        i = int(i); j = int(j)
        geom = float(np.sqrt(np.mean(np.square(descriptors[i] - descriptors[j], dtype=np.float64))))
        df = forces[i] - forces[j]
        fpair = float(np.sqrt(np.mean(np.square(df, dtype=np.float64))))
        if np.any(rigid_mask):
            dt = torques[i, rigid_mask, :] - torques[j, rigid_mask, :]
            tpair = float(np.sqrt(np.mean(np.square(dt, dtype=np.float64))))
        else:
            tpair = math.nan
        rows.append({
            "window_ps": float(width_ps),
            "pair_set": pair_set,
            "control": control,
            "sample_i": i,
            "sample_j": j,
            "raw_frame_i": int(frame_ids[i]),
            "raw_frame_j": int(frame_ids[j]),
            "time_i_ps": float(frame_times[i]),
            "time_j_ps": float(frame_times[j]),
            "copy_i": int(copy_ids[i]),
            "copy_j": int(copy_ids[j]),
            "geometry_rmsd_nm": geom,
            "force_pair_rms_kj_mol_nm": fpair,
            "torque_pair_rms_kj_mol": tpair,
        })


def main() -> None:
    ap = argparse.ArgumentParser(description="TEL22 self-force temporal averaging conditional-noise diagnostic")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--raw-topology", type=Path, required=True)
    ap.add_argument("--raw-trr", type=Path, required=True)
    ap.add_argument("--copy-dir", type=Path, required=True)
    ap.add_argument("--copy-manifest", type=Path, required=True)
    ap.add_argument("--window-ps", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    ap.add_argument("--same-copy-gap-frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    widths = sorted({float(x) for x in args.window_ps})
    if not widths or any((not np.isfinite(x) or x <= 0.0) for x in widths):
        raise SystemExit("--window-ps values must be finite and >0")
    if args.same_copy_gap_frames < 1:
        raise SystemExit("--same-copy-gap-frames must be >=1")

    manifest = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    times, _f0, _t0, _sig0 = dsi.load_targets(args.copy_dir / "copy_00.gro", args.copy_dir / "copy_00_rerun.trr")
    times = np.asarray(times, dtype=np.float64)
    self_f, self_t, _sig = dsi.load_self_targets(args.copy_dir, manifest, times)
    self_f = np.asarray(self_f, dtype=np.float32)
    self_t = np.asarray(self_t, dtype=np.float32)
    if self_f.shape[0] != len(times):
        raise RuntimeError("self-target time dimension mismatch")

    raw_indices = np.asarray(fs.raw_time_to_frame_indices(args.raw_topology, args.raw_trr, times), dtype=np.int64)
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise RuntimeError("raw trajectory time mapping produced duplicate dataset frame indices")

    # Build the current-CG descriptor and rotate each instantaneous self target with the
    # Kabsch rotation of its own copy/frame. Temporal averaging is then performed in this
    # common co-moving reference frame, avoiding cancellation from global molecular rotation.
    full = dsi.geometry_with_targets(
        args.dataset,
        raw_indices,
        {"instant": (self_f, self_t)},
    )
    ntime = len(times)
    copies = int(full["copies"])
    period = int(full["period"])
    if len(full["descriptors"]) != ntime * copies:
        raise RuntimeError("unexpected flattened sample count")

    descriptors3 = np.asarray(full["descriptors"], dtype=np.float32).reshape(ntime, copies, -1)
    force3 = np.asarray(full["instant"]["forces"], dtype=np.float32).reshape(ntime, copies, period, 3)
    torque3 = np.asarray(full["instant"]["torques"], dtype=np.float32).reshape(ntime, copies, period, 3)

    expected_frames = np.repeat(raw_indices, copies)
    expected_copies = np.tile(np.arange(copies, dtype=np.int16), ntime)
    if not np.array_equal(np.asarray(full["frame_ids"]), expected_frames):
        raise RuntimeError("geometry sample ordering is not frame-major as expected")
    if not np.array_equal(np.asarray(full["copy_ids"]), expected_copies):
        raise RuntimeError("geometry sample ordering is not copy-minor as expected")

    edges = sample_cell_edges(times)
    centers = common_centers(times, edges, widths)
    center_desc = descriptors3[centers].reshape(len(centers) * copies, -1)
    center_frame_ids = np.repeat(raw_indices[centers], copies).astype(np.int32)
    center_times = np.repeat(times[centers], copies).astype(np.float64)
    center_copy_ids = np.tile(np.arange(copies, dtype=np.int16), len(centers))
    all_idx = np.arange(len(center_desc), dtype=np.int64)

    nearest_pairs = cn.nearest_same_copy_gap(
        center_desc,
        center_copy_ids,
        center_frame_ids,
        args.same_copy_gap_frames,
    )
    rng = np.random.default_rng(args.seed)
    random_pairs = cn.random_control_pairs(
        rng,
        len(nearest_pairs),
        all_idx,
        center_copy_ids,
        center_frame_ids,
        "same_copy_gap",
        args.same_copy_gap_frames,
    )
    if len(nearest_pairs) == 0 or len(random_pairs) == 0:
        raise RuntimeError("failed to construct same-copy nearest/random pairs")

    rigid_mask = np.asarray(full["rigid_mask"], dtype=bool)
    labels = list(full["labels"])
    report_windows: Dict[str, Dict] = {}
    pair_rows: List[Dict] = []
    baseline_key = None

    for width in widths:
        key = f"{width:g}ps"
        if baseline_key is None:
            baseline_key = key
        avg_f, support_info, drift = temporal_average_aligned(
            force3, descriptors3, times, edges, centers, width
        )
        avg_t, support_info_t, drift_t = temporal_average_aligned(
            torque3, descriptors3, times, edges, centers, width
        )
        if support_info != support_info_t or not np.allclose(drift, drift_t, atol=0.0, rtol=0.0):
            raise RuntimeError("force/torque temporal support bookkeeping diverged")

        f = avg_f.reshape(len(centers) * copies, period, 3)
        t = avg_t.reshape(len(centers) * copies, period, 3)
        f_rms = cn.rms_components(f)
        t_rms = cn.rms_components(t[:, rigid_mask, :]) if np.any(rigid_mask) else math.nan
        nearest, _ = cn.pair_metrics(
            f"temporal_{key}_nearest_same_copy_gap",
            nearest_pairs,
            center_desc,
            f,
            t,
            labels,
            rigid_mask,
            f_rms,
            t_rms,
        )
        random, _ = cn.pair_metrics(
            f"temporal_{key}_nearest_same_copy_gap_random",
            random_pairs,
            center_desc,
            f,
            t,
            labels,
            rigid_mask,
            f_rms,
            t_rms,
        )
        nearest_f_frac = float(nearest["force_half_pair_difference_mse_fraction_of_target_mse"])
        nearest_t_frac = float(nearest["torque_half_pair_difference_mse_fraction_of_target_mse"])
        absolute_half_f_mse = nearest_f_frac * f_rms * f_rms
        absolute_half_t_mse = nearest_t_frac * t_rms * t_rms if np.isfinite(t_rms) else math.nan

        report_windows[key] = {
            "window_ps": float(width),
            "support": support_info,
            "target_scale": {
                "force_component_rms_kj_mol_nm": float(f_rms),
                "torque_component_rms_kj_mol_multisite_only": float(t_rms),
            },
            "center_to_support_cg_rmsd_nm": pstats(drift),
            "nearest_same_copy_gap": nearest,
            "random_control": random,
            "absolute_half_pair_mse": {
                "force_kj2_mol2_nm2": float(absolute_half_f_mse),
                "torque_kj2_mol2": float(absolute_half_t_mse),
            },
            "nearest_vs_random": {
                "force_half_mse_ratio": float(nearest_f_frac / random["force_half_pair_difference_mse_fraction_of_target_mse"]),
                "torque_half_mse_ratio": float(nearest_t_frac / random["torque_half_pair_difference_mse_fraction_of_target_mse"]),
            },
        }

        write_pair_rows(pair_rows, width, "nearest_same_copy_gap", "nearest", nearest_pairs,
                        center_desc, f, t, rigid_mask, center_frame_ids, center_times, center_copy_ids)
        write_pair_rows(pair_rows, width, "nearest_same_copy_gap", "random", random_pairs,
                        center_desc, f, t, rigid_mask, center_frame_ids, center_times, center_copy_ids)

    if baseline_key is None:
        raise RuntimeError("no averaging windows analyzed")
    base = report_windows[baseline_key]
    base_f_frac = float(base["nearest_same_copy_gap"]["force_half_pair_difference_mse_fraction_of_target_mse"])
    base_t_frac = float(base["nearest_same_copy_gap"]["torque_half_pair_difference_mse_fraction_of_target_mse"])
    base_f_rms = float(base["target_scale"]["force_component_rms_kj_mol_nm"])
    base_t_rms = float(base["target_scale"]["torque_component_rms_kj_mol_multisite_only"])
    base_abs_f = float(base["absolute_half_pair_mse"]["force_kj2_mol2_nm2"])
    base_abs_t = float(base["absolute_half_pair_mse"]["torque_kj2_mol2"])

    comparison = {}
    for key, item in report_windows.items():
        f_frac = float(item["nearest_same_copy_gap"]["force_half_pair_difference_mse_fraction_of_target_mse"])
        t_frac = float(item["nearest_same_copy_gap"]["torque_half_pair_difference_mse_fraction_of_target_mse"])
        f_rms = float(item["target_scale"]["force_component_rms_kj_mol_nm"])
        t_rms = float(item["target_scale"]["torque_component_rms_kj_mol_multisite_only"])
        abs_f = float(item["absolute_half_pair_mse"]["force_kj2_mol2_nm2"])
        abs_t = float(item["absolute_half_pair_mse"]["torque_kj2_mol2"])
        comparison[key] = {
            "window_ps": float(item["window_ps"]),
            "force_normalized_noise_floor_ratio_vs_baseline": float(f_frac / base_f_frac),
            "torque_normalized_noise_floor_ratio_vs_baseline": float(t_frac / base_t_frac),
            "force_target_rms_ratio_vs_baseline": float(f_rms / base_f_rms),
            "torque_target_rms_ratio_vs_baseline": float(t_rms / base_t_rms),
            "force_absolute_half_pair_mse_ratio_vs_baseline": float(abs_f / base_abs_f),
            "torque_absolute_half_pair_mse_ratio_vs_baseline": float(abs_t / base_abs_t),
            "median_center_to_support_cg_rmsd_nm": float(item["center_to_support_cg_rmsd_nm"]["p50"]),
        }

    dt = np.diff(times)
    report = {
        "definition": {
            "purpose": "test whether fast atomistic self-force fluctuations dominate the ~0.8 TEL22 conditional-noise floor",
            "descriptor": "current retained CG geometry at the central frame; same nearest/random pair identities are reused for every averaging width",
            "target_transport": "instantaneous same-copy generalized forces/torques are Kabsch-rotated with each support frame into the common copy reference frame before temporal averaging",
            "temporal_kernel": "exact centered boxcar based on overlap with sample-centered time cells",
            "baseline": f"{baseline_key}; with 1-ps sampling, a 1-ps boxcar contains exactly the central instantaneous target",
            "target": "single-copy GROMACS self generalized force/torque; no water, ions, or other TEL22 copies",
        },
        "inputs": {
            "target_frames": int(len(times)),
            "common_center_frames": int(len(centers)),
            "copies_per_frame": copies,
            "residues_per_copy": period,
            "current_cg_sites_per_copy": int(full["sites_per_copy"]),
            "time_start_ps": float(times[0]),
            "time_end_ps": float(times[-1]),
            "sampling_dt_ps_median": float(np.median(dt)),
            "sampling_dt_ps_min": float(np.min(dt)),
            "sampling_dt_ps_max": float(np.max(dt)),
            "common_center_time_start_ps": float(times[centers[0]]),
            "common_center_time_end_ps": float(times[centers[-1]]),
            "same_copy_min_gap_frames": int(args.same_copy_gap_frames),
            "windows_ps": [float(x) for x in widths],
            "nearest_pairs": int(len(nearest_pairs)),
            "random_pairs": int(len(random_pairs)),
            "seed": int(args.seed),
        },
        "windows": report_windows,
        "primary_comparison_vs_baseline": comparison,
        "interpretation": {
            "fast_noise_signal": "A strong, monotonic drop of the normalized noise-floor ratio below 1 while center-to-support CG drift remains small indicates that fast force fluctuations are being averaged away.",
            "mapping_state_signal": "If the normalized floor stays near 1 even when absolute force variance decreases, temporal smoothing does not recover predictability; missing CG state variables / mapping resolution remain the stronger explanation.",
            "large_window_guardrail": "Do not overinterpret windows whose center-to-support CG RMSD is comparable to or larger than the nearest-pair geometry RMSD: those windows average across materially different CG states.",
            "same_pairs_guardrail": "All windows use exactly the same central frames and pair identities, so changes in the reported floor come from target averaging rather than nearest-neighbor reselection.",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_ps", "pair_set", "control", "sample_i", "sample_j",
        "raw_frame_i", "raw_frame_j", "time_i_ps", "time_j_ps", "copy_i", "copy_j",
        "geometry_rmsd_nm", "force_pair_rms_kj_mol_nm", "torque_pair_rms_kj_mol",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    print("[TEMPORAL AVERAGING] common centers:", len(centers), "pairs:", len(nearest_pairs))
    for key, item in comparison.items():
        win = report_windows[key]
        ff = win["nearest_same_copy_gap"]["force_half_pair_difference_mse_fraction_of_target_mse"]
        tf = win["nearest_same_copy_gap"]["torque_half_pair_difference_mse_fraction_of_target_mse"]
        print(
            f"  {key:>6s}: Fhalf={ff:.6f} xbase={item['force_normalized_noise_floor_ratio_vs_baseline']:.4f} "
            f"| Thalf={tf:.6f} xbase={item['torque_normalized_noise_floor_ratio_vs_baseline']:.4f} "
            f"| Frms xbase={item['force_target_rms_ratio_vs_baseline']:.4f} "
            f"| CG drift p50={item['median_center_to_support_cg_rmsd_nm']:.5f} nm"
        )
    print("[WRITE]", args.output_json)
    print("[WRITE]", args.output_csv)


if __name__ == "__main__":
    main()
