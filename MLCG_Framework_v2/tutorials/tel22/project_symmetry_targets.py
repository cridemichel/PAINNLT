#!/usr/bin/env python3
"""Project CG force-matching targets onto the global symmetry-compatible subspace.

Diagnostic only.  The input dataset is never modified in place.

For each physical frame:
  1. Remove the uniform net-force mode from molecular target forces:
       F_i <- F_i - mean_j F_j
  2. Using PBC-aware molecule centers, compute the remaining global generalized
     torque about the frame center:
       T = sum_i r_i x F_i + sum_{rigid i} tau_i
  3. Remove that residual torque uniformly from torque-bearing (multi-site)
     rigid bodies:
       tau_i <- tau_i - T / N_rigid

This enforces the two exact global constraints obeyed by an isolated internal
potential: sum_i F_i = 0 and sum_i(r_i x F_i + tau_i) = 0.

Zero-target OOD decoys are copied unchanged so the existing trainer can still
identify them and keep them out of physical validation.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

I32 = struct.Struct("=i")
F32_3 = struct.Struct("=3f")
SITE = struct.Struct("=ifff")


def read_exact(handle, n):
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"unexpected EOF: requested {n} bytes, got {len(data)}")
    return data


def read_dataset(path):
    frames = []
    with open(path, "rb") as fh:
        nframes = I32.unpack(read_exact(fh, I32.size))[0]
        for _ in range(nframes):
            nmol = I32.unpack(read_exact(fh, I32.size))[0]
            nsites_total = I32.unpack(read_exact(fh, I32.size))[0]
            box = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
            mols = []
            counted_sites = 0
            for _m in range(nmol):
                mol_id = I32.unpack(read_exact(fh, I32.size))[0]
                nsites = I32.unpack(read_exact(fh, I32.size))[0]
                center = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                force = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                torque = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                sites = []
                for _s in range(nsites):
                    st, x, y, z = SITE.unpack(read_exact(fh, SITE.size))
                    sites.append((st, x, y, z))
                counted_sites += nsites
                mols.append({
                    "id": mol_id,
                    "nsites": nsites,
                    "center": center,
                    "force": force,
                    "torque": torque,
                    "sites": sites,
                })
            if counted_sites != nsites_total:
                raise ValueError(
                    f"binary frame site count mismatch: header={nsites_total}, parsed={counted_sites}"
                )
            frames.append({"box": box, "mols": mols, "nsites_total": nsites_total})
        if fh.read(1):
            raise ValueError("unexpected trailing bytes after dataset")
    return frames


def write_dataset(path, frames):
    with open(path, "wb") as fh:
        fh.write(I32.pack(len(frames)))
        for frame in frames:
            mols = frame["mols"]
            fh.write(I32.pack(len(mols)))
            fh.write(I32.pack(frame["nsites_total"]))
            fh.write(F32_3.pack(*np.asarray(frame["box"], dtype=np.float32)))
            for mol in mols:
                fh.write(I32.pack(int(mol["id"])))
                fh.write(I32.pack(int(mol["nsites"])))
                fh.write(F32_3.pack(*np.asarray(mol["center"], dtype=np.float32)))
                fh.write(F32_3.pack(*np.asarray(mol["force"], dtype=np.float32)))
                fh.write(F32_3.pack(*np.asarray(mol["torque"], dtype=np.float32)))
                for site in mol["sites"]:
                    fh.write(SITE.pack(int(site[0]), float(site[1]), float(site[2]), float(site[3])))


def is_zero_target_decoy(frame):
    for mol in frame["mols"]:
        if np.any(mol["force"] != 0.0) or np.any(mol["torque"] != 0.0):
            return False
    return bool(frame["mols"])


def centered_mic_coordinates(centers, box):
    """Return PBC-aware centers relative to a common origin, then center them."""
    ref = centers[0].copy()
    rel = centers - ref
    for k in range(3):
        L = float(box[k])
        if L > 0.0:
            rel[:, k] -= L * np.rint(rel[:, k] / L)
    rel -= rel.mean(axis=0, keepdims=True)
    return rel


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def project_frames(frames):
    physical = 0
    decoys = 0
    frames_without_rigid = 0

    raw_force_sum2 = 0.0
    force_corr_sum2 = 0.0
    n_force_components = 0
    raw_torque_sum2 = 0.0
    torque_corr_sum2 = 0.0
    n_torque_components = 0

    net_f_before = []
    net_f_after = []
    net_t_before = []
    net_t_after = []

    for frame in frames:
        if is_zero_target_decoy(frame):
            decoys += 1
            continue
        physical += 1
        mols = frame["mols"]
        if not mols:
            continue

        forces_raw = np.stack([m["force"] for m in mols], axis=0)
        torques_raw = np.stack([m["torque"] for m in mols], axis=0)
        centers = np.stack([m["center"] for m in mols], axis=0)
        rigid = np.asarray([m["nsites"] > 1 for m in mols], dtype=bool)
        r = centered_mic_coordinates(centers, frame["box"])

        fnet = forces_raw.sum(axis=0)
        forces_proj = forces_raw - fnet[None, :] / float(len(mols))

        # Decouple translation and rotation: define the global torque after
        # removing net translation, so it is origin independent.
        torque_global = np.cross(r, forces_proj).sum(axis=0)
        if np.any(rigid):
            torque_global += torques_raw[rigid].sum(axis=0)

        torques_proj = torques_raw.copy()
        nrigid = int(rigid.sum())
        if nrigid > 0:
            torques_proj[rigid] -= torque_global[None, :] / float(nrigid)
        else:
            frames_without_rigid += 1

        fnet_after = forces_proj.sum(axis=0)
        torque_after = np.cross(r, forces_proj).sum(axis=0)
        if nrigid > 0:
            torque_after += torques_proj[rigid].sum(axis=0)

        net_f_before.append(float(np.linalg.norm(fnet)))
        net_f_after.append(float(np.linalg.norm(fnet_after)))
        net_t_before.append(float(np.linalg.norm(torque_global)))
        net_t_after.append(float(np.linalg.norm(torque_after)))

        raw_force_sum2 += float(np.square(forces_raw).sum())
        force_corr_sum2 += float(np.square(forces_proj - forces_raw).sum())
        n_force_components += int(forces_raw.size)

        if nrigid > 0:
            raw_torque_sum2 += float(np.square(torques_raw[rigid]).sum())
            torque_corr_sum2 += float(np.square(torques_proj[rigid] - torques_raw[rigid]).sum())
            n_torque_components += int(torques_raw[rigid].size)

        for idx, mol in enumerate(mols):
            mol["force"] = forces_proj[idx]
            if rigid[idx]:
                mol["torque"] = torques_proj[idx]

    force_mse = raw_force_sum2 / max(1, n_force_components)
    force_corr_mse = force_corr_sum2 / max(1, n_force_components)
    torque_mse = raw_torque_sum2 / max(1, n_torque_components)
    torque_corr_mse = torque_corr_sum2 / max(1, n_torque_components)

    report = {
        "projection": {
            "translation": "subtract frame mean molecular force",
            "rotation": "after translation projection, subtract residual global generalized torque uniformly from multi-site molecular torques",
            "pbc_reference": "molecule centers MIC-unwrapped relative to first center, then mean-centered",
        },
        "counts": {
            "frames_total": len(frames),
            "physical_frames": physical,
            "zero_target_decoys_unchanged": decoys,
            "physical_frames_without_multisite_rigid_body": frames_without_rigid,
        },
        "force": {
            "raw_component_mse": force_mse,
            "projection_correction_component_mse": force_corr_mse,
            "global_translation_fraction_of_raw_mse": force_corr_mse / force_mse if force_mse > 0 else 0.0,
            "net_force_norm_before": {
                "p50": percentile(net_f_before, 50),
                "p95": percentile(net_f_before, 95),
                "max": max(net_f_before, default=0.0),
            },
            "net_force_norm_after": {
                "p95": percentile(net_f_after, 95),
                "max": max(net_f_after, default=0.0),
            },
        },
        "torque": {
            "raw_multisite_component_mse": torque_mse,
            "projection_correction_component_mse": torque_corr_mse,
            "global_rotation_correction_fraction_of_raw_torque_mse": torque_corr_mse / torque_mse if torque_mse > 0 else 0.0,
            "residual_global_torque_norm_before": {
                "p50": percentile(net_t_before, 50),
                "p95": percentile(net_t_before, 95),
                "max": max(net_t_before, default=0.0),
            },
            "residual_global_torque_norm_after": {
                "p95": percentile(net_t_after, 95),
                "max": max(net_t_after, default=0.0),
            },
        },
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)
    if in_path.resolve() == out_path.resolve():
        raise SystemExit("Refusing in-place projection: output must differ from input")

    frames = read_dataset(in_path)
    report = project_frames(frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(out_path, frames)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    c = report["counts"]
    f = report["force"]
    t = report["torque"]
    print("[SYMPROJ] Dataset scritto:", out_path)
    print(f"[SYMPROJ] frames: physical={c['physical_frames']} | decoy unchanged={c['zero_target_decoys_unchanged']}")
    print("[SYMPROJ] global translation fraction of force MSE: "
          f"{100.0*f['global_translation_fraction_of_raw_mse']:.6f}%")
    print("[SYMPROJ] global rotation correction / raw torque MSE: "
          f"{100.0*t['global_rotation_correction_fraction_of_raw_torque_mse']:.6f}%")
    print("[SYMPROJ] post-projection max |sum F|: "
          f"{f['net_force_norm_after']['max']:.6g}")
    print("[SYMPROJ] post-projection max |global torque|: "
          f"{t['residual_global_torque_norm_after']['max']:.6g}")
    print("[SYMPROJ] report:", report_path)


if __name__ == "__main__":
    main()
