#!/usr/bin/env python3
"""Compare TEL22 residual-force datasets with Morse prior ON vs OFF.

The two datasets must have identical boxes, molecule centers and virtual-site
geometry.  Their target difference is therefore exactly the generalized force
contribution removed by the Morse prior:

    F_off - F_on = F_Morse
    T_off - T_on = T_Morse

This script checks that invariant structurally and reports component RMS,
relative scale and alignment of the Morse contribution with the OFF residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def read_dataset(path: Path):
    frames = []
    with path.open("rb") as fh:
        nframes = I32.unpack(read_exact(fh, I32.size))[0]
        if nframes <= 0:
            raise ValueError(f"invalid frame count in {path}: {nframes}")
        for frame_idx in range(nframes):
            nmol = I32.unpack(read_exact(fh, I32.size))[0]
            nsites_total = I32.unpack(read_exact(fh, I32.size))[0]
            box = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
            mols = []
            counted_sites = 0
            for _ in range(nmol):
                mol_id = I32.unpack(read_exact(fh, I32.size))[0]
                nsites = I32.unpack(read_exact(fh, I32.size))[0]
                center = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                force = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                torque = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                sites = []
                for _s in range(nsites):
                    st, x, y, z = SITE.unpack(read_exact(fh, SITE.size))
                    sites.append((int(st), float(x), float(y), float(z)))
                counted_sites += nsites
                mols.append(
                    {
                        "id": int(mol_id),
                        "nsites": int(nsites),
                        "center": center,
                        "force": force,
                        "torque": torque,
                        "sites": sites,
                    }
                )
            if counted_sites != nsites_total:
                raise ValueError(
                    f"{path}: frame {frame_idx} site-count mismatch: "
                    f"header={nsites_total}, parsed={counted_sites}"
                )
            frames.append({"box": box, "mols": mols, "nsites_total": nsites_total})
        if fh.read(1):
            raise ValueError(f"{path}: unexpected trailing bytes")
    return frames


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else float("nan")


def assert_same_geometry(on_frames, off_frames, atol=2e-7):
    if len(on_frames) != len(off_frames):
        raise ValueError(f"frame count differs: ON={len(on_frames)}, OFF={len(off_frames)}")
    max_center = 0.0
    max_site = 0.0
    max_box = 0.0
    for fi, (a, b) in enumerate(zip(on_frames, off_frames)):
        if len(a["mols"]) != len(b["mols"]):
            raise ValueError(f"frame {fi}: molecule count differs")
        if a["nsites_total"] != b["nsites_total"]:
            raise ValueError(f"frame {fi}: total site count differs")
        max_box = max(max_box, float(np.max(np.abs(a["box"] - b["box"]))))
        for ma, mb in zip(a["mols"], b["mols"]):
            if ma["id"] != mb["id"] or ma["nsites"] != mb["nsites"]:
                raise ValueError(f"frame {fi}: molecule identity/site count differs")
            max_center = max(
                max_center, float(np.max(np.abs(ma["center"] - mb["center"])))
            )
            if len(ma["sites"]) != len(mb["sites"]):
                raise ValueError(f"frame {fi} mol {ma['id']}: site count differs")
            for sa, sb in zip(ma["sites"], mb["sites"]):
                if sa[0] != sb[0]:
                    raise ValueError(f"frame {fi} mol {ma['id']}: site type differs")
                max_site = max(
                    max_site,
                    float(np.max(np.abs(np.asarray(sa[1:]) - np.asarray(sb[1:])))),
                )
    if max(max_box, max_center, max_site) > atol:
        raise ValueError(
            "ON/OFF geometry is not identical within tolerance: "
            f"box={max_box:.3g}, center={max_center:.3g}, site={max_site:.3g}, atol={atol}"
        )
    return {"max_box_delta_nm": max_box, "max_center_delta_nm": max_center, "max_site_delta_nm": max_site}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, type=Path)
    ap.add_argument("--off", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-csv", required=True, type=Path)
    args = ap.parse_args()

    on_frames = read_dataset(args.on)
    off_frames = read_dataset(args.off)
    geometry = assert_same_geometry(on_frames, off_frames)

    f_on = []
    f_off = []
    f_morse = []
    t_on = []
    t_off = []
    t_morse = []
    per_frame = []

    for fi, (on, off) in enumerate(zip(on_frames, off_frames)):
        fon = np.stack([m["force"] for m in on["mols"]])
        foff = np.stack([m["force"] for m in off["mols"]])
        ton = np.stack([m["torque"] for m in on["mols"]])
        toff = np.stack([m["torque"] for m in off["mols"]])
        fm = foff - fon
        tm = toff - ton
        rigid = np.asarray([m["nsites"] > 1 for m in on["mols"]], dtype=bool)

        f_on.append(fon)
        f_off.append(foff)
        f_morse.append(fm)
        if np.any(rigid):
            t_on.append(ton[rigid])
            t_off.append(toff[rigid])
            t_morse.append(tm[rigid])

        per_frame.append(
            {
                "frame": fi,
                "force_on_rms": rms(fon),
                "force_off_rms": rms(foff),
                "morse_force_rms": rms(fm),
                "torque_on_rms": rms(ton[rigid]) if np.any(rigid) else 0.0,
                "torque_off_rms": rms(toff[rigid]) if np.any(rigid) else 0.0,
                "morse_torque_rms": rms(tm[rigid]) if np.any(rigid) else 0.0,
            }
        )

    f_on = np.concatenate(f_on, axis=0)
    f_off = np.concatenate(f_off, axis=0)
    f_morse = np.concatenate(f_morse, axis=0)
    t_on = np.concatenate(t_on, axis=0) if t_on else np.zeros((0, 3))
    t_off = np.concatenate(t_off, axis=0) if t_off else np.zeros((0, 3))
    t_morse = np.concatenate(t_morse, axis=0) if t_morse else np.zeros((0, 3))

    f_on_rms = rms(f_on)
    f_off_rms = rms(f_off)
    fm_rms = rms(f_morse)
    t_on_rms = rms(t_on)
    t_off_rms = rms(t_off)
    tm_rms = rms(t_morse)

    report = {
        "definition": {
            "on_target": "F_ref - F_harmonic - F_WCA - F_Morse",
            "off_target": "F_ref - F_harmonic - F_WCA",
            "difference": "OFF - ON = Morse generalized force/torque",
        },
        "counts": {
            "frames": len(on_frames),
            "molecules_total_samples": int(f_on.shape[0]),
            "rigid_molecule_samples": int(t_on.shape[0]),
        },
        "geometry_identity": geometry,
        "force": {
            "residual_on_component_rms": f_on_rms,
            "residual_off_component_rms": f_off_rms,
            "morse_component_rms": fm_rms,
            "morse_over_off_residual_rms": fm_rms / f_off_rms if f_off_rms > 0 else None,
            "morse_over_on_residual_rms": fm_rms / f_on_rms if f_on_rms > 0 else None,
            "on_over_off_residual_rms": f_on_rms / f_off_rms if f_off_rms > 0 else None,
            "cosine_morse_vs_off_residual": cosine(f_morse, f_off),
            "cosine_morse_vs_on_residual": cosine(f_morse, f_on),
        },
        "torque_multisite": {
            "residual_on_component_rms": t_on_rms,
            "residual_off_component_rms": t_off_rms,
            "morse_component_rms": tm_rms,
            "morse_over_off_residual_rms": tm_rms / t_off_rms if t_off_rms > 0 else None,
            "morse_over_on_residual_rms": tm_rms / t_on_rms if t_on_rms > 0 else None,
            "on_over_off_residual_rms": t_on_rms / t_off_rms if t_off_rms > 0 else None,
            "cosine_morse_vs_off_residual": cosine(t_morse, t_off),
            "cosine_morse_vs_on_residual": cosine(t_morse, t_on),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)

    print("======================================================")
    print(" MORSE PRIOR TARGET ABLATION")
    print("======================================================")
    print(f"Frames: {len(on_frames)} | molecule samples: {f_on.shape[0]}")
    print(
        "Force component RMS: "
        f"ON={f_on_rms:.6g} | OFF={f_off_rms:.6g} | Morse={fm_rms:.6g} kJ/(mol nm)"
    )
    print(
        "Morse / OFF residual RMS: "
        f"{(fm_rms / f_off_rms if f_off_rms else float('nan')):.3%}"
    )
    print(
        "Residual RMS change ON/OFF: "
        f"{(f_on_rms / f_off_rms if f_off_rms else float('nan')):.6f}"
    )
    print(
        "cos(Morse, OFF residual): "
        f"{report['force']['cosine_morse_vs_off_residual']:.6f}"
    )
    if t_off.size:
        print(
            "Torque component RMS: "
            f"ON={t_on_rms:.6g} | OFF={t_off_rms:.6g} | Morse={tm_rms:.6g} kJ/mol"
        )
        print(
            "Morse torque / OFF residual RMS: "
            f"{(tm_rms / t_off_rms if t_off_rms else float('nan')):.3%}"
        )
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
