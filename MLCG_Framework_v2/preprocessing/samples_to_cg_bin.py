#!/usr/bin/env python3
"""Converte i campioni di una corsa CG (run_cg_md.py --sample_npz) nel formato
binario del dataset, cosi' che train_relent li legga come i frame AA.

Il dataset di riferimento (--template) fornisce la topologia: numero di corpi,
siti per corpo e loro tipi.  I campioni portano solo le posizioni (COM e siti
virtuali), ordinate per (molecola, sito) come in run_cg_md; si verifica che
l'ordine coincida con quello del template prima di scrivere.  Forze e coppie
si scrivono a zero: train_relent non le usa, e nessun frame CG deve poter
finire per errore in un allenamento a force matching (train_painn li
tratterebbe come esche, i "decoy" a residuo nullo, e li scarterebbe).

Uso:
  python3 samples_to_cg_bin.py --template tel26_lp1_dataset.bin \\
      --samples samples_relent_it01.npz [altri.npz ...] \\
      [--skip-ps 5] [--stride 1] --out relent_cg_it01.bin
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read_template(path: Path):
    """Topologia del primo frame: lista di (num_siti, [tipi]) per corpo."""
    with path.open("rb") as f:
        (num_frames,) = struct.unpack("i", f.read(4))
        if num_frames <= 0:
            raise ValueError(f"{path}: nessun frame")
        num_molecules, num_total_sites = struct.unpack("ii", f.read(8))
        f.read(12)  # box
        bodies = []
        for m in range(num_molecules):
            mol_id, num_sites = struct.unpack("ii", f.read(8))
            if mol_id != m:
                raise ValueError(f"{path}: molecule_id non sequenziale ({mol_id} != {m})")
            f.read(36)  # centro, forza, coppia
            types = []
            for _ in range(num_sites):
                (stype,) = struct.unpack("i", f.read(4))
                f.read(12)
                types.append(stype)
            bodies.append(types)
    if sum(len(t) for t in bodies) != num_total_sites:
        raise ValueError(f"{path}: num_total_sites incoerente")
    return bodies


def load_samples(path: Path, skip_ps: float, stride: int):
    with np.load(path, allow_pickle=False) as d:
        if int(d.get("complete", 1)) != 1:
            raise ValueError(f"{path}: corsa incompleta")
        time_ps = np.asarray(d["time_ps"], dtype=float)
        keep = np.flatnonzero(time_ps >= time_ps[0] + skip_ps)[::stride]
        return {
            "com": np.asarray(d["com"], dtype=float)[keep],
            "sites": np.asarray(d["sites"], dtype=float)[keep],
            "site_molecule": np.asarray(d["site_molecule"], dtype=np.int64),
            "site_index": np.asarray(d["site_index"], dtype=np.int64),
            "box": np.asarray(d["box"], dtype=float),
            "time_ps": time_ps[keep],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--samples", required=True, type=Path, nargs="+")
    ap.add_argument("--skip-ps", type=float, default=0.0,
                    help="scarta i primi ps di ogni corsa (rilassamento dopo il cambio di modello)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride deve essere >= 1")
    if args.out.exists():
        raise SystemExit(f"[ERROR] esiste gia': {args.out}")

    bodies = read_template(args.template)
    expected_mol = np.asarray([m for m, t in enumerate(bodies) for _ in t], dtype=np.int64)
    expected_idx = np.asarray([s for t in bodies for s in range(len(t))], dtype=np.int64)
    flat_types = [s for t in bodies for s in t]

    frames = []
    for path in args.samples:
        run = load_samples(path, args.skip_ps, args.stride)
        if not np.array_equal(run["site_molecule"], expected_mol) or \
           not np.array_equal(run["site_index"], expected_idx):
            raise SystemExit(f"[ERROR] {path}: ordine dei siti diverso dal template")
        if run["com"].shape[1] != len(bodies):
            raise SystemExit(f"[ERROR] {path}: {run['com'].shape[1]} corpi, attesi {len(bodies)}")
        if not np.all(np.isfinite(run["sites"])):
            raise SystemExit(f"[ERROR] {path}: coordinate non finite")
        print(f"[INFO] {path}: {len(run['time_ps'])} frame "
              f"({run['time_ps'][0]:.2f}-{run['time_ps'][-1]:.2f} ps)" if len(run["time_ps"])
              else f"[WARNING] {path}: nessun frame dopo --skip-ps")
        frames.append(run)
    total = sum(len(r["time_ps"]) for r in frames)
    if total == 0:
        raise SystemExit("[ERROR] nessun frame da scrivere")

    zeros9 = struct.pack("9f", *([0.0] * 9))
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("i", total))
        for run in frames:
            box = struct.pack("3f", *run["box"])
            for com, sites in zip(run["com"], run["sites"]):
                f.write(struct.pack("ii", len(bodies), len(flat_types)))
                f.write(box)
                k = 0
                for m, types in enumerate(bodies):
                    f.write(struct.pack("ii", m, len(types)))
                    # Il centro e' il COM: serve solo a chi volesse le coppie.
                    f.write(struct.pack("3f", *com[m]))
                    f.write(zeros9[:24])
                    for stype in types:
                        f.write(struct.pack("i3f", stype, *sites[k]))
                        k += 1
    tmp.rename(args.out)
    print(f"[DONE] {total} frame CG -> {args.out}")


if __name__ == "__main__":
    main()
