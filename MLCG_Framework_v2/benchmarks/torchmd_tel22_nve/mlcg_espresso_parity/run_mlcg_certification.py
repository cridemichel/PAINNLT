#!/usr/bin/env python3
"""Run ESPResSo/LibTorch NVE on the exact shared synthetic PaiNN Hamiltonian."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from analysis import analyze_energy_series, certify_metrics  # noqa: E402

DEFAULT_DTS = (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", default="../results/shared_painn_case")
    p.add_argument("--precision", choices=("float32", "float64"), default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--duration-ps", type=float, default=0.60)
    p.add_argument("--dts", nargs="+", type=float, default=list(DEFAULT_DTS))
    p.add_argument("--output-dir", default="../results/mlcg_painn_cpu_float64")
    p.add_argument("--slope-min", type=float, default=1.7)
    p.add_argument("--slope-max", type=float, default=2.3)
    p.add_argument("--min-r2", type=float, default=0.97)
    p.add_argument("--max-relative-drift", type=float, default=1.0e-4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker-dt", type=float, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def load_case(case_dir: Path):
    meta = json.loads((case_dir / "metadata.json").read_text())
    with np.load(case_dir / "state_mlcg_units.npz") as npz:
        state = {key: np.asarray(npz[key]) for key in npz.files}
    return meta, state


def kinetic_energy(system) -> float:
    total = 0.0
    for p in system.part:
        v = np.asarray(p.v, dtype=np.float64)
        total += 0.5 * float(p.mass) * float(np.dot(v, v))
    return total


def static_parity(system, painn, case_dir: Path, meta: dict, precision: str) -> dict:
    system.integrator.run(0, recalc_forces=True)
    got_energy = float(painn.get_painn_energy())
    got_force = np.asarray([p.f for p in system.part], dtype=np.float64)
    reference = meta["static_reference"][precision]
    ref_energy = float(reference["potential_energy_kj_mol"])
    ref_force = np.load(case_dir / reference["force_file"])
    if got_force.shape != ref_force.shape:
        raise RuntimeError(f"static force shape mismatch: {got_force.shape} != {ref_force.shape}")
    diff = got_force - ref_force
    ref_rms = float(np.sqrt(np.mean(ref_force * ref_force)))
    diff_rms = float(np.sqrt(np.mean(diff * diff)))
    energy_abs = abs(got_energy - ref_energy)
    energy_rel = energy_abs / max(abs(ref_energy), 1.0)
    force_rel_rms = diff_rms / max(ref_rms, 1.0e-30)
    max_force_abs = float(np.max(np.abs(diff)))
    if precision == "float64":
        energy_rel_tol, force_rel_tol = 1.0e-9, 2.0e-8
    else:
        energy_rel_tol, force_rel_tol = 5.0e-5, 2.0e-4
    got_kinetic = kinetic_energy(system)
    ref_kinetic = float(meta["initial_kinetic_energy"]["mlcg_expected_kj_mol"])
    kinetic_rel = abs(got_kinetic - ref_kinetic) / max(abs(ref_kinetic), 1.0)
    kinetic_rel_tol = 1.0e-10
    passed = (
        energy_rel <= energy_rel_tol
        and force_rel_rms <= force_rel_tol
        and kinetic_rel <= kinetic_rel_tol
    )
    return {
        "pass": bool(passed),
        "energy_mlcg_kj_mol": got_energy,
        "energy_torchmd_reference_kj_mol": ref_energy,
        "energy_abs_error_kj_mol": energy_abs,
        "energy_relative_error": energy_rel,
        "force_rms_reference_kj_mol_nm": ref_rms,
        "force_rms_error_kj_mol_nm": diff_rms,
        "force_relative_rms_error": force_rel_rms,
        "force_max_abs_error_kj_mol_nm": max_force_abs,
        "kinetic_energy_mlcg_kj_mol": got_kinetic,
        "kinetic_energy_expected_kj_mol": ref_kinetic,
        "kinetic_energy_relative_error": kinetic_rel,
        "energy_relative_tolerance": energy_rel_tol,
        "kinetic_energy_relative_tolerance": kinetic_rel_tol,
        "force_relative_rms_tolerance": force_rel_tol,
    }


def create_system(espressomd, painn, state: dict, meta: dict, dt_ps: float, precision: str, device: str):
    positions = state["positions_nm"]
    velocities = state["velocities_nm_ps"]
    masses = state["masses_amu"][:, 0]
    species = state["species"]
    max_pos = float(np.max(state["equilibrium_nm"]))
    box_l = max(10.0, max_pos + 5.0)
    system = espressomd.System(box_l=[box_l, box_l, box_l])
    system.time_step = dt_ps
    system.cell_system.skin = 0.4
    system.force_cap = 0.0
    system.integrator.set_vv()
    system.thermostat.turn_off()
    for i in range(len(positions)):
        p = system.part.add(
            pos=positions[i],
            type=int(species[i]),
            mass=float(masses[i]),
            mol_id=i,
        )
        p.v = velocities[i]

    cfg = meta["painn"]
    painn.activate_painn_potential(
        "__MLCG_SYNTHETIC_PAINN_BENCHMARK__",
        int(cfg["num_species"]),
        int(cfg["hidden_channels"]),
        int(cfg["num_layers"]),
        int(cfg["num_rbf"]),
        float(cfg["cutoff_nm"]),
        float(cfg["toxvaerd_alpha"]),
        device=device,
        precision=precision,
    )
    return system


def run_one(espressomd, painn, case_dir, meta, state, dt_ps, duration_ps, precision, device, out_dir):
    exact = duration_ps / dt_ps
    steps = int(round(exact))
    if not math.isclose(steps * dt_ps, duration_ps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"duration {duration_ps:g} ps is not commensurate with dt={dt_ps:g}")
    system = create_system(espressomd, painn, state, meta, dt_ps, precision, device)
    parity = static_parity(system, painn, case_dir, meta, precision)
    if not parity["pass"]:
        raise RuntimeError("static TorchMD/MLCG energy-force parity failed: " + json.dumps(parity, sort_keys=True))

    times = [0.0]
    epot = [float(painn.get_painn_energy())]
    ekin = [kinetic_energy(system)]
    etot = [epot[0] + ekin[0]]
    start = time.perf_counter()
    for step in range(1, steps + 1):
        system.integrator.run(1)
        pe = float(painn.get_painn_energy())
        ke = kinetic_energy(system)
        times.append(step * dt_ps)
        epot.append(pe)
        ekin.append(ke)
        etot.append(pe + ke)
    wall = time.perf_counter() - start
    if not np.isfinite(etot).all():
        raise RuntimeError(f"non-finite total energy at dt={dt_ps:g}")
    metrics = analyze_energy_series(times, etot)
    metrics.update({
        "dt_ps": float(dt_ps),
        "steps": steps,
        "duration_ps": duration_ps,
        "wall_seconds": wall,
        "ms_per_step": 1000.0 * wall / steps,
        "C2_sigma_over_dt2": metrics["sigma_E"] / (dt_ps * dt_ps),
        "static_parity": parity,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "energy.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["Time_ps", "E_pot_kJ_mol", "E_kin_kJ_mol", "E_tot_kJ_mol"])
        w.writerows(zip(times, epot, ekin, etot))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def _run_worker(args: argparse.Namespace, case_dir: Path, out_dir: Path) -> int:
    """Run exactly one dt in a fresh ESPResSo process.

    ESPResSo deliberately permits only one System instance per Python process.
    The parent certification process therefore spawns one worker per dt instead
    of trying to construct six System objects sequentially.
    """
    meta, state = load_case(case_dir)
    os.environ["MLCG_SYNTHETIC_PAINN_CASE_DIR"] = str(case_dir)

    import espressomd
    import espressomd.painn as painn

    metrics = run_one(
        espressomd,
        painn,
        case_dir,
        meta,
        state,
        float(args.worker_dt),
        args.duration_ps,
        args.precision,
        args.device,
        out_dir,
    )
    if not metrics["static_parity"]["pass"]:
        return 2
    return 0


def _worker_command(
    args: argparse.Namespace,
    case_dir: Path,
    out_dir: Path,
    dt_ps: float,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case-dir",
        str(case_dir),
        "--precision",
        args.precision,
        "--device",
        args.device,
        "--duration-ps",
        f"{args.duration_ps:.17g}",
        "--output-dir",
        str(out_dir),
        "--worker-dt",
        f"{dt_ps:.17g}",
    ]


def main() -> int:
    args = parse_args()
    case_dir = (HERE / args.case_dir).resolve() if not Path(args.case_dir).is_absolute() else Path(args.case_dir)
    out_root = (HERE / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if args.dry_run:
        print("[DRY-RUN] MLCG/ESPResSo exact synthetic-PaiNN NVE parity benchmark")
        print(f"[DRY-RUN] case={case_dir}")
        print(f"[DRY-RUN] precision={args.precision} device={args.device} duration={args.duration_ps:g} ps")
        print("[DRY-RUN] dts=" + ",".join(f"{x:g}" for x in args.dts))
        print("[DRY-RUN] execution model=one fresh ESPResSo Python process per dt")
        return 0

    # Internal worker mode: exactly one System is created in this process.
    if args.worker_dt is not None:
        out_root.mkdir(parents=True, exist_ok=True)
        return _run_worker(args, case_dir, out_root)

    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {out_root}; pass --overwrite")
    out_root.mkdir(parents=True, exist_ok=True)
    meta, _ = load_case(case_dir)

    print("[MLCG/ESPRESSO SYNTHETIC PAINN NVE CERTIFICATION]")
    print(f"device / precision : {args.device} / {args.precision}")
    print(f"particles          : {meta['particles']}")
    print(f"edges              : {meta['graph']['directed_edges']} fixed directed")
    print(f"duration           : {args.duration_ps:g} ps per dt")
    print("execution          : one fresh ESPResSo process per dt")
    runs = []
    for dt in args.dts:
        print(f"[RUN] dt={dt:g} ps", flush=True)
        label = str(dt).replace(".", "p")
        dt_dir = out_root / f"dt_{label}"
        completed = subprocess.run(
            _worker_command(args, case_dir, dt_dir, dt),
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
        metrics_path = dt_dir / "metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"worker completed without metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append(metrics)
        p = metrics["static_parity"]
        print(
            f"      sigma_E={metrics['sigma_E']:.8g} kJ/mol  C2={metrics['C2_sigma_over_dt2']:.8g}  "
            f"drift={metrics['relative_block_mean_drift']:.3e}  {metrics['ms_per_step']:.3f} ms/step"
        )
        print(
            f"      static parity: dErel={p['energy_relative_error']:.3e}  "
            f"dFrmsrel={p['force_relative_rms_error']:.3e}  "
            f"dKrel={p['kinetic_energy_relative_error']:.3e}"
        )

    cert = certify_metrics(
        runs,
        slope_min=args.slope_min,
        slope_max=args.slope_max,
        min_r2=args.min_r2,
        max_relative_drift=args.max_relative_drift,
    )
    report = {
        "kind": "mlcg_espresso_exact_synthetic_painn_nve_certification",
        "scope": "Exact shared Hamiltonian with TorchMD synthetic PaiNN: fixed graph, identical canonical weights, harmonic background and initial state",
        "execution_model": "one fresh ESPResSo Python process per dt (ESPResSo System singleton safe)",
        "device": args.device,
        "precision": args.precision,
        "duration_ps": args.duration_ps,
        "dts_ps": list(args.dts),
        "particles": meta["particles"],
        "graph": meta["graph"],
        "painn": meta["painn"],
        "residual_calibration": meta["residual_calibration"],
        "runs": runs,
        "certification": cert,
    }
    report_path = out_root / "mlcg_painn_nve_certification_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    scaling = cert["scaling"]
    print("[RESULT]")
    print(f"  p              = {scaling['exponent_p']:.6f}")
    print(f"  log-log R2     = {scaling['loglog_r2']:.6f}")
    print(f"  C2 spread      = {cert['c2_spread_max_over_min']:.3f}")
    print(f"  max drift      = {max(r['relative_block_mean_drift'] for r in runs):.3e}")
    print(f"  certification  = {'PASS' if cert['pass'] else 'FAIL'}")
    print(f"  report         = {report_path}")
    return 0 if cert["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
