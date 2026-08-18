#!/usr/bin/env python3
"""Export the TorchMD synthetic-PaiNN case in MLCG/ESPResSo units.

The exported case is the bridge used by the ESPResSo/LibTorch benchmark.  It
contains the exact fixed directed graph, canonical float64 PaiNN parameters,
initial state, harmonic background, residual scale, and static FP32/FP64
reference energies/forces from the existing TorchMD-side implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

from run_certification import build_initial_state  # noqa: E402
from run_painn_certification import canonical_calibration  # noqa: E402
from synthetic_painn import (  # noqa: E402
    SyntheticPaiNNConfig,
    build_synthetic_painn,
    fixed_knn_edge_index,
    model_fingerprint,
    species_pattern,
)

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1
# TorchMD's TIMEFACTOR is the rounded MD-unit time conversion used by its
# integrator.  For the shared physical initial state, convert velocity by the
# exact energy-unit relation so 1/2 m v^2 maps exactly from kcal/mol to kJ/mol.
# 100/TIMEFACTOR differs from sqrt(4.184) by only ~6e-8 relative.
TORCHMD_TIMEFACTOR = 48.88821
TORCH_VELOCITY_TO_NM_PS = math.sqrt(KCAL_TO_KJ)
TORCHMD_NUMERICAL_VELOCITY_TO_NM_PS = 100.0 / TORCHMD_TIMEFACTOR
FORCE_KCAL_A_TO_KJ_NM = KCAL_TO_KJ / ANGSTROM_TO_NM
K_KCAL_A2_TO_KJ_NM2 = KCAL_TO_KJ / (ANGSTROM_TO_NM * ANGSTROM_TO_NM)
ESPRESSO_BOUNDARY_MARGIN_NM = 1.0


def translate_state_for_espresso(
    equilibrium_nm: np.ndarray,
    positions_nm: np.ndarray,
    margin_nm: float = ESPRESSO_BOUNDARY_MARGIN_NM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly translate the shared state away from ESPResSo periodic boundaries.

    TorchMD permits the synthetic coordinates to be slightly negative. ESPResSo
    folds such coordinates into the periodic box, which would make the harmonic
    displacement appear box-sized although the physical state is unchanged. A
    common rigid translation preserves every pair displacement, harmonic delta,
    PaiNN energy, and force.
    """
    equilibrium_nm = np.asarray(equilibrium_nm, dtype=np.float64)
    positions_nm = np.asarray(positions_nm, dtype=np.float64)
    if equilibrium_nm.shape != positions_nm.shape or equilibrium_nm.ndim != 2 or equilibrium_nm.shape[1] != 3:
        raise ValueError("equilibrium_nm and positions_nm must both have shape [N,3]")
    if not np.isfinite(equilibrium_nm).all() or not np.isfinite(positions_nm).all():
        raise ValueError("non-finite coordinates in shared synthetic state")
    if not (margin_nm > 0.0):
        raise ValueError("margin_nm must be positive")
    component_min = np.minimum(equilibrium_nm.min(axis=0), positions_nm.min(axis=0))
    translation_nm = np.maximum(float(margin_nm) - component_min, 0.0)
    return (
        equilibrium_nm + translation_nm,
        positions_nm + translation_nm,
        translation_nm,
    )


class InitialArgs:
    particles = 820
    seed = 220526
    displacement_sigma_a = 0.20
    mass_g_mol = 72.0
    temperature_k = 300.0
    k_min = 20.0
    k_max = 80.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results/shared_painn_case")
    p.add_argument("--particles", type=int, default=820)
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--num-rbf", type=int, default=32)
    p.add_argument("--num-species", type=int, default=8)
    p.add_argument("--neighbors", type=int, default=24)
    p.add_argument("--cutoff-a", type=float, default=12.616)
    p.add_argument("--toxvaerd-alpha", type=float, default=0.1)
    p.add_argument("--residual-force-fraction", type=float, default=0.50)
    p.add_argument("--initial-seed", type=int, default=220526)
    p.add_argument("--network-seed", type=int, default=220528)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def cpp_parameter_name(py_name: str) -> str:
    parts = py_name.split(".")
    if parts[0] == "messages":
        return ".".join([f"message_{parts[1]}"] + parts[2:])
    if parts[0] == "updates":
        return ".".join([f"update_{parts[1]}"] + parts[2:])
    return py_name


def write_weights(path: Path, model) -> list[str]:
    rows = []
    names = []
    for py_name, tensor in model.state_dict().items():
        if py_name in {"edge_index", "species"}:
            continue
        name = cpp_parameter_name(py_name)
        values = tensor.detach().cpu().to(dtype=model.embedding.weight.dtype).contiguous().view(-1).numpy()
        names.append(name)
        rows.append((name, values))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("MLCG_SYNTHETIC_PAINN_WEIGHTS_V1\n")
        handle.write(f"{len(rows)}\n")
        for name, values in rows:
            handle.write(f"{name} {values.size}\n")
            # 17 significant digits preserve canonical float64 parameters.
            for start in range(0, values.size, 8):
                handle.write(" ".join(f"{float(x):.17g}" for x in values[start:start + 8]) + "\n")
        handle.write("END\n")
    return names


def static_reference(torch, model, initial, residual_scale: float, dtype) -> tuple[float, np.ndarray]:
    pos = torch.as_tensor(initial["positions"], dtype=dtype).unsqueeze(0).clone().requires_grad_(True)
    eq = torch.as_tensor(initial["equilibrium"], dtype=dtype).unsqueeze(0)
    k = torch.as_tensor(initial["stiffness"], dtype=dtype).unsqueeze(0)
    harmonic = 0.5 * torch.sum(k * (pos - eq) * (pos - eq), dim=(1, 2))
    residual = model(pos) * residual_scale
    energy = harmonic + residual
    force = -torch.autograd.grad(energy.sum(), pos, create_graph=False, retain_graph=False)[0]
    energy_kj = float(energy.detach().cpu().to(torch.float64).item() * KCAL_TO_KJ)
    force_kj_nm = force.detach().cpu().to(torch.float64).numpy()[0] * FORCE_KCAL_A_TO_KJ_NM
    return energy_kj, force_kj_nm


def main() -> int:
    args = parse_args()
    if args.neighbors >= args.particles:
        raise ValueError("--neighbors must be smaller than --particles")

    out = Path(args.output_dir).expanduser().resolve()
    if args.dry_run:
        print("[DRY-RUN] export shared TorchMD/MLCG synthetic PaiNN case")
        print(f"[DRY-RUN] output={out}")
        print(
            f"[DRY-RUN] N={args.particles}, PaiNN={args.layers}x{args.hidden_channels}, "
            f"RBF={args.num_rbf}, kNN={args.neighbors}, species={args.num_species}"
        )
        return 0

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {out}; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to export the shared case") from exc

    ia = InitialArgs()
    ia.particles = args.particles
    ia.seed = args.initial_seed
    initial = build_initial_state(ia)
    config = SyntheticPaiNNConfig(
        hidden_channels=args.hidden_channels,
        num_layers=args.layers,
        num_rbf=args.num_rbf,
        num_species=args.num_species,
        neighbors_per_particle=args.neighbors,
        cutoff_a=args.cutoff_a,
        toxvaerd_alpha=args.toxvaerd_alpha,
        seed=args.network_seed,
        residual_force_fraction=args.residual_force_fraction,
    )
    edge_np = fixed_knn_edge_index(initial["equilibrium"], config.neighbors_per_particle)
    species_np = species_pattern(args.particles, config.num_species)

    calibration, fingerprint, parameter_count = canonical_calibration(
        torch, ia, initial, config, edge_np, species_np
    )
    cpu = torch.device("cpu")
    canonical = build_synthetic_painn(
        torch,
        config,
        edge_index=torch.as_tensor(edge_np, dtype=torch.long),
        species=torch.as_tensor(species_np, dtype=torch.long),
        dtype=torch.float64,
        device=cpu,
    )
    if model_fingerprint(canonical) != fingerprint:
        raise RuntimeError("canonical PaiNN fingerprint changed during export")

    weight_names = write_weights(out / "weights.txt", canonical)
    np.savetxt(out / "graph.txt", edge_np.T, fmt="%d", header=str(edge_np.shape[1]), comments="")

    eq_nm_raw = initial["equilibrium"] * ANGSTROM_TO_NM
    pos_nm_raw = initial["positions"] * ANGSTROM_TO_NM
    eq_nm, pos_nm, espresso_translation_nm = translate_state_for_espresso(
        eq_nm_raw, pos_nm_raw
    )
    vel_nm_ps = initial["velocities"] * TORCH_VELOCITY_TO_NM_PS
    stiffness_kj_nm2 = initial["stiffness"] * K_KCAL_A2_TO_KJ_NM2
    np.savez_compressed(
        out / "state_mlcg_units.npz",
        equilibrium_nm=eq_nm,
        positions_nm=pos_nm,
        velocities_nm_ps=vel_nm_ps,
        masses_amu=initial["masses"],
        stiffness_kj_mol_nm2=stiffness_kj_nm2,
        species=species_np,
        edge_index=edge_np,
    )
    # Preserve the exact numerical units used by the TorchMD synthetic benchmark.
    # The ESPResSo benchmark branch evaluates the Hamiltonian in Angstrom and
    # kcal/mol inside LibTorch, then converts only the final energy/forces at the
    # engine boundary. This prevents FP32 from receiving a hidden double-precision
    # harmonic background.
    with (out / "harmonic_torchmd_units.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{args.particles}\n")
        for xyz, kval in zip(initial["equilibrium"], initial["stiffness"][:, 0]):
            handle.write(f"{xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} {kval:.17g}\n")

    initial_kinetic_kcal = 0.5 * float(np.sum(initial["masses"] * initial["velocities"] * initial["velocities"]))
    initial_kinetic_kj = initial_kinetic_kcal * KCAL_TO_KJ

    references = {}
    for precision, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        model = build_synthetic_painn(
            torch,
            config,
            edge_index=torch.as_tensor(edge_np, dtype=torch.long),
            species=torch.as_tensor(species_np, dtype=torch.long),
            dtype=dtype,
            device=cpu,
        )
        e_kj, f_kj_nm = static_reference(
            torch, model, initial, calibration["energy_scale"], dtype
        )
        np.save(out / f"static_force_{precision}_kj_mol_nm.npy", f_kj_nm)
        references[precision] = {
            "potential_energy_kj_mol": e_kj,
            "force_file": f"static_force_{precision}_kj_mol_nm.npy",
        }

    energy_scale_kj = calibration["energy_scale"] * KCAL_TO_KJ
    config_txt = out / "config.txt"
    config_txt.write_text(
        "MLCG_SYNTHETIC_PAINN_CASE_V3\n"
        f"particles {args.particles}\n"
        f"num_species {config.num_species}\n"
        f"hidden_channels {config.hidden_channels}\n"
        f"num_layers {config.num_layers}\n"
        f"num_rbf {config.num_rbf}\n"
        f"cutoff_nm {config.cutoff_a * ANGSTROM_TO_NM:.17g}\n"
        f"cutoff_A {config.cutoff_a:.17g}\n"
        f"toxvaerd_alpha {config.toxvaerd_alpha:.17g}\n"
        f"energy_scale_kcal {calibration['energy_scale']:.17g}\n"
        f"espresso_translation_A {espresso_translation_nm[0] / ANGSTROM_TO_NM:.17g} "
        f"{espresso_translation_nm[1] / ANGSTROM_TO_NM:.17g} "
        f"{espresso_translation_nm[2] / ANGSTROM_TO_NM:.17g}\n",
        encoding="utf-8",
    )

    metadata = {
        "kind": "shared_torchmd_mlcg_synthetic_painn_case",
        "particles": args.particles,
        "graph": {
            "policy": "fixed directed kNN from equilibrium coordinates",
            "directed_edges": int(edge_np.shape[1]),
            "neighbors_per_particle": config.neighbors_per_particle,
        },
        "painn": {
            "hidden_channels": config.hidden_channels,
            "num_layers": config.num_layers,
            "num_rbf": config.num_rbf,
            "num_species": config.num_species,
            "cutoff_A": config.cutoff_a,
            "cutoff_nm": config.cutoff_a * ANGSTROM_TO_NM,
            "toxvaerd_alpha": config.toxvaerd_alpha,
            "network_seed": config.seed,
            "parameter_count": parameter_count,
            "parameter_sha256_canonical_float64": fingerprint,
            "exported_cpp_parameter_names": weight_names,
        },
        "residual_calibration": calibration,
        "energy_scale_kcal_per_raw_model_unit": calibration["energy_scale"],
        "energy_scale_kj_per_raw_model_unit": energy_scale_kj,
        "espresso_coordinate_translation": {
            "translation_nm": espresso_translation_nm.tolist(),
            "minimum_boundary_margin_nm": ESPRESSO_BOUNDARY_MARGIN_NM,
            "reason": "avoid ESPResSo periodic folding of slightly negative TorchMD synthetic coordinates",
            "physics_invariant": "rigid translation preserves harmonic deltas and all fixed-graph PaiNN pair displacements",
        },
        "benchmark_internal_units": {
            "length": "angstrom",
            "energy": "kcal/mol",
            "force": "kcal/mol/angstrom",
            "note": "MLCG benchmark evaluates the full synthetic Hamiltonian in TorchMD numerical units before boundary conversion",
        },
        "unit_conversion": {
            "kcal_to_kj": KCAL_TO_KJ,
            "angstrom_to_nm": ANGSTROM_TO_NM,
            "torchmd_timefactor": TORCHMD_TIMEFACTOR,
            "physical_velocity_to_nm_ps": TORCH_VELOCITY_TO_NM_PS,
            "torchmd_numerical_velocity_to_nm_ps": TORCHMD_NUMERICAL_VELOCITY_TO_NM_PS,
            "relative_velocity_conversion_difference": TORCHMD_NUMERICAL_VELOCITY_TO_NM_PS / TORCH_VELOCITY_TO_NM_PS - 1.0,
            "force_kcal_per_A_to_kj_per_nm": FORCE_KCAL_A_TO_KJ_NM,
            "harmonic_k_kcal_per_A2_to_kj_per_nm2": K_KCAL_A2_TO_KJ_NM2,
        },
        "static_reference": references,
        "initial_kinetic_energy": {
            "torchmd_kcal_mol": initial_kinetic_kcal,
            "mlcg_expected_kj_mol": initial_kinetic_kj,
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("[PASS] exported shared synthetic PaiNN case")
    print(f"case       : {out}")
    print(f"edges      : {edge_np.shape[1]}")
    print(f"fingerprint: {fingerprint}")
    print(f"scale kcal : {calibration['energy_scale']:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
