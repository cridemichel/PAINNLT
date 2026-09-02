#!/usr/bin/env python3
"""Train and simulate the pinned official dense CGnet Ala2 tutorial model."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


TRAIN_FRAMES = 8000
VALIDATION_FRAMES = 2000
N_LAYERS = 5
N_NODES = 160
BATCH_SIZE = 512
LEARNING_RATE = 0.003
RATE_DECAY = 0.3
LIPSCHITZ_STRENGTH = 4.0
SCHEDULER_MILESTONES = [1, 2, 3, 4, 5]


def select_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CGNET_DEVICE=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("CGNET_DEVICE=mps requested but MPS is unavailable")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"Unknown device: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def matched_frame_count(paths: list[Path]) -> int:
    counts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "complete" not in data.files or int(np.asarray(data["complete"]).item()) != 1:
                raise ValueError(f"Incomplete matched runtime sample: {path}")
            counts.append(int(np.asarray(data["sites"]).shape[0]))
    if len(set(counts)) != 1 or counts[0] < 1:
        raise ValueError(f"Matched trajectories have inconsistent frame counts: {counts}")
    return counts[0]


def state_dict_on_cpu(model) -> dict[str, object]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def evaluate(model, loader, torch) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    elements = 0
    for coordinates, forces, _ in loader:
        _, prediction = model(coordinates)
        difference = prediction - forces
        squared_error += float(torch.sum(difference.detach() ** 2).cpu())
        absolute_error += float(torch.sum(torch.abs(difference.detach())).cpu())
        elements += difference.numel()
    return {"mse": squared_error / elements, "mae": absolute_error / elements}


def prior_only_metrics(model, loader, torch) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    zero_squared_error = 0.0
    zero_absolute_error = 0.0
    elements = 0
    for coordinates, forces, _ in loader:
        geometry = model.feature(coordinates)
        energy = torch.zeros((coordinates.shape[0], 1), device=coordinates.device)
        for prior in model.priors:
            energy = energy + prior(geometry[:, prior.callback_indices])
        prediction = torch.autograd.grad(-torch.sum(energy), coordinates)[0]
        difference = prediction - forces
        squared_error += float(torch.sum(difference.detach() ** 2).cpu())
        absolute_error += float(torch.sum(torch.abs(difference.detach())).cpu())
        zero_squared_error += float(torch.sum(forces.detach() ** 2).cpu())
        zero_absolute_error += float(torch.sum(torch.abs(forces.detach())).cpu())
        elements += difference.numel()
    return {
        "mse": squared_error / elements,
        "mae": absolute_error / elements,
        "raw_zero_predictor_mse": zero_squared_error / elements,
        "raw_zero_predictor_mae": zero_absolute_error / elements,
    }


def simulate_branch(
    model,
    initial_coordinates: np.ndarray,
    total_steps: int,
    sample_interval: int,
    beta: float,
    dt: float,
    random_seed: int,
    torch,
    Simulation,
) -> tuple[np.ndarray, float]:
    """Run one official Brownian branch with an independently reset RNG."""
    device = torch.device("cpu")
    model.mount(device)
    model.eval()
    initial = torch.tensor(
        initial_coordinates, dtype=torch.float32, requires_grad=True
    )
    log_interval = max(
        sample_interval,
        (max(total_steps // 10, sample_interval) // sample_interval) * sample_interval,
    )
    start = time.perf_counter()
    simulation = Simulation(
        model,
        initial,
        length=total_steps,
        save_interval=sample_interval,
        beta=beta,
        dt=dt,
        random_seed=random_seed,
        device=device,
        log_interval=log_interval,
        log_type="print",
    )
    coordinates = np.asarray(simulation.simulate(), dtype=np.float32)
    return coordinates, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cgnet-source", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--forces", required=True, type=Path)
    parser.add_argument("--matched-runtime-samples", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--equil-steps", type=int, default=25000)
    parser.add_argument("--burnin-steps", type=int, default=10000)
    parser.add_argument("--sample-interval", type=int, default=50)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    args = parser.parse_args()

    if args.epochs < 1 or args.equil_steps < 0 or args.burnin_steps < 0:
        raise ValueError("Epoch/equilibration/burn-in values must be non-negative")
    if args.sample_interval < 1 or args.dt <= 0.0:
        raise ValueError("Sample interval and time step must be positive")
    if (args.equil_steps + args.burnin_steps) % args.sample_interval != 0:
        raise ValueError("equil-steps + burnin-steps must be divisible by sample-interval")

    source = args.cgnet_source.resolve()
    if not (source / "cgnet" / "__init__.py").is_file():
        raise FileNotFoundError(f"Invalid official CGnet source root: {source}")
    sys.path.insert(0, str(source))

    # The pinned source predates NumPy 1.24. This alias restores its old public
    # name without changing a byte of the downloaded reference implementation.
    if "bool" not in np.__dict__:
        setattr(np, "bool", np.bool_)

    try:
        import torch
        import torch.nn as nn
        from torch.optim.lr_scheduler import MultiStepLR
        from torch.utils.data import DataLoader
        from cgnet.feature import GeometryFeature, GeometryStatistics, LinearLayer, MoleculeDataset
        from cgnet.network import (
            CGnet,
            ForceLoss,
            HarmonicLayer,
            Simulation,
            ZscoreLayer,
            lipschitz_projection,
        )
    except ImportError as error:
        raise RuntimeError(
            "Official CGnet requires torch, scipy and numpy in the active Python environment"
        ) from error

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch, args.device)
    print(f"[INFO] Official CGnet training device: {device}")

    coordinates = np.load(args.coordinates, allow_pickle=False)
    forces = np.load(args.forces, allow_pickle=False)
    expected_shape = (TRAIN_FRAMES + VALIDATION_FRAMES, 5, 3)
    if coordinates.shape != expected_shape or forces.shape != expected_shape:
        raise ValueError(
            f"Expected coordinate/force arrays with shape {expected_shape}; "
            f"got {coordinates.shape} and {forces.shape}"
        )
    if not np.isfinite(coordinates).all() or not np.isfinite(forces).all():
        raise ValueError("Non-finite values in official Ala2 arrays")
    coordinates = np.asarray(coordinates, dtype=np.float32)
    forces = np.asarray(forces, dtype=np.float32)
    train_coordinates = coordinates[:TRAIN_FRAMES]
    train_forces = forces[:TRAIN_FRAMES]
    validation_coordinates = coordinates[TRAIN_FRAMES:]
    validation_forces = forces[TRAIN_FRAMES:]

    stats = GeometryStatistics(
        train_coordinates,
        backbone_inds="all",
        get_all_distances=True,
        get_backbone_angles=True,
        get_backbone_dihedrals=True,
        temperature=300.0,
    )
    all_stats, _ = stats.get_prior_statistics(as_list=True)
    zscores, _ = stats.get_zscore_array()
    bond_parameters, _ = stats.get_prior_statistics(features="Bonds", as_list=True)
    angle_parameters, _ = stats.get_prior_statistics(features="Angles", as_list=True)
    bond_indices = stats.return_indices("Bonds")
    angle_indices = stats.return_indices("Angles")

    layers = [ZscoreLayer(zscores)]
    layers += LinearLayer(len(all_stats), N_NODES, activation=nn.Tanh())
    for _ in range(N_LAYERS - 1):
        layers += LinearLayer(N_NODES, N_NODES, activation=nn.Tanh())
    layers += LinearLayer(N_NODES, 1, activation=None)
    priors = [
        HarmonicLayer(bond_indices, bond_parameters),
        HarmonicLayer(angle_indices, angle_parameters),
    ]
    feature = GeometryFeature(feature_tuples=stats.feature_tuples, device=device)
    model = CGnet(layers, ForceLoss(), feature=feature, priors=priors)
    model.mount(device)

    train_dataset = MoleculeDataset(train_coordinates, train_forces, device=device)
    validation_dataset = MoleculeDataset(
        validation_coordinates, validation_forces, device=device
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(
        optimizer, milestones=SCHEDULER_MILESTONES, gamma=RATE_DECAY
    )
    history = []
    best_validation_mse = float("inf")
    best_epoch = 0
    best_state = None
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_error = 0.0
        absolute_error = 0.0
        elements = 0
        current_lr = float(optimizer.param_groups[0]["lr"])
        for coordinates_batch, force_batch, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            _, prediction = model(coordinates_batch)
            difference = prediction - force_batch
            loss = torch.mean(difference * difference)
            loss.backward()
            optimizer.step()
            lipschitz_projection(model, strength=LIPSCHITZ_STRENGTH)
            squared_error += float(torch.sum(difference.detach() ** 2).cpu())
            absolute_error += float(torch.sum(torch.abs(difference.detach())).cpu())
            elements += difference.numel()
        validation = evaluate(model, validation_loader, torch)
        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_mse": squared_error / elements,
            "train_mae": absolute_error / elements,
            "validation_mse": validation["mse"],
            "validation_mae": validation["mae"],
        }
        history.append(row)
        print(
            f"[CGnet] Epoch {epoch}/{args.epochs} | lr={current_lr:.6g} | "
            f"train MSE={row['train_mse']:.6g} | val MSE={row['validation_mse']:.6g} | "
            f"val MAE={row['validation_mae']:.6g}"
        )
        if validation["mse"] < best_validation_mse:
            best_validation_mse = validation["mse"]
            best_epoch = epoch
            best_state = state_dict_on_cpu(model)
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Official CGnet training produced no checkpoint")
    training_seconds = time.perf_counter() - training_start
    model.load_state_dict(best_state)
    final_train = evaluate(model, train_loader, torch)
    final_validation = evaluate(model, validation_loader, torch)
    validation_prior = prior_only_metrics(model, validation_loader, torch)
    explained_residual_variance = 1.0 - final_validation["mse"] / validation_prior["mse"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "official_cgnet_state_dict.pt")
    with (args.output_dir / "official_cgnet_training_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    target_frames = matched_frame_count(args.matched_runtime_samples)
    replicas = len(args.matched_runtime_samples)
    initial_indices = np.linspace(0, len(coordinates) - 1, replicas, dtype=int)
    discard_steps = args.equil_steps + args.burnin_steps
    total_steps = discard_steps + target_frames * args.sample_interval
    print(
        f"[INFO] Official overdamped CGnet simulation: {replicas} replicas x "
        f"{total_steps} steps; retaining {target_frames} frames/replica."
    )

    simulation_seed = args.seed + 1
    initial_coordinates = coordinates[initial_indices]
    full_arch_state = state_dict_on_cpu(model.arch)
    with torch.no_grad():
        for parameter in model.arch.parameters():
            parameter.zero_()

    print("[INFO] Brownian branch 1/2: harmonic prior only.")
    prior_simulated, prior_simulation_seconds = simulate_branch(
        model,
        initial_coordinates,
        total_steps,
        args.sample_interval,
        float(stats.beta),
        args.dt,
        simulation_seed,
        torch,
        Simulation,
    )
    model.arch.load_state_dict(full_arch_state)
    print("[INFO] Brownian branch 2/2: harmonic prior + official CGnet.")
    simulated, full_simulation_seconds = simulate_branch(
        model,
        initial_coordinates,
        total_steps,
        args.sample_interval,
        float(stats.beta),
        args.dt,
        simulation_seed,
        torch,
        Simulation,
    )
    discard_frames = discard_steps // args.sample_interval
    prior_retained = prior_simulated[
        :, discard_frames : discard_frames + target_frames
    ]
    retained = simulated[:, discard_frames : discard_frames + target_frames]
    if prior_retained.shape != (replicas, target_frames, 5, 3):
        raise RuntimeError(
            f"Unexpected prior-only CGnet trajectory shape: {prior_retained.shape}"
        )
    if retained.shape != (replicas, target_frames, 5, 3):
        raise RuntimeError(f"Unexpected official CGnet trajectory shape: {retained.shape}")
    if not np.isfinite(prior_retained).all() or not np.isfinite(retained).all():
        raise RuntimeError("A Brownian comparison branch generated non-finite coordinates")

    prior_sample_paths = []
    sample_paths = []
    for replica, (prior_sample, sample) in enumerate(zip(prior_retained, retained)):
        prior_path = args.output_dir / f"official_cgnet_prior_replica_{replica:02d}.npy"
        path = args.output_dir / f"official_cgnet_replica_{replica:02d}.npy"
        np.save(prior_path, prior_sample)
        np.save(path, sample)
        prior_sample_paths.append(str(prior_path.resolve()))
        sample_paths.append(str(path.resolve()))
    np.save(
        args.output_dir / "official_cgnet_prior_samples.npy",
        prior_retained.reshape(-1, 5, 3),
    )
    np.save(args.output_dir / "official_cgnet_samples.npy", retained.reshape(-1, 5, 3))

    package_versions = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unknown"
    report = {
        "schema_version": 1,
        "status": "pass",
        "implementation": {
            "name": "official dense CGnet Ala2 tutorial",
            "source_root": str(source),
            "source_files_modified": False,
            "numpy_compatibility_alias": "np.bool = np.bool_",
            "architecture": {
                "invariant_features": len(all_stats),
                "hidden_layers": N_LAYERS,
                "hidden_nodes": N_NODES,
                "activation": "Tanh",
                "harmonic_bonds": len(bond_indices),
                "harmonic_angles": len(angle_indices),
                "excluded_volume_prior": False,
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
            },
        },
        "controlled_protocol": {
            "training_frames": TRAIN_FRAMES,
            "validation_frames": VALIDATION_FRAMES,
            "split": "frames 0:7999 train; 8000:9999 validation",
            "statistics_and_priors_fit_on": "training frames only",
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "scheduler_milestones": SCHEDULER_MILESTONES,
            "scheduler_gamma": RATE_DECAY,
            "lipschitz_strength": LIPSCHITZ_STRENGTH,
            "seed": args.seed,
            "training_device": str(device),
            "simulation_device": "cpu",
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_mse_kcal2_mol2_angstrom2": best_validation_mse,
            "best_checkpoint_train": final_train,
            "best_checkpoint_validation": final_validation,
            "validation_prior_only": validation_prior,
            "explained_validation_residual_force_variance": explained_residual_variance,
            "training_seconds": training_seconds,
            "history": history,
        },
        "simulation": {
            "dynamics": "official CGnet overdamped Langevin (Brownian)",
            "beta_mol_per_kcal": float(stats.beta),
            "dt": args.dt,
            "equilibration_steps": args.equil_steps,
            "burnin_steps": args.burnin_steps,
            "sample_interval": args.sample_interval,
            "total_steps": total_steps,
            "replicas": replicas,
            "frames_per_replica": target_frames,
            "initial_reference_frame_indices": initial_indices.tolist(),
            "sample_units": "angstrom",
            "matched_branches": {
                "prior_only": {
                    "network_energy_disabled_by_zeroing_arch_parameters": True,
                    "sample_paths": prior_sample_paths,
                    "simulation_seconds": prior_simulation_seconds,
                },
                "prior_plus_cgnet": {
                    "sample_paths": sample_paths,
                    "simulation_seconds": full_simulation_seconds,
                },
                "same_initial_coordinates": True,
                "same_brownian_random_seed": simulation_seed,
                "same_brownian_noise_sequence": True,
            },
            "prior_only_sample_paths": prior_sample_paths,
            "sample_paths": sample_paths,
            "simulation_seconds": prior_simulation_seconds + full_simulation_seconds,
        },
        "versions": package_versions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "[PASS] Official CGnet comparator complete; explained validation residual variance: "
        f"{100.0 * explained_residual_variance:.3f}%"
    )


if __name__ == "__main__":
    main()
