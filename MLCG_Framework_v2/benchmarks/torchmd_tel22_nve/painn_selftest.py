#!/usr/bin/env python3
"""Source-level and numerical sanity checks for the synthetic PaiNN NVE diagnostic."""
from __future__ import annotations

import math
import sys

import numpy as np

from synthetic_painn import (
    AutogradPaiNNForces,
    SyntheticPaiNNConfig,
    build_synthetic_painn,
    fixed_knn_edge_index,
    model_fingerprint,
    species_pattern,
)


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"[ERROR] PyTorch unavailable: {exc}", file=sys.stderr)
        return 2

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    rng = np.random.default_rng(20260818)
    n = 24
    equilibrium = rng.normal(size=(n, 3)) * 2.0
    positions = equilibrium + rng.normal(scale=0.05, size=(n, 3))
    edge_np = fixed_knn_edge_index(equilibrium, 6)
    assert edge_np.shape == (2, n * 6)
    assert not np.any(edge_np[0] == edge_np[1])
    assert np.array_equal(edge_np, fixed_knn_edge_index(equilibrium, 6))

    config = SyntheticPaiNNConfig(
        hidden_channels=8,
        num_layers=2,
        num_rbf=8,
        num_species=4,
        neighbors_per_particle=6,
        cutoff_a=8.0,
        seed=99,
        residual_force_fraction=0.5,
    )
    edge = torch.as_tensor(edge_np, dtype=torch.long)
    species = torch.as_tensor(species_pattern(n, config.num_species), dtype=torch.long)
    model_a = build_synthetic_painn(
        torch, config, edge_index=edge, species=species, dtype=torch.float64, device=torch.device("cpu")
    )
    model_b = build_synthetic_painn(
        torch, config, edge_index=edge, species=species, dtype=torch.float64, device=torch.device("cpu")
    )
    assert model_fingerprint(model_a) == model_fingerprint(model_b)
    assert all(not p.requires_grad for p in model_a.parameters())

    x = torch.tensor(positions, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    e = model_a(x)
    force = -torch.autograd.grad(e.sum(), x)[0]
    # The PaiNN residual depends only on relative coordinates, so net force must vanish.
    assert float(torch.linalg.vector_norm(force.sum(dim=1)).item()) < 2.0e-10

    # Rotation invariance of the scalar energy.
    theta = 0.37
    rot = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    e_rot = model_a(x.detach() @ rot.T)
    assert abs(float((e.detach() - e_rot.detach()).item())) < 2.0e-10

    # One-coordinate finite difference against autograd.
    atom, dim = 3, 1
    h = 2.0e-6
    xp = x.detach().clone(); xp[0, atom, dim] += h
    xm = x.detach().clone(); xm[0, atom, dim] -= h
    fd_force = -float((model_a(xp) - model_a(xm)).item()) / (2.0 * h)
    assert math.isclose(fd_force, float(force[0, atom, dim]), rel_tol=2.0e-5, abs_tol=2.0e-7)

    # Force-provider smoke with harmonic background.
    eq = torch.tensor(equilibrium, dtype=torch.float64).unsqueeze(0)
    stiffness = torch.full((1, n, 1), 20.0, dtype=torch.float64)
    provider = AutogradPaiNNForces(torch, model_a, eq, stiffness, residual_scale=1.0)
    forces = torch.zeros_like(x.detach())
    energy = provider.compute(x.detach(), torch.zeros((1, 3, 3), dtype=torch.float64), forces)
    assert np.isfinite(energy).all() and torch.isfinite(forces).all()
    assert provider.compute_calls == 1

    print("[PASS] synthetic PaiNN self-test: graph, determinism, invariance, finite-difference forces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
