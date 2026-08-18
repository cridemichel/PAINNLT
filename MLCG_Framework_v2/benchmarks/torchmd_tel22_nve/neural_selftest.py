#!/usr/bin/env python3
"""Fast checks for the deterministic neural potential; TorchMD itself is not required."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from neural_potential import (  # noqa: E402
    AutogradNeuralForces,
    NeuralPotentialConfig,
    build_neural_model,
    model_fingerprint,
)


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"[ERROR] PyTorch unavailable: {exc}", file=sys.stderr)
        return 2

    config = NeuralPotentialConfig(hidden_channels=16, hidden_layers=2, seed=1234)
    model_a = build_neural_model(torch, config, dtype=torch.float64, device=torch.device("cpu"))
    model_b = build_neural_model(torch, config, dtype=torch.float64, device=torch.device("cpu"))
    assert model_fingerprint(model_a) == model_fingerprint(model_b)
    assert all(not p.requires_grad for p in model_a.parameters())

    equilibrium = torch.zeros((1, 4, 3), dtype=torch.float64)
    stiffness = torch.tensor([[[20.0], [30.0], [40.0], [50.0]]], dtype=torch.float64)
    pos = torch.tensor(
        [[[0.15, -0.05, 0.02], [0.04, 0.11, -0.03], [-0.08, 0.07, 0.10], [0.02, -0.09, 0.06]]],
        dtype=torch.float64,
    )
    forces = torch.zeros_like(pos)
    provider = AutogradNeuralForces(torch, model_a, equilibrium, stiffness)
    energy = float(provider.compute(pos, torch.zeros((1, 3, 3)), forces)[0])
    assert torch.isfinite(forces).all() and energy > 0.0

    atom, axis = 1, 2
    h = 1.0e-6
    plus = pos.clone()
    minus = pos.clone()
    plus[0, atom, axis] += h
    minus[0, atom, axis] -= h
    with torch.no_grad():
        e_plus = float(model_a(plus - equilibrium, stiffness)[0])
        e_minus = float(model_a(minus - equilibrium, stiffness)[0])
    fd_force = -(e_plus - e_minus) / (2.0 * h)
    autograd_force = float(forces[0, atom, axis])
    assert abs(fd_force - autograd_force) <= 2.0e-7 * max(1.0, abs(fd_force), abs(autograd_force))
    print("[PASS] frozen neural potential autograd/finite-difference self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
