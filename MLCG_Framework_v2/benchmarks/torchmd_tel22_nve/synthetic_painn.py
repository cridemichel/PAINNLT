#!/usr/bin/env python3
"""Deterministic PaiNN-like conservative potential for isolated TorchMD NVE diagnostics."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticPaiNNConfig:
    hidden_channels: int = 64
    num_layers: int = 2
    num_rbf: int = 32
    num_species: int = 8
    neighbors_per_particle: int = 32
    cutoff_a: float = 12.616
    toxvaerd_alpha: float = 0.1
    epsilon: float = 1.0e-8
    seed: int = 220528
    residual_force_fraction: float = 0.50

    def validate(self) -> None:
        if self.hidden_channels < 2:
            raise ValueError("hidden_channels must be >= 2")
        if self.num_layers < 1 or self.num_rbf < 2 or self.num_species < 1:
            raise ValueError("num_layers, num_rbf, and num_species must be positive")
        if self.neighbors_per_particle < 1:
            raise ValueError("neighbors_per_particle must be positive")
        if self.cutoff_a <= 0.0 or self.toxvaerd_alpha <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("cutoff, toxvaerd_alpha, and epsilon must be positive")
        if not (0.0 < self.residual_force_fraction <= 2.0):
            raise ValueError("residual_force_fraction must be in (0, 2]")

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cutoff_A"] = out.pop("cutoff_a")
        return out


def fixed_knn_edge_index(equilibrium: np.ndarray, neighbors_per_particle: int) -> np.ndarray:
    """Build a deterministic directed kNN graph from equilibrium coordinates."""
    xyz = np.asarray(equilibrium, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("equilibrium must have shape [N,3]")
    n = xyz.shape[0]
    if not 1 <= neighbors_per_particle < n:
        raise ValueError("neighbors_per_particle must satisfy 1 <= k < N")
    delta = xyz[:, None, :] - xyz[None, :, :]
    d2 = np.sum(delta * delta, axis=-1)
    np.fill_diagonal(d2, np.inf)
    # Stable lexicographic tie breaking: distance first, particle index second.
    indices = np.arange(n)
    neigh = np.empty((n, neighbors_per_particle), dtype=np.int64)
    for receiver in range(n):
        order = np.lexsort((indices, d2[receiver]))
        neigh[receiver] = order[:neighbors_per_particle]
    col = np.repeat(np.arange(n, dtype=np.int64), neighbors_per_particle)
    row = neigh.reshape(-1)
    return np.stack((row, col), axis=0)


def species_pattern(n: int, num_species: int) -> np.ndarray:
    if n < 1 or num_species < 1:
        raise ValueError("n and num_species must be positive")
    return (np.arange(n, dtype=np.int64) % num_species).astype(np.int64)


def _init_module_weights(torch, model, seed: int) -> str:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                module.weight.copy_(
                    torch.randn(module.weight.shape, generator=generator, dtype=torch.float64) * 0.40
                )
            elif isinstance(module, torch.nn.Linear):
                scale = 0.80 / math.sqrt(max(module.in_features, 1))
                module.weight.copy_(
                    torch.randn(module.weight.shape, generator=generator, dtype=torch.float64) * scale
                )
                if module.bias is not None:
                    module.bias.copy_(
                        torch.randn(module.bias.shape, generator=generator, dtype=torch.float64) * 0.05
                    )
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_synthetic_painn(torch, config: SyntheticPaiNNConfig, *, edge_index, species, dtype, device):
    """Build a deterministic Python implementation mirroring MLCG's PaiNN block structure."""
    config.validate()
    H = config.hidden_channels
    R = config.num_rbf

    class MessageBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scalar_mlp = torch.nn.Sequential(
                torch.nn.Linear(H, H, dtype=torch.float64),
                torch.nn.SiLU(),
                torch.nn.Linear(H, 3 * H, dtype=torch.float64),
            )
            self.filter_mlp = torch.nn.Linear(R, 3 * H, bias=False, dtype=torch.float64)

        def forward(self, s, v, edge_index_t, rbf, rhat):
            row, col = edge_index_t[0], edge_index_t[1]
            w = self.filter_mlp(rbf)
            interaction = self.scalar_mlp(s.index_select(0, row)) * w
            ds_edge, dv_edge, dr_edge = interaction.chunk(3, dim=1)
            delta_v_edges = (
                v.index_select(0, row) * dv_edge.unsqueeze(1)
                + dr_edge.unsqueeze(1) * rhat.unsqueeze(2)
            )
            delta_s = torch.zeros_like(s)
            delta_v = torch.zeros_like(v)
            delta_s.index_add_(0, col, ds_edge)
            delta_v.index_add_(0, col, delta_v_edges)
            return delta_s, delta_v

    class UpdateBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_v = torch.nn.Linear(H, H, bias=False, dtype=torch.float64)
            self.linear_u = torch.nn.Linear(H, H, bias=False, dtype=torch.float64)
            self.scalar_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * H, H, dtype=torch.float64),
                torch.nn.SiLU(),
                torch.nn.Linear(H, 3 * H, dtype=torch.float64),
            )

        def forward(self, s, v):
            vv = self.linear_v(v)
            vu = self.linear_u(v)
            vv_norm = torch.sqrt(torch.sum(vv * vv, dim=1) + config.epsilon)
            out = self.scalar_mlp(torch.cat((s, vv_norm), dim=1))
            a, b, c = out.chunk(3, dim=1)
            delta_s = a + torch.sum(vv * vu, dim=1) * b
            return s + delta_s, v + vu * c.unsqueeze(1)

    class SyntheticPaiNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(config.num_species, H, dtype=torch.float64)
            self.messages = torch.nn.ModuleList([MessageBlock() for _ in range(config.num_layers)])
            self.updates = torch.nn.ModuleList([UpdateBlock() for _ in range(config.num_layers)])
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(H, H // 2, dtype=torch.float64),
                torch.nn.SiLU(),
                torch.nn.Linear(H // 2, 1, dtype=torch.float64),
            )
            self.register_buffer("edge_index", edge_index.clone().to(dtype=torch.long))
            self.register_buffer("species", species.clone().to(dtype=torch.long))

        def expansion_rbf(self, d):
            rc = config.cutoff_a
            x = (rc - d) / rc
            xn = torch.pow(x, 4)
            cutoff = xn / (xn + config.toxvaerd_alpha ** 4)
            cutoff = torch.where(d > rc, torch.zeros_like(cutoff), cutoff)
            centers = torch.linspace(0.0, rc, R, dtype=d.dtype, device=d.device)
            sigma = rc / R
            rbf = torch.exp(-torch.pow(d.unsqueeze(1) - centers, 2) / (sigma * sigma))
            return rbf * cutoff.unsqueeze(1)

        def _isolated_reference(self):
            species_all = torch.arange(config.num_species, device=self.species.device, dtype=torch.long)
            s = self.embedding(species_all)
            v = torch.zeros((config.num_species, 3, H), dtype=s.dtype, device=s.device)
            for update in self.updates:
                s, v = update(s, v)
            return self.readout(s).squeeze(-1)

        def forward(self, positions):
            if positions.ndim != 3 or positions.shape[0] != 1 or positions.shape[-1] != 3:
                raise ValueError("positions must have shape [1,N,3]")
            xyz = positions[0]
            row, col = self.edge_index[0], self.edge_index[1]
            rij = xyz.index_select(0, row) - xyz.index_select(0, col)
            d = torch.sqrt(torch.sum(rij * rij, dim=1) + config.epsilon)
            rhat = rij / d.unsqueeze(1)
            rbf = self.expansion_rbf(d)
            s = self.embedding(self.species)
            v = torch.zeros((s.shape[0], 3, H), dtype=s.dtype, device=s.device)
            for message, update in zip(self.messages, self.updates):
                ds, dv = message(s, v, self.edge_index, rbf, rhat)
                s = s + ds
                v = v + dv
                s, v = update(s, v)
            atom_e = self.readout(s).squeeze(-1) - self._isolated_reference().index_select(0, self.species)
            return atom_e.sum().reshape(1)

    model = SyntheticPaiNN()
    fingerprint = _init_module_weights(torch, model, config.seed)
    model._canonical_parameter_sha256 = fingerprint
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_fingerprint(model) -> str:
    return str(model._canonical_parameter_sha256)


def calibrate_residual_scale(torch, model, equilibrium, stiffness, positions, target_fraction: float) -> dict[str, float]:
    """Choose a fixed energy multiplier so PaiNN RMS forces are a requested fraction of harmonic RMS forces."""
    with torch.enable_grad():
        x = positions.detach().to(dtype=torch.float64, device="cpu").requires_grad_(True)
        eq = equilibrium.detach().to(dtype=torch.float64, device="cpu")
        k = stiffness.detach().to(dtype=torch.float64, device="cpu")
        # Build an identical canonical float64 model on CPU when caller model is not already there.
        raw_e = model(x)
        raw_f = -torch.autograd.grad(raw_e.sum(), x, create_graph=False, retain_graph=False)[0]
        harmonic_f = -k * (x - eq)
    raw_rms = float(torch.sqrt(torch.mean(raw_f * raw_f)).item())
    harmonic_rms = float(torch.sqrt(torch.mean(harmonic_f * harmonic_f)).item())
    if not math.isfinite(raw_rms) or raw_rms <= 1.0e-14:
        raise RuntimeError("raw PaiNN force RMS is zero/non-finite; cannot calibrate")
    scale = target_fraction * harmonic_rms / raw_rms
    return {
        "energy_scale": float(scale),
        "raw_painn_force_rms": raw_rms,
        "harmonic_force_rms": harmonic_rms,
        "target_force_fraction": float(target_fraction),
        "scaled_force_fraction": float(scale * raw_rms / harmonic_rms),
    }


class AutogradPaiNNForces:
    """TorchMD force provider for harmonic background + frozen PaiNN residual."""

    def __init__(self, torch, model, equilibrium, stiffness, residual_scale: float):
        self.torch = torch
        self.model = model
        self.equilibrium = equilibrium
        self.stiffness = stiffness
        self.residual_scale = float(residual_scale)
        self.compute_calls = 0

    def energy_tensor(self, pos):
        delta = pos - self.equilibrium
        harmonic = 0.5 * self.torch.sum(self.stiffness * delta * delta, dim=(1, 2))
        residual = self.model(pos) * self.residual_scale
        return harmonic + residual

    def compute(self, pos, box, forces, *args, **kwargs):
        del box, args, kwargs
        self.compute_calls += 1
        with self.torch.enable_grad():
            coordinates = pos.detach().requires_grad_(True)
            energy = self.energy_tensor(coordinates)
            gradient = self.torch.autograd.grad(
                energy.sum(), coordinates, create_graph=False, retain_graph=False, allow_unused=False
            )[0]
        forces.copy_(-gradient.detach())
        return energy.detach().cpu().numpy()
