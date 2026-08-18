#!/usr/bin/env python3
"""Small deterministic conservative neural potential used by the TorchMD NVE benchmark."""
from __future__ import annotations

import hashlib
import math
from typing import Any


class NeuralPotentialConfig:
    def __init__(
        self,
        *,
        hidden_channels: int = 64,
        hidden_layers: int = 2,
        modulation_amplitude: float = 0.05,
        length_scale_a: float = 0.50,
        seed: int = 220527,
    ):
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        if not (0.0 < modulation_amplitude < 1.0):
            raise ValueError("modulation_amplitude must be between 0 and 1")
        if length_scale_a <= 0.0:
            raise ValueError("length_scale_a must be positive")
        self.hidden_channels = int(hidden_channels)
        self.hidden_layers = int(hidden_layers)
        self.modulation_amplitude = float(modulation_amplitude)
        self.length_scale_a = float(length_scale_a)
        self.seed = int(seed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_channels": self.hidden_channels,
            "hidden_layers": self.hidden_layers,
            "modulation_amplitude": self.modulation_amplitude,
            "length_scale_A": self.length_scale_a,
            "seed": self.seed,
        }


def build_neural_model(torch, config: NeuralPotentialConfig, *, dtype, device):
    """Build identical deterministic weights in float64, then cast to the requested precision."""

    class ConservativeNeuralConfinement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_features = 1
            for _ in range(config.hidden_layers):
                layers.append(torch.nn.Linear(in_features, config.hidden_channels, dtype=torch.float64))
                layers.append(torch.nn.SiLU())
                in_features = config.hidden_channels
            layers.append(torch.nn.Linear(in_features, 1, dtype=torch.float64))
            self.network = torch.nn.Sequential(*layers)
            self.modulation_amplitude = config.modulation_amplitude
            self.length_scale_sq = config.length_scale_a * config.length_scale_a

        def forward(self, displacement, stiffness):
            r2 = torch.sum(displacement * displacement, dim=-1, keepdim=True)
            x = r2 / self.length_scale_sq
            modulation = 1.0 + self.modulation_amplitude * torch.tanh(self.network(x))
            per_particle = 0.5 * stiffness * r2 * modulation
            return torch.sum(per_particle, dim=(1, 2))

    model = ConservativeNeuralConfinement()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                scale = 0.35 / math.sqrt(max(module.in_features, 1))
                module.weight.copy_(
                    torch.randn(module.weight.shape, generator=generator, dtype=torch.float64) * scale
                )
                module.bias.copy_(
                    torch.randn(module.bias.shape, generator=generator, dtype=torch.float64) * 0.03
                )
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    model._canonical_parameter_sha256 = digest.hexdigest()
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_fingerprint(model) -> str:
    """Hash the pre-cast float64 weights; identical for FP32/FP64 runs with the same config."""
    return str(model._canonical_parameter_sha256)


class AutogradNeuralForces:
    """TorchMD force provider: E from the frozen MLP, F = -dE/dr via torch.autograd.grad."""

    def __init__(self, torch, model, equilibrium, stiffness):
        self.torch = torch
        self.model = model
        self.equilibrium = equilibrium
        self.stiffness = stiffness
        self.compute_calls = 0

    def compute(self, pos, box, forces, *args, **kwargs):
        del box, args, kwargs
        self.compute_calls += 1
        with self.torch.enable_grad():
            coordinates = pos.detach().requires_grad_(True)
            energy = self.model(coordinates - self.equilibrium, self.stiffness)
            gradient = self.torch.autograd.grad(
                energy.sum(),
                coordinates,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
        forces.copy_(-gradient.detach())
        return energy.detach().cpu().numpy()
