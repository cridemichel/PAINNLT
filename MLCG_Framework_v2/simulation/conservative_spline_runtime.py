"""Runtime loader for MLCG conservative bonded spline interactions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from framework_utils import resolve_referenced_path


def create_conservative_spline_interaction(espressomd_interactions, entry, *, kind: str, priors_path):
    if kind not in {"bond", "angle"}:
        raise ValueError(f"Unsupported conservative spline runtime kind: {kind!r}")
    class_name = {
        "bond": "ConservativeSplineDistance",
        "angle": "ConservativeSplineAngle",
    }[kind]
    cls = getattr(espressomd_interactions, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"ESPResSo Python class {class_name} is unavailable. Install the MLCG conservative "
            "spline plugin with simulation/espresso_plugin/install_conservative_spline_bond.py "
            "and rebuild ESPResSo before using type='conservative_spline'."
        )
    path = resolve_referenced_path(entry["file"], priors_path)
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 3:
        raise ValueError(f"Invalid conservative spline file: {path}")
    if str(entry.get("spline_schema", "pchip_hermite_v1")) != "pchip_hermite_v1":
        raise ValueError(f"Unsupported conservative spline schema in {path}")

    x = np.asarray(data[:, 0], dtype=np.float64)
    energy = np.asarray(data[:, 1], dtype=np.float64)
    derivative = np.asarray(data[:, 2], dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(derivative))):
        raise ValueError(f"Conservative spline contains NaN/Inf: {path}")
    spacing = np.diff(x)
    if np.any(spacing <= 0.0):
        raise ValueError(f"Conservative spline grid must be strictly increasing: {path}")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"Conservative spline grid must be uniform: {path}")

    minimum = float(entry["min"])
    maximum = float(entry["max"])
    if not maximum > minimum:
        raise ValueError(f"Invalid conservative spline range [{minimum}, {maximum}]: {path}")
    if not np.isclose(x[0], minimum, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Spline first x={x[0]} disagrees with prior min={minimum}: {path}")
    if not np.isclose(x[-1], maximum, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Spline last x={x[-1]} disagrees with prior max={maximum}: {path}")
    if kind == "angle":
        if not np.isclose(minimum, 0.0, atol=1.0e-12) or not np.isclose(maximum, np.pi, atol=1.0e-10):
            raise ValueError(f"Conservative angle spline must span exactly 0..pi: {path}")

    return cls(
        min=minimum,
        max=maximum,
        energy=energy.tolist(),
        derivative=derivative.tolist(),
    )
