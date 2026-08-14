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
    return cls(
        min=float(entry["min"]),
        max=float(entry["max"]),
        energy=np.asarray(data[:, 1], dtype=float).tolist(),
        derivative=np.asarray(data[:, 2], dtype=float).tolist(),
    )
