"""Conservative one-dimensional bonded spline representation.

A conservative spline stores a scalar energy profile ``U(q)`` and its nodal
first derivatives ``dU/dq`` on a uniform grid.  Runtime and preprocessing use
the same cubic Hermite polynomial on every interval, so force is obtained by
analytical differentiation of the *same* interpolant used for energy.

The representation is intentionally independent of ESPResSo.  The companion
ESPResSo plugin implements the same formulas in C++.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA = "pchip_hermite_v1"
SUPPORTED_KINDS = {"bond", "angle", "dihedral"}


@dataclass(frozen=True)
class ConservativeSplinePrior:
    x: np.ndarray
    energy: np.ndarray
    derivative: np.ndarray
    minimum: float
    maximum: float
    kind: str
    path: Path
    schema: str = SCHEMA


def resolve_spline_path(filename: str | Path, priors_path: str | Path | None = None) -> Path:
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    if priors_path is not None:
        return Path(priors_path).expanduser().resolve().parent / path
    return Path.cwd() / path


def load_conservative_spline(
    prior: Mapping[str, object],
    *,
    kind: str,
    priors_path: str | Path | None = None,
) -> ConservativeSplinePrior:
    if kind not in SUPPORTED_KINDS:
        raise ValueError(
            f"Conservative spline kind {kind!r} is not supported by schema {SCHEMA}; "
            "current certified scope is bond+angle+dihedral"
        )
    if str(prior.get("spline_schema", SCHEMA)) != SCHEMA:
        raise ValueError(f"Unsupported conservative spline schema: {prior.get('spline_schema')!r}")
    for key in ("file", "min", "max"):
        if key not in prior:
            raise ValueError(f"Conservative spline {kind} prior is missing {key!r}")

    path = resolve_spline_path(str(prior["file"]), priors_path)
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 3:
        raise ValueError(
            f"Conservative spline file must contain >=2 rows and x/U/dU_dq columns: {path}"
        )
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

    minimum = float(prior["min"])
    maximum = float(prior["max"])
    if not maximum > minimum:
        raise ValueError(f"Invalid conservative spline range [{minimum}, {maximum}]")
    if not np.isclose(x[0], minimum, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Spline first x={x[0]} disagrees with prior min={minimum}: {path}")
    if not np.isclose(x[-1], maximum, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Spline last x={x[-1]} disagrees with prior max={maximum}: {path}")
    if kind == "angle":
        if not np.isclose(minimum, 0.0, atol=1.0e-12) or not np.isclose(maximum, np.pi, atol=1.0e-10):
            raise ValueError("Conservative angle splines must span exactly 0..pi")
    elif kind == "dihedral":
        if not np.isclose(minimum, 0.0, atol=1.0e-12) or not np.isclose(maximum, 2.0 * np.pi, atol=1.0e-10):
            raise ValueError("Conservative dihedral splines must span exactly 0..2*pi")
        escale = max(1.0, abs(float(energy[0])), abs(float(energy[-1])))
        dscale = max(1.0, abs(float(derivative[0])), abs(float(derivative[-1])))
        if abs(float(energy[0] - energy[-1])) > 1.0e-10 * escale or abs(float(derivative[0] - derivative[-1])) > 1.0e-10 * dscale:
            raise ValueError("Conservative dihedral spline must have periodic U and dU/dphi endpoints")

    return ConservativeSplinePrior(x, energy, derivative, minimum, maximum, kind, path)


def _interval(table: ConservativeSplinePrior, coordinate: float) -> tuple[int, float, float]:
    """Return interval index, local Hermite coordinate and grid spacing.

    Coordinates below the first node use conservative tangent extrapolation.
    Bond coordinates at/above ``maximum`` are rejected by the bonded wrapper,
    matching the existing finite-domain distance-bond semantics.  Angles are
    clamped to their exact geometric domain only for floating-point roundoff.
    """
    q = float(coordinate)
    h = (table.maximum - table.minimum) / (len(table.x) - 1)
    if q <= table.minimum:
        return 0, (q - table.minimum) / h, h
    if q >= table.maximum:
        return len(table.x) - 2, (q - table.x[-2]) / h, h
    scaled = (q - table.minimum) / h
    i = min(int(np.floor(scaled)), len(table.x) - 2)
    return i, scaled - i, h


def conservative_spline_value(
    table: ConservativeSplinePrior,
    coordinate: float,
) -> tuple[float, float]:
    """Evaluate ``(U, dU/dq)`` from the same cubic Hermite polynomial."""
    q = float(coordinate)
    if table.kind == "angle":
        q = float(np.clip(q, table.minimum, table.maximum))
    elif table.kind == "dihedral":
        period = table.maximum - table.minimum
        q = table.minimum + ((q - table.minimum) % period)

    # Tangent continuation below the distance grid is exactly conservative and
    # avoids the unrelated quadratic energy/linear-force extrapolation of the
    # legacy ESPResSo tabulated interaction.
    if q < table.minimum:
        dq = q - table.minimum
        return (
            float(table.energy[0] + table.derivative[0] * dq),
            float(table.derivative[0]),
        )

    i, t, h = _interval(table, q)
    y0, y1 = float(table.energy[i]), float(table.energy[i + 1])
    m0, m1 = float(table.derivative[i]), float(table.derivative[i + 1])

    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    energy = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    derivative = (dh00 * y0 + dh01 * y1) / h + dh10 * m0 + dh11 * m1
    return float(energy), float(derivative)


def conservative_distance_forces(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    box_dim: np.ndarray,
    table: ConservativeSplinePrior,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(pos_j, dtype=float) - np.asarray(pos_i, dtype=float)
    box = np.asarray(box_dim, dtype=float)
    delta -= box * np.round(delta / box)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-15:
        raise ValueError("ConservativeSplineDistance is undefined for zero endpoint separation")
    if distance >= table.maximum:
        raise ValueError(
            f"Conservative spline distance bond is outside its runtime domain: "
            f"r={distance:.12g} >= max={table.maximum:.12g}"
        )
    _energy, derivative = conservative_spline_value(table, distance)
    # Existing ESPResSo distance geometry consumes the radial force -dU/dr.
    radial = -derivative
    force_i = -radial * delta / distance
    return force_i, -force_i


def conservative_angle_forces(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_k: np.ndarray,
    box_dim: np.ndarray,
    table: ConservativeSplinePrior,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = np.asarray(box_dim, dtype=float)
    r_ji = np.asarray(pos_i, dtype=float) - np.asarray(pos_j, dtype=float)
    r_jk = np.asarray(pos_k, dtype=float) - np.asarray(pos_j, dtype=float)
    r_ji -= box * np.round(r_ji / box)
    r_jk -= box * np.round(r_jk / box)
    d_ji = float(np.linalg.norm(r_ji))
    d_jk = float(np.linalg.norm(r_jk))
    if d_ji <= 1.0e-15 or d_jk <= 1.0e-15:
        raise ValueError("ConservativeSplineAngle is undefined for zero-length legs")
    cos_theta = float(np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0))
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    if sin_theta <= 1.0e-12:
        raise ValueError("ConservativeSplineAngle force is singular at theta=0 or pi")
    theta = float(np.arccos(cos_theta))
    _energy, gradient = conservative_spline_value(table, theta)

    grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji * d_ji)
    grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk * d_jk)
    force_i = (gradient / sin_theta) * grad_i_cos
    force_k = (gradient / sin_theta) * grad_k_cos
    force_j = -(force_i + force_k)
    return force_i, force_j, force_k


def conservative_dihedral_forces(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_k: np.ndarray,
    pos_l: np.ndarray,
    box_dim: np.ndarray,
    table: ConservativeSplinePrior,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative dihedral forces using ESPResSo's torsion geometry."""
    from prior_kernels import espresso_dihedral_geometry

    geom = espresso_dihedral_geometry(pos_i, pos_j, pos_k, pos_l, box_dim)
    if geom is None:
        zeros = np.zeros(3, dtype=float)
        return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
    phi, _cos_phi, v12, v23, v34, n12, l12, n23, l23 = geom
    _energy, derivative = conservative_spline_value(table, phi)
    bnorm = float(np.linalg.norm(v23))
    b2 = bnorm * bnorm
    grad_i = -(bnorm / l12) * n12
    grad_l = (bnorm / l23) * n23
    a = float(np.dot(v12, v23) / b2)
    c = float(np.dot(v34, v23) / b2)
    grad_j = -(1.0 + a) * grad_i + c * grad_l
    grad_k = a * grad_i - (1.0 + c) * grad_l
    force_i = -derivative * grad_i
    force_j = -derivative * grad_j
    force_k = -derivative * grad_k
    force_l = -derivative * grad_l
    return force_i, force_j, force_k, force_l


def save_conservative_spline(path: str | Path, x, energy, derivative) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((
        np.asarray(x, dtype=np.float64),
        np.asarray(energy, dtype=np.float64),
        np.asarray(derivative, dtype=np.float64),
    ))
    np.savetxt(
        path,
        data,
        fmt="%.17g",
        header="q  U_kJmol  dU_dq   # conservative spline pchip_hermite_v1",
    )
