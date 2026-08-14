"""Numerically testable prior kernels used during dataset construction.

The tabulated bonded kernels in this module mirror ESPResSo 5.x semantics:
linear interpolation of the supplied table values, clamping to the tabulated
interval inside ``TabulatedPotential``, and the bonded distance/angle/dihedral
force geometry implemented by ``bonded_tab.hpp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TabulatedPrior:
    """Validated uniformly sampled ESPResSo bonded table."""

    x: np.ndarray
    energy: np.ndarray
    force: np.ndarray
    minimum: float
    maximum: float
    kind: str
    path: Path


def switched_morse_radial_force_array(r, D, a, r0, r_switch, r_cut):
    """Return signed radial switched-Morse forces for NumPy arrays.

    This is the preprocessing counterpart of the analytic switched Morse used
    by the ESPResSo runtime.  The sign convention is radial force along the
    vector from the second particle to the first particle.
    """
    r = np.asarray(r, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    r_switch = np.asarray(r_switch, dtype=np.float64)
    r_cut = np.asarray(r_cut, dtype=np.float64)
    force = np.zeros_like(r)
    active = r < r_cut
    if not np.any(active):
        return force

    rr = r[active]
    DD = D[active]
    aa = a[active]
    rr0 = r0[active]
    rsw = r_switch[active]
    rcut = r_cut[active]
    y = np.exp(-aa * (rr - rr0))
    base_energy = DD * (y * y - 2.0 * y)
    base_force = 2.0 * DD * aa * y * (y - 1.0)
    local_force = base_force.copy()

    switched = rr > rsw
    if np.any(switched):
        width = rcut[switched] - rsw[switched]
        t = (rr[switched] - rsw[switched]) / width
        switch = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
        d_switch_dr = -30.0 * t * t * (1.0 - t) * (1.0 - t) / width
        local_force[switched] = (
            switch * base_force[switched]
            - base_energy[switched] * d_switch_dr
        )

    force[active] = local_force
    return force


def resolve_tabulated_path(filename: str | Path, priors_path: str | Path | None = None) -> Path:
    """Resolve a table path relative to the JSON file that references it."""
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    if priors_path is not None:
        return Path(priors_path).expanduser().resolve().parent / path
    return Path.cwd() / path


def load_tabulated_prior(
    prior: Mapping[str, object],
    *,
    kind: str,
    priors_path: str | Path | None = None,
) -> TabulatedPrior:
    """Load and validate an ESPResSo bonded table.

    ``kind`` is one of ``bond``, ``angle`` or ``dihedral``.  Angle and
    dihedral tables are required to span exactly the domains used internally
    by ESPResSo (0..pi and 0..2*pi respectively).  This matters because the
    C++ constructors overwrite their ``minval``/``maxval`` after the table
    spacing has already been computed.
    """
    if kind not in {"bond", "angle", "dihedral"}:
        raise ValueError(f"Unsupported tabulated prior kind: {kind!r}")
    if "file" not in prior:
        raise ValueError(f"Tabulated {kind} prior is missing 'file'")
    if "min" not in prior or "max" not in prior:
        raise ValueError(f"Tabulated {kind} prior requires explicit 'min' and 'max'")

    path = resolve_tabulated_path(str(prior["file"]), priors_path)
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] < 2:
        raise ValueError(f"Tabulated {kind} file must contain >=2 rows and x/energy/force columns: {path}")

    x = np.asarray(data[:, 0], dtype=np.float64)
    energy = np.asarray(data[:, 1], dtype=np.float64)
    force = np.asarray(data[:, 2], dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force))):
        raise ValueError(f"Tabulated {kind} file contains NaN/Inf: {path}")
    spacing = np.diff(x)
    if np.any(spacing <= 0.0):
        raise ValueError(f"Tabulated {kind} grid must be strictly increasing: {path}")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"ESPResSo requires a uniform tabulated grid: {path}")

    minimum = float(prior["min"])
    maximum = float(prior["max"])
    if not maximum > minimum:
        raise ValueError(f"Invalid tabulated {kind} range [{minimum}, {maximum}]")
    if not np.isclose(x[0], minimum, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Table first x={x[0]} disagrees with prior min={minimum}: {path}")
    if not np.isclose(x[-1], maximum, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Table last x={x[-1]} disagrees with prior max={maximum}: {path}")

    if kind == "angle":
        if not np.isclose(minimum, 0.0, atol=1.0e-12) or not np.isclose(maximum, np.pi, atol=1.0e-10):
            raise ValueError("ESPResSo TabulatedAngle tables must span exactly 0..pi")
    elif kind == "dihedral":
        if not np.isclose(minimum, 0.0, atol=1.0e-12) or not np.isclose(maximum, 2.0 * np.pi, atol=1.0e-10):
            raise ValueError("ESPResSo TabulatedDihedral tables must span exactly 0..2*pi")

    return TabulatedPrior(x, energy, force, minimum, maximum, kind, path)


def tabulated_value(table: TabulatedPrior, coordinate: float, *, column: str = "force") -> float:
    """Mirror ``TabulatedPotential::{force,energy}`` linear interpolation.

    ESPResSo clamps the lookup coordinate to the table interval before linear
    interpolation.  A distance bond nevertheless breaks at ``r >= max`` in
    the outer bonded-distance wrapper; callers that model a distance bond
    must perform that domain check separately.
    """
    values = table.force if column == "force" else table.energy
    if column not in {"force", "energy"}:
        raise ValueError("column must be 'force' or 'energy'")
    q = float(np.clip(float(coordinate), table.minimum, table.maximum))
    step = (table.maximum - table.minimum) / (len(values) - 1)
    scaled = (q - table.minimum) / step
    lo = min(int(np.floor(scaled)), len(values) - 2)
    frac = scaled - lo
    if q >= table.maximum:
        lo = len(values) - 2
        frac = 1.0
    return float((1.0 - frac) * values[lo] + frac * values[lo + 1])


def tabulated_distance_forces(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    box_dim: np.ndarray,
    table: TabulatedPrior,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forces on endpoints for an ESPResSo ``TabulatedDistance`` bond."""
    delta = np.asarray(pos_j, dtype=float) - np.asarray(pos_i, dtype=float)
    box = np.asarray(box_dim, dtype=float)
    delta -= box * np.round(delta / box)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-15:
        raise ValueError("TabulatedDistance is undefined for zero endpoint separation")
    if distance >= table.maximum:
        raise ValueError(
            f"TabulatedDistance bond is outside its runtime domain: r={distance:.12g} >= max={table.maximum:.12g}"
        )
    # The table stores the signed radial force convention.  With r_hat from i
    # to j, ESPResSo's force on i is -F_table * r_hat.
    radial = tabulated_value(table, distance, column="force")
    force_i = -radial * delta / distance
    return force_i, -force_i


def tabulated_angle_forces(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_k: np.ndarray,
    box_dim: np.ndarray,
    table: TabulatedPrior,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ESPResSo ``TabulatedAngle`` forces on i, j, k.

    The angle table's ``force`` column is the angular gradient ``dU/dtheta``;
    ESPResSo applies the required ``-1/sin(theta)`` geometric factor internally.
    """
    box = np.asarray(box_dim, dtype=float)
    r_ji = np.asarray(pos_i, dtype=float) - np.asarray(pos_j, dtype=float)
    r_jk = np.asarray(pos_k, dtype=float) - np.asarray(pos_j, dtype=float)
    r_ji -= box * np.round(r_ji / box)
    r_jk -= box * np.round(r_jk / box)
    d_ji = float(np.linalg.norm(r_ji))
    d_jk = float(np.linalg.norm(r_jk))
    if d_ji <= 1.0e-15 or d_jk <= 1.0e-15:
        raise ValueError("TabulatedAngle is undefined for zero-length legs")
    cos_theta = float(np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0))
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    if sin_theta <= 1.0e-12:
        raise ValueError("TabulatedAngle force is singular at theta=0 or pi")
    theta = float(np.arccos(cos_theta))
    gradient = tabulated_value(table, theta, column="force")

    grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji * d_ji)
    grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk * d_jk)
    force_i = (gradient / sin_theta) * grad_i_cos
    force_k = (gradient / sin_theta) * grad_k_cos
    force_j = -(force_i + force_k)
    return force_i, force_j, force_k


def espresso_dihedral_geometry(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_k: np.ndarray,
    pos_l: np.ndarray,
    box_dim: np.ndarray,
):
    """Reproduce ESPResSo ``calc_dihedral_angle`` geometry.

    Returns ``(phi, cos_phi, v12, v23, v34, n12, l12, n23, l23)`` with
    ``phi`` in [0, 2*pi), or ``None`` when the dihedral is undefined.
    """
    box = np.asarray(box_dim, dtype=float)

    def mic(a, b):
        v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        return v - box * np.round(v / box)

    v12 = mic(pos_i, pos_j)
    v23 = mic(pos_j, pos_k)
    v34 = mic(pos_k, pos_l)
    n12 = np.cross(v12, v23)
    n23 = np.cross(v23, v34)
    l12 = float(np.linalg.norm(n12))
    l23 = float(np.linalg.norm(n23))
    if l12 <= 1.0e-4 or l23 <= 1.0e-4:
        return None
    n12 = n12 / l12
    n23 = n23 / l23
    # Match ESPResSo calc_dihedral_angle(): normalize the two plane normals,
    # round values numerically indistinguishable from +/-1, then take acos.
    cos_phi = float(np.dot(n12, n23))
    if abs(abs(cos_phi) - 1.0) < 1.0e-10:
        cos_phi = float(np.round(cos_phi))
    phi = float(np.arccos(cos_phi))
    if float(np.dot(n12, v34)) < 0.0:
        phi = 2.0 * np.pi - phi
    return phi, cos_phi, v12, v23, v34, n12, l12, n23, l23


def tabulated_dihedral_forces(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_k: np.ndarray,
    pos_l: np.ndarray,
    box_dim: np.ndarray,
    table: TabulatedPrior,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ESPResSo ``TabulatedDihedral`` forces on i, j, k, l.

    ESPResSo does *not* interpret the dihedral table's third column as the raw
    scalar ``-dU/dphi``.  It multiplies that value directly by its torsional
    geometry vectors.  IBI table generation must therefore store the same
    force-factor convention.
    """
    geom = espresso_dihedral_geometry(pos_i, pos_j, pos_k, pos_l, box_dim)
    if geom is None:
        zeros = np.zeros(3, dtype=float)
        return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
    phi, cos_phi, v12, v23, v34, n12, l12, n23, l23 = geom
    f1 = (n23 - cos_phi * n12) / l12
    f4 = (n12 - cos_phi * n23) / l23
    v23_x_f1 = np.cross(v23, f1)
    v23_x_f4 = np.cross(v23, f4)
    v34_x_f4 = np.cross(v34, f4)
    v12_x_f1 = np.cross(v12, f1)
    fac = tabulated_value(table, phi, column="force")

    # C++ return order is (p2, p1, p3, p4). Reorder to i,j,k,l.
    force_i = fac * v23_x_f1
    force_j = fac * (v34_x_f4 - v12_x_f1 - v23_x_f1)
    force_k = fac * (v12_x_f1 - v23_x_f4 - v34_x_f4)
    force_l = -(force_i + force_j + force_k)
    return force_i, force_j, force_k, force_l
