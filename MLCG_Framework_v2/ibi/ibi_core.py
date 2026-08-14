"""Reusable DBI/IBI numerical core for MLCG Framework v2.

This module intentionally contains no ESPResSo orchestration.  It generates
ESPResSo-compatible bonded tables; simulation/sampling is a separate layer.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, PchipInterpolator

DEFAULT_IBI_SETTINGS = {
    "kT": 2.49,
    "alpha": 0.25,
    "min_count": 3,
    "relative_density_threshold": 1.0e-4,
    "max_gap_bins": 3,
    "min_support_points": 8,
    "avgpoints": 7,
    "taper_bins": 12,
    "histogram_smoothing_sigma": 1.5,
    "update_smoothing_sigma": 1.0,
    "max_update_kT": 3.0,
    "bond": {
        "hist_min": 0.01,
        "hist_max": 3.0,
        "hist_edges": 300,
        "table_min": 0.01,
        "table_max": 5.0,
        "table_points": 4001,
        "left_function": "exponential",
        "right_function": "exponential",
        "left_curvature": 6.0,
        "right_curvature": 1.0,
        "left_guard": 0.05,
        "right_guard": 2.6,
        "left_guard_force": 100.0,
        "right_guard_force": 75.0,
        "force_max": 150.0,
    },
    "angle": {
        "hist_edges": 300,
        "table_points": 2001,
        "wall_width": 0.1,
        "wall_k": 5000.0,
        "force_max": 150.0,
    },
    "dihedral": {
        "hist_edges": 300,
        "table_points": 2001,
        "force_max": 150.0,
    },
    "simulation": {
        "dt": 0.0005,
        "burn_in_steps": 8000,
        "steps": 40000,
        "log_interval": 40,
    },
}


def recursive_update(base, override):
    """Recursively merge JSON configuration values into *base*."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def load_ibi_settings(filename=None):
    """Load IBI/extrapolation settings while retaining safe defaults."""
    settings = copy.deepcopy(DEFAULT_IBI_SETTINGS)
    if filename:
        with open(filename, "r") as handle:
            override = json.load(handle)
        recursive_update(settings, override)
    return settings


def normalize_density(hist, grid):
    """Return a non-negative probability density normalized on *grid*."""
    hist = np.clip(np.asarray(hist, dtype=float), 0.0, None)
    grid = np.asarray(grid, dtype=float)
    if hist.shape != grid.shape:
        raise ValueError(f"Histogram/grid shape mismatch: {hist.shape} vs {grid.shape}")
    norm = np.trapezoid(hist, grid)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize an empty or non-finite distribution")
    return hist / norm


def histogram_density(values, bins):
    """Return counts, normalized density and centers without silent truncation."""
    values = np.asarray(values, dtype=float)
    edges = np.asarray(bins, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Histogram values must be a non-empty finite one-dimensional array")
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all():
        raise ValueError("Histogram bin edges must be a finite one-dimensional array")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("Histogram bin edges must be strictly increasing")
    below = int(np.count_nonzero(values < edges[0]))
    above = int(np.count_nonzero(values > edges[-1]))
    if below or above:
        raise ValueError(
            f"Histogram range [{edges[0]:.12g}, {edges[-1]:.12g}] excludes "
            f"{below + above}/{values.size} samples ({below} below, {above} above); "
            "expand the IBI histogram range instead of silently dropping data"
        )
    counts, edges = np.histogram(values, bins=edges, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    density = normalize_density(counts.astype(float) / widths, centers)
    return counts.astype(float), density, centers


def _fill_short_gaps(mask, max_gap_bins):
    mask = np.asarray(mask, dtype=bool).copy()
    true_idx = np.flatnonzero(mask)
    if true_idx.size < 2:
        return mask
    for left, right in zip(true_idx[:-1], true_idx[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap_bins:
            mask[left + 1:right] = True
    return mask


def statistical_support(counts, density, settings):
    """Identify the statistically supported interval, VOTCA-style.

    VOTCA marks reliable table samples as interpolation points and extrapolates
    the unsupported outer regions. Here reliability requires both a minimum
    raw count and a minimum density relative to the distribution maximum.
    Short internal holes are bridged, but outer unsampled tails remain excluded.
    """
    counts = np.asarray(counts, dtype=float)
    density = np.asarray(density, dtype=float)
    if counts.shape != density.shape:
        raise ValueError("Counts and density must have matching shapes")
    if not np.isfinite(density).all() or np.max(density) <= 0.0:
        raise ValueError("Distribution contains no finite statistical support")

    threshold = float(settings["relative_density_threshold"]) * float(np.max(density))
    sample_mask = (counts >= int(settings["min_count"])) & (density >= threshold)
    extent_mask = _fill_short_gaps(sample_mask, int(settings["max_gap_bins"]))

    if np.count_nonzero(sample_mask) < int(settings["min_support_points"]):
        sample_mask = counts > 0
        extent_mask = _fill_short_gaps(sample_mask, int(settings["max_gap_bins"]))
        if np.count_nonzero(sample_mask) < 4:
            raise ValueError("Too few populated histogram bins for stable extrapolation")

    idx = np.flatnonzero(extent_mask)
    return sample_mask, int(idx[0]), int(idx[-1])


def make_spline(x, y, periodic=False):
    """Build a cubic spline, closing the last interval for periodic data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Spline data must be one-dimensional and have matching shapes")
    if periodic:
        dx = x[1] - x[0]
        period = (x[-1] - x[0]) + dx
        x_ext = np.concatenate((x, [x[0] + period]))
        y_ext = np.concatenate((y, [y[0]]))
        return CubicSpline(x_ext, y_ext, bc_type="periodic")
    return CubicSpline(x, y, bc_type="not-a-knot")


def _boundary_fit(x, potential, side, avgpoints, fit_width=None):
    """Estimate boundary energy and slope from several supported points."""
    x = np.asarray(x, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if side == "left":
        boundary = x[0]
        if fit_width is None:
            sel = np.arange(min(avgpoints, len(x)))
        else:
            sel = np.flatnonzero(x <= boundary + fit_width)
            sel = sel[:max(avgpoints, 2)] if sel.size > max(avgpoints, 2) else sel
    else:
        boundary = x[-1]
        if fit_width is None:
            sel = np.arange(max(0, len(x) - avgpoints), len(x))
        else:
            sel = np.flatnonzero(x >= boundary - fit_width)
            sel = sel[-max(avgpoints, 2):] if sel.size > max(avgpoints, 2) else sel

    if sel.size < 2:
        raise ValueError("At least two supported points are required at each boundary")
    slope, intercept = np.polyfit(x[sel], potential[sel], 1)
    value = slope * boundary + intercept
    return float(value), float(slope)


def _tail_potential(x_tail, boundary_x, boundary_u, boundary_slope, side, function,
                    curvature, guard_x, guard_force):
    """Construct a C1 tail with linear, quadratic or exponential curvature.

    The correction has zero value and zero first derivative at the support
    boundary, so the fitted interior slope is retained exactly.  The configured
    guard force is reached at *guard_x* and then grows until the table force cap
    is applied.  This is more robust than extrapolating noisy clipped log-ratios.
    """
    x_tail = np.asarray(x_tail, dtype=float)
    if x_tail.size == 0:
        return np.empty(0, dtype=float)

    if side == "left":
        distance = boundary_x - x_tail
        guard_distance = max(boundary_x - float(guard_x), float(np.max(distance)) * 0.25, 1e-8)
        target_slope = -abs(float(guard_force))
        linear = boundary_u - boundary_slope * distance
        required_change = max(0.0, boundary_slope - target_slope)
    elif side == "right":
        distance = x_tail - boundary_x
        guard_distance = max(float(guard_x) - boundary_x, float(np.max(distance)) * 0.25, 1e-8)
        target_slope = abs(float(guard_force))
        linear = boundary_u + boundary_slope * distance
        required_change = max(0.0, target_slope - boundary_slope)
    else:
        raise ValueError(f"Unknown tail side: {side}")

    function = str(function).lower()
    if function in {"linear", "constant"} or required_change == 0.0:
        return linear

    if function == "quadratic":
        coefficient = required_change / (2.0 * guard_distance)
        return linear + coefficient * distance**2

    if function == "exponential":
        kappa = max(float(curvature), 1e-8)
        guard_arg = min(kappa * guard_distance, 50.0)
        denominator = kappa * np.expm1(guard_arg)
        coefficient = required_change / max(denominator, 1e-12)
        arg = np.clip(kappa * distance, 0.0, 50.0)
        return linear + coefficient * (np.exp(arg) - 1.0 - arg)

    raise ValueError(f"Unsupported extrapolation function: {function}")


def extrapolate_supported_potential(hist_x, potential_hist, support_mask, target_grid,
                                    settings, target_type):
    """Interpolate supported samples and extrapolate only the unsupported tails."""
    hist_x = np.asarray(hist_x, dtype=float)
    potential_hist = np.asarray(potential_hist, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    target_grid = np.asarray(target_grid, dtype=float)

    if target_type == "dihedral":
        filled = np.interp(hist_x, hist_x[support_mask], potential_hist[support_mask], period=2.0 * np.pi)
        spline = make_spline(hist_x, filled, periodic=True)
        result = spline(target_grid)
        result[-1] = result[0]
        result -= np.min(result)
        return result, (0.0, 2.0 * np.pi)

    valid_x = hist_x[support_mask]
    valid_u = potential_hist[support_mask]
    if valid_x.size < 4:
        raise ValueError("At least four supported samples are required for PCHIP interpolation")

    first_x, last_x = float(valid_x[0]), float(valid_x[-1])
    interior = (target_grid >= first_x) & (target_grid <= last_x)
    if not np.any(interior):
        raise ValueError("Supported histogram interval does not overlap the table grid")

    interpolator = PchipInterpolator(valid_x, valid_u, extrapolate=False)
    result = np.empty_like(target_grid)
    result[interior] = interpolator(target_grid[interior])

    avgpoints = int(settings["avgpoints"])
    hist_dx = float(np.median(np.diff(hist_x)))
    fit_width = max(avgpoints, 2) * hist_dx
    interior_x = target_grid[interior]
    interior_u = result[interior]
    left_u, left_slope = _boundary_fit(interior_x, interior_u, "left", avgpoints, fit_width)
    right_u, right_slope = _boundary_fit(interior_x, interior_u, "right", avgpoints, fit_width)

    if target_type == "bond":
        cfg = settings["bond"]
        left = target_grid < first_x
        right = target_grid > last_x
        result[left] = _tail_potential(
            target_grid[left], first_x, left_u, left_slope, "left",
            cfg["left_function"], cfg["left_curvature"],
            cfg["left_guard"], cfg["left_guard_force"],
        )
        result[right] = _tail_potential(
            target_grid[right], last_x, right_u, right_slope, "right",
            cfg["right_function"], cfg["right_curvature"],
            cfg["right_guard"], cfg["right_guard_force"],
        )
    else:
        # Angles remain inside their exact physical domain.  Continue the fitted
        # slope to the endpoints; add_angle_walls supplies conservative barriers.
        left = target_grid < first_x
        right = target_grid > last_x
        result[left] = left_u + left_slope * (target_grid[left] - first_x)
        result[right] = right_u + right_slope * (target_grid[right] - last_x)

    result -= np.min(result)
    return result, (first_x, last_x)


def _cosine_taper(x, left, right, width):
    weights = np.zeros_like(x, dtype=float)
    inside = (x >= left) & (x <= right)
    weights[inside] = 1.0
    width = max(float(width), 0.0)
    if width <= 0.0 or right <= left:
        return weights
    effective = min(width, 0.5 * (right - left))
    left_ramp = (x >= left) & (x < left + effective)
    right_ramp = (x > right - effective) & (x <= right)
    weights[left_ramp] = 0.5 * (1.0 - np.cos(np.pi * (x[left_ramp] - left) / effective))
    weights[right_ramp] = 0.5 * (1.0 - np.cos(np.pi * (right - x[right_ramp]) / effective))
    return weights


def integrate_tabulated_force(x, force, target_type, reference_energy=None):
    """Integrate the force convention used by ESPResSo bonded tables.

    Distance tables store ``-dU/dr``.  Angle tables store ``+dU/dtheta``.
    Dihedral tables store the C++ torsional force factor, which is related to
    the physical derivative by ``dU/dphi = -force_factor * sin(phi)`` away
    from the geometric singularities.
    """
    x = np.asarray(x, dtype=float)
    force = np.asarray(force, dtype=float)
    if x.shape != force.shape:
        raise ValueError("Force and table grid must have matching shapes")

    if reference_energy is None:
        anchor = 0
        anchor_value = 0.0
    else:
        reference_energy = np.asarray(reference_energy, dtype=float)
        anchor = int(np.argmin(reference_energy))
        anchor_value = float(reference_energy[anchor])

    if target_type == "angle":
        derivative = force
    elif target_type == "bond":
        derivative = -force
    elif target_type == "dihedral":
        derivative = -force * np.sin(x)
    else:
        raise ValueError(f"Unsupported target_type: {target_type!r}")

    energy = np.zeros_like(force)
    energy[anchor] = anchor_value
    for i in range(anchor, len(x) - 1):
        dx = x[i + 1] - x[i]
        energy[i + 1] = energy[i] + 0.5 * (derivative[i] + derivative[i + 1]) * dx
    for i in range(anchor, 0, -1):
        dx = x[i] - x[i - 1]
        energy[i - 1] = energy[i] - 0.5 * (derivative[i] + derivative[i - 1]) * dx

    energy -= np.min(energy)
    return energy


def add_angle_walls(x, energy, espresso_gradient, wall_width=0.1, wall_k=5000.0):
    """Add conservative walls using the TabulatedAngle +dU/dtheta convention."""
    x = np.asarray(x, dtype=float)
    energy = np.asarray(energy, dtype=float).copy()
    espresso_gradient = np.asarray(espresso_gradient, dtype=float).copy()

    left = x < wall_width
    dx_left = wall_width - x[left]
    energy[left] += 0.5 * wall_k * dx_left**2
    espresso_gradient[left] -= wall_k * dx_left

    right_edge = np.pi - wall_width
    right = x > right_edge
    dx_right = x[right] - right_edge
    energy[right] += 0.5 * wall_k * dx_right**2
    espresso_gradient[right] += wall_k * dx_right

    energy -= np.min(energy)
    return energy, espresso_gradient


def _dihedral_force_factor(grid, derivative, force_max):
    """Convert dU/dphi to ESPResSo TabulatedDihedral's force-factor table."""
    grid = np.asarray(grid, dtype=float)
    derivative = np.asarray(derivative, dtype=float)
    sin_phi = np.sin(grid)
    factor = np.empty_like(derivative)
    regular = np.abs(sin_phi) > 1.0e-6
    factor[regular] = -derivative[regular] / sin_phi[regular]

    # The bonded geometry itself is singular at 0 and pi.  At table nodes
    # exactly on those angles we use the nearest regular one-sided factor;
    # clipping below prevents unstable tails.  This is substantially closer
    # to ESPResSo's C++ convention than the v1 implementation, which stored
    # raw -dU/dphi and therefore had the wrong force geometry everywhere.
    bad = ~regular
    good_idx = np.flatnonzero(regular)
    if good_idx.size == 0:
        raise ValueError("Dihedral grid contains no regular points")
    for idx in np.flatnonzero(bad):
        nearest = good_idx[np.argmin(np.abs(good_idx - idx))]
        factor[idx] = factor[nearest]
    return np.clip(factor, -force_max, force_max)


def table_from_potential(target_grid, potential, target_type, periodic=False, settings=None):
    """Create an ESPResSo-compatible table from a scalar potential profile."""
    settings = settings or DEFAULT_IBI_SETTINGS
    target_grid = np.asarray(target_grid, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if target_grid.shape != potential.shape:
        raise ValueError("Potential and target grid must have matching shapes")
    if target_type not in {"bond", "angle", "dihedral"}:
        raise ValueError(f"Unsupported target type {target_type!r}")

    cfg = settings[target_type]
    force_max = float(cfg["force_max"])
    if periodic:
        potential = potential.copy()
        potential[-1] = potential[0]
        spline = CubicSpline(target_grid, potential, bc_type="periodic")
    else:
        spline = PchipInterpolator(target_grid, potential, extrapolate=False)

    derivative = np.asarray(spline(target_grid, 1), dtype=float)
    if target_type == "angle":
        force = np.clip(derivative, -force_max, force_max)
    elif target_type == "bond":
        force = np.clip(-derivative, -force_max, force_max)
    else:
        force = _dihedral_force_factor(target_grid, derivative, force_max)

    energy = integrate_tabulated_force(target_grid, force, target_type, potential)
    if target_type == "angle":
        energy, force = add_angle_walls(
            target_grid, energy, force,
            wall_width=float(cfg["wall_width"]),
            wall_k=float(cfg["wall_k"]),
        )
    elif periodic:
        energy[-1] = energy[0]
        force[-1] = force[0]
    return energy, force


def validate_extrapolated_table(x, energy, force, target_type, support=None):
    """Fail early if a generated table has unsafe or non-finite tails."""
    x = np.asarray(x, dtype=float)
    energy = np.asarray(energy, dtype=float)
    force = np.asarray(force, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(energy).all() and np.isfinite(force).all()):
        raise ValueError(f"{target_type} table contains NaN or Inf")
    if target_type == "bond":
        edge = max(3, min(25, len(x) // 20))
        if float(np.mean(force[:edge])) <= 0.0:
            raise ValueError("Bond left tail is not repulsive")
        if float(np.mean(force[-edge:])) >= 0.0:
            raise ValueError("Bond right tail is not restoring")
        if support is not None:
            left, right = support
            left_idx = np.searchsorted(x, left)
            right_idx = np.searchsorted(x, right)
            if left_idx > 1 and energy[0] <= energy[left_idx]:
                raise ValueError("Bond left extrapolated energy does not rise toward the boundary")
            if right_idx < len(x) - 1 and energy[-1] <= energy[right_idx]:
                raise ValueError("Bond right extrapolated energy does not rise toward the boundary")


def calculate_dbi_potential(values, bins, target_grid, kT=None, periodic=False,
                            jacobian_type=None, settings=None):
    settings = settings or DEFAULT_IBI_SETTINGS
    kT = float(settings["kT"] if kT is None else kT)
    values = np.asarray(values, dtype=float)
    if jacobian_type == "dihedral":
        values = np.mod(values, 2.0 * np.pi)

    counts, target_hist, bin_centers = histogram_density(values, bins)
    support_mask, first, last = statistical_support(counts, target_hist, settings)

    pmf_hist = target_hist.copy()
    if jacobian_type == "bond":
        pmf_hist /= np.clip(bin_centers**2, 1e-12, None)
    elif jacobian_type == "angle":
        pmf_hist /= np.clip(np.sin(bin_centers), 1e-6, None)
    pmf_hist = normalize_density(pmf_hist, bin_centers)

    potential_hist = -kT * np.log(np.clip(pmf_hist, 1e-300, None))
    potential_hist -= np.min(potential_hist[support_mask])

    valid_x = bin_centers[support_mask]
    valid_u = potential_hist[support_mask]
    valid_u = gaussian_filter1d(
        valid_u,
        sigma=float(settings["histogram_smoothing_sigma"]),
        mode="wrap" if periodic else "nearest",
    )
    smoothed = potential_hist.copy()
    smoothed[support_mask] = valid_u

    potential_grid, support = extrapolate_supported_potential(
        bin_centers, smoothed, support_mask, target_grid, settings, jacobian_type
    )
    energy, force = table_from_potential(
        target_grid, potential_grid, target_type=jacobian_type,
        periodic=periodic, settings=settings,
    )
    validate_extrapolated_table(target_grid, energy, force, jacobian_type, support)
    return target_grid, energy, force, target_hist, bin_centers, counts, support


def update_ibi_potential(
    V_i,
    P_i,
    P_target,
    bin_centers,
    target_grid,
    target_counts,
    sim_counts,
    kT=None,
    alpha=None,
    periodic=False,
    target_type="bond",
    settings=None,
    previous_force=None,
):
    """Apply a support-aware IBI update and reconstruct safe outer tails."""
    settings = settings or DEFAULT_IBI_SETTINGS
    kT = float(settings["kT"] if kT is None else kT)
    alpha = float(settings["alpha"] if alpha is None else alpha)
    V_i = np.asarray(V_i, dtype=float)
    target_grid = np.asarray(target_grid, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)
    if V_i.shape != target_grid.shape:
        raise ValueError(f"V_i/target_grid mismatch: {V_i.shape} vs {target_grid.shape}")

    P_i = normalize_density(P_i, bin_centers)
    P_target = normalize_density(P_target, bin_centers)
    target_mask, _, _ = statistical_support(target_counts, P_target, settings)
    sim_mask, _, _ = statistical_support(sim_counts, P_i, settings)
    update_mask = target_mask & sim_mask

    if np.count_nonzero(update_mask) < int(settings["min_support_points"]):
        print(f"[WARN] {target_type}: insufficient target/simulation overlap; preserving previous potential")
        if previous_force is None:
            previous_force = table_from_potential(
                target_grid, V_i, target_type, periodic=periodic, settings=settings
            )[1]
        return target_grid, V_i.copy(), np.asarray(previous_force, dtype=float).copy()

    delta_values = alpha * kT * np.log(
        np.clip(P_i[update_mask], 1e-300, None)
        / np.clip(P_target[update_mask], 1e-300, None)
    )
    update_limit = float(settings["max_update_kT"]) * kT
    delta_values = np.clip(delta_values, -update_limit, update_limit)
    delta_values = gaussian_filter1d(
        delta_values,
        sigma=float(settings["update_smoothing_sigma"]),
        mode="wrap" if periodic else "nearest",
    )

    if periodic:
        delta_hist = np.interp(
            bin_centers, bin_centers[update_mask], delta_values,
            period=2.0 * np.pi,
        )
        delta_spline = make_spline(bin_centers, delta_hist, periodic=True)
        delta_grid = delta_spline(target_grid)
        potential_grid = V_i + delta_grid
        potential_grid[-1] = potential_grid[0]
        support = (0.0, 2.0 * np.pi)
    else:
        update_x = bin_centers[update_mask]
        left, right = float(update_x[0]), float(update_x[-1])
        interpolator = PchipInterpolator(update_x, delta_values, extrapolate=False)
        delta_grid = np.zeros_like(target_grid)
        inside = (target_grid >= left) & (target_grid <= right)
        delta_grid[inside] = interpolator(target_grid[inside])
        taper_width = int(settings["taper_bins"]) * float(np.median(np.diff(bin_centers)))
        delta_grid *= _cosine_taper(target_grid, left, right, taper_width)
        potential_grid = V_i + delta_grid

        # Rebuild only the unsupported target tails.  The IBI correction is zero
        # there, preventing noisy clipped probabilities from destroying confinement.
        target_support_x = bin_centers[target_mask]
        support_left, support_right = float(target_support_x[0]), float(target_support_x[-1])
        support_grid = (target_grid >= support_left) & (target_grid <= support_right)
        support_mask_grid = np.zeros_like(target_grid, dtype=bool)
        support_mask_grid[support_grid] = True
        potential_grid, support = extrapolate_supported_potential(
            target_grid, potential_grid, support_mask_grid,
            target_grid, settings, target_type,
        )

    potential_grid -= np.min(potential_grid)
    energy, force = table_from_potential(
        target_grid, potential_grid, target_type=target_type,
        periodic=periodic, settings=settings,
    )
    validate_extrapolated_table(target_grid, energy, force, target_type, support)
    return target_grid, energy, force


def save_tabulated_potential(filename, x, energy, force):
    directory = os.path.dirname(str(filename))
    if directory:
        os.makedirs(directory, exist_ok=True)
    x = np.asarray(x, dtype=float)
    energy = np.asarray(energy, dtype=float)
    force = np.asarray(force, dtype=float)
    if x.shape != energy.shape or x.shape != force.shape:
        raise ValueError("Table columns must have identical shapes")
    spacing = np.diff(x)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError(f"ESPResSo requires a uniform table grid: {filename}")
    data = np.column_stack((x, energy, force))
    np.savetxt(filename, data, fmt="%.16e", header="x energy force")
