import os
import sys
import numpy as np
import argparse
import json
import struct
import subprocess
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, PchipInterpolator

# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "numpy",
#     "scipy"
# ]
# ///

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
        "steps": 40000,
        "log_interval": 40,
        "equilibration_sd_steps": 5000,
        "equilibration_md_steps": 8000,
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
    import copy

    settings = copy.deepcopy(DEFAULT_IBI_SETTINGS)
    if filename:
        with open(filename, "r") as handle:
            override = json.load(handle)
        recursive_update(settings, override)
    return settings

def mic_vector(pos1, pos2, box_dim):
    dvec = pos2 - pos1
    return dvec - box_dim * np.round(dvec / box_dim)

def get_angle(pos_i, pos_j, pos_k, box_dim):
    r_ji = mic_vector(pos_j, pos_i, box_dim)
    r_jk = mic_vector(pos_j, pos_k, box_dim)
    d_ji = np.linalg.norm(r_ji)
    d_jk = np.linalg.norm(r_jk)
    if d_ji < 1e-6 or d_jk < 1e-6: return 0.0
    cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
    return np.arccos(cos_theta)

def get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim):
    b1 = mic_vector(pos_i, pos_j, box_dim)
    b2 = mic_vector(pos_j, pos_k, box_dim)
    b3 = mic_vector(pos_k, pos_l, box_dim)
    m1 = np.cross(b1, b2)
    m2 = np.cross(b2, b3)
    m1_sq = np.dot(m1, m1)
    m2_sq = np.dot(m2, m2)
    if m1_sq < 1e-12 or m2_sq < 1e-12: return 0.0
    b2_norm = np.linalg.norm(b2)
    cos_phi = np.clip(np.dot(m1, m2) / np.sqrt(m1_sq * m2_sq), -1.0, 1.0)
    sin_phi = np.dot(b2, np.cross(m1, m2)) / (b2_norm * np.sqrt(m1_sq * m2_sq))
    return np.arctan2(sin_phi, cos_phi)

def read_dataset_distributions(bin_file, priors):
    """
    Reads the binary dataset and computes exact target distributions
    for bonds, angles, dihedrals defined in cg_priors.json.
    Returns: bond_dists, angle_dists, dihedral_dists, first_frame_centers
    """
    bond_dists = {idx: [] for idx in range(len(priors.get("bonds", [])))}
    angle_dists = {idx: [] for idx in range(len(priors.get("angles", [])))}
    dihedral_dists = {idx: [] for idx in range(len(priors.get("dihedrals", [])))}
    first_frame_centers = None
    first_frame_types = None
    target_box_dim = None
    
    with open(bin_file, "rb") as f:
        data = f.read(4)
        if not data: return bond_dists, angle_dists, dihedral_dists, [], []
        num_frames = struct.unpack("i", data)[0]
        
        for frame_idx in range(num_frames):
            num_molecules = struct.unpack("i", f.read(4))[0]
            num_total_sites = struct.unpack("i", f.read(4))[0]
            box_dim = np.array(struct.unpack("3f", f.read(12)))
            if target_box_dim is None: target_box_dim = box_dim
            
            frame_centers = []
            frame_sites = []
            frame_types = []
            
            for _ in range(num_molecules):
                mol_id = struct.unpack("i", f.read(4))[0]
                num_sites = struct.unpack("i", f.read(4))[0]
                center = np.array(struct.unpack("3f", f.read(12)))
                force = struct.unpack("3f", f.read(12))
                torque = struct.unpack("3f", f.read(12))
                
                sites = []
                first_site_type = None
                for _ in range(num_sites):
                    site_type = struct.unpack("i", f.read(4))[0]
                    if first_site_type is None:
                        first_site_type = site_type
                    site_pos = np.array(struct.unpack("3f", f.read(12)))
                    sites.append(site_pos)
                
                frame_centers.append(center)
                frame_sites.append(sites)
                # In our generic dataset, the CG bead type corresponds to the first site type.
                frame_types.append(first_site_type if first_site_type is not None else 0)
                
            if frame_idx == 0:
                first_frame_centers = frame_centers
                first_frame_types = frame_types
                
            # Extract bond lengths
            for idx, b in enumerate(priors.get("bonds", [])):
                i, j = b["mol_i"], b["mol_j"]
                site_i, site_j = b.get("site_i", -1), b.get("site_j", -1)
                if i >= len(frame_centers) or j >= len(frame_centers): continue
                
                pos_i = frame_centers[i] if site_i == -1 else frame_sites[i][site_i]
                pos_j = frame_centers[j] if site_j == -1 else frame_sites[j][site_j]
                
                r_vec = mic_vector(pos_i, pos_j, box_dim)
                r = np.linalg.norm(r_vec)
                bond_dists[idx].append(r)
                
            # Extract angles
            for idx, a in enumerate(priors.get("angles", [])):
                i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
                site_i, site_j, site_k = a.get("site_i", -1), a.get("site_j", -1), a.get("site_k", -1)
                if i >= len(frame_centers) or j >= len(frame_centers) or k >= len(frame_centers): continue
                
                pos_i = frame_centers[i] if site_i == -1 else frame_sites[i][site_i]
                pos_j = frame_centers[j] if site_j == -1 else frame_sites[j][site_j]
                pos_k = frame_centers[k] if site_k == -1 else frame_sites[k][site_k]
                
                theta = get_angle(pos_i, pos_j, pos_k, box_dim)
                angle_dists[idx].append(theta)
                
            # Extract dihedrals
            for idx, d in enumerate(priors.get("dihedrals", [])):
                i, j, k, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
                site_i, site_j, site_k, site_l = d.get("site_i", -1), d.get("site_j", -1), d.get("site_k", -1), d.get("site_l", -1)
                if i >= len(frame_centers) or j >= len(frame_centers) or k >= len(frame_centers) or l >= len(frame_centers): continue
                
                pos_i = frame_centers[i] if site_i == -1 else frame_sites[i][site_i]
                pos_j = frame_centers[j] if site_j == -1 else frame_sites[j][site_j]
                pos_k = frame_centers[k] if site_k == -1 else frame_sites[k][site_k]
                pos_l = frame_centers[l] if site_l == -1 else frame_sites[l][site_l]
                
                phi = get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim)
                dihedral_dists[idx].append(phi)
                
    return bond_dists, angle_dists, dihedral_dists, first_frame_centers, np.array(first_frame_types), target_box_dim

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
    """Return raw counts, normalized density and bin centers on a uniform grid."""
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, density=False)
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
    """Integrate the exact force convention expected by ESPResSo tables."""
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

    energy = np.zeros_like(force)
    energy[anchor] = anchor_value
    sign = 1.0 if target_type == "angle" else -1.0

    for i in range(anchor, len(x) - 1):
        dx = x[i + 1] - x[i]
        energy[i + 1] = energy[i] + sign * 0.5 * (force[i] + force[i + 1]) * dx
    for i in range(anchor, 0, -1):
        dx = x[i] - x[i - 1]
        energy[i - 1] = energy[i] - sign * 0.5 * (force[i] + force[i - 1]) * dx

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


def table_from_potential(target_grid, potential, target_type, periodic=False, settings=None):
    """Create a table whose node energies are consistent with its capped forces."""
    settings = settings or DEFAULT_IBI_SETTINGS
    target_grid = np.asarray(target_grid, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if target_grid.shape != potential.shape:
        raise ValueError("Potential and target grid must have matching shapes")

    cfg = settings[target_type]
    force_max = float(cfg["force_max"])
    if periodic:
        potential = potential.copy()
        potential[-1] = potential[0]
        spline = CubicSpline(target_grid, potential, bc_type="periodic")
    else:
        spline = PchipInterpolator(target_grid, potential, extrapolate=False)

    derivative = spline(target_grid, 1)
    force = derivative if target_type == "angle" else -derivative
    force = np.clip(force, -force_max, force_max)

    if periodic and target_type != "angle":
        period = target_grid[-1] - target_grid[0]
        force -= np.trapezoid(force, target_grid) / period
        force[-1] = force[0]

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
    target_mask, target_first, target_last = statistical_support(target_counts, P_target, settings)
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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



def main():
    parser = argparse.ArgumentParser(description="Run IBI loop using exact tabulated targets.")
    parser.add_argument("--dataset", required=True, help="Path to the binary dataset file")
    parser.add_argument("--priors", required=True, help="Path to cg_priors.json")
    parser.add_argument("--config", required=True, help="Path to config.json (for run_cg_md)")
    parser.add_argument("--rb_info", required=True, help="Path to rigid_bodies_info.json (for run_cg_md)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of IBI iterations")
    parser.add_argument("--outdir", default="ibi_priors", help="Output directory for potentials")
    parser.add_argument("--ibi_config", default=None, help="Optional JSON with support, extrapolation and MD settings")
    default_pypresso = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "espresso", "build", "pypresso")
    )
    parser.add_argument("--pypresso", type=str, default=default_pypresso, help="Path to the ESPResSo pypresso executable")
    args = parser.parse_args()
    settings = load_ibi_settings(args.ibi_config)
    
    print("[INFO] =========================================")
    print(f"[INFO] Starting Iterative Boltzmann Inversion")
    print(f"[INFO] Iterations: {args.iterations}")
    if args.ibi_config:
        print(f"[INFO] IBI/extrapolation config: {args.ibi_config}")
    print(f"[INFO] Bond table domain: {settings['bond']['table_min']} .. {settings['bond']['table_max']} nm")
    print("[INFO] =========================================\n")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    with open(args.priors, "r") as f:
        priors_data = json.load(f)
        
    print(f"[INFO] Reading target dataset: {args.dataset}")
    bond_dists, angle_dists, dihedral_dists, first_frame_centers, types, target_box_dim = read_dataset_distributions(args.dataset, priors_data)
    
    # Check dataset limitsial positions for ESPResSo
    initial_pos_file = "_tmp_initial_pos.npy"
    np.save(initial_pos_file, np.array(first_frame_centers))
    
    # We will store active IBI potentials
    ibi_tables = {} # type -> idx -> (x, V, F, P_target)
    
    # ---------------------------------------------------------
    # STEP 1: Direct Boltzmann Inversion (DBI)
    # ---------------------------------------------------------
    print("[INFO] Performing Direct Boltzmann Inversion (DBI) to get V_0...")
    
    ibi_tables["bonds"] = {}
    ibi_tables["angles"] = {}
    ibi_tables["dihedrals"] = {}
    
    # Process Bonds
    pooled_bonds = {}
    for idx, b in enumerate(priors_data.get("bonds", [])):
        b_type = b.get("type", "unknown")
        if b_type in ["ibi", "dbi"]:
            name = b.get("name", f"idx_{idx}")
            if name not in pooled_bonds: pooled_bonds[name] = {"dists": [], "type": b_type}
            pooled_bonds[name]["dists"].extend(bond_dists[idx])
            
    for name, pool in pooled_bonds.items():
        if len(pool["dists"]) == 0: continue
        bond_cfg = settings["bond"]
        bins = np.linspace(bond_cfg["hist_min"], bond_cfg["hist_max"], int(bond_cfg["hist_edges"]))
        target_grid = np.linspace(bond_cfg["table_min"], bond_cfg["table_max"], int(bond_cfg["table_points"]))
        r, V_0, F_0, P_target, hist_x, target_counts, support = calculate_dbi_potential(
            pool["dists"], bins, target_grid, jacobian_type="bond", settings=settings
        )
        print(f"[INFO] Bond {name}: sampled support {support[0]:.4f} .. {support[1]:.4f} nm; table extends to {target_grid[-1]:.2f} nm")
        filename = f"{args.outdir}/bond_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["bonds"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "target_counts": target_counts, "hist_x": hist_x, "bins": bins, "support": support}

    for idx, b in enumerate(priors_data.get("bonds", [])):
        if b.get("type") in ["ibi", "dbi"]:
            name = b.get("name", f"idx_{idx}")
            b["type"] = "tabulated"
            b["file"] = f"{args.outdir}/bond_tabulated_{name}.dat"
            b["min"] = float(settings["bond"]["table_min"])
            b["max"] = float(settings["bond"]["table_max"])

    # Process Angles
    pooled_angles = {}
    for idx, a in enumerate(priors_data.get("angles", [])):
        a_type = a.get("type", "harmonic")
        if a_type in ["ibi", "dbi"]:
            name = a.get("name", f"idx_{idx}")
            if name not in pooled_angles: pooled_angles[name] = {"dists": [], "type": a_type}
            pooled_angles[name]["dists"].extend(angle_dists[idx])
            
    for name, pool in pooled_angles.items():
        if len(pool["dists"]) == 0: continue
        angle_cfg = settings["angle"]
        bins = np.linspace(0.0, np.pi, int(angle_cfg["hist_edges"]))
        target_grid = np.linspace(0.0, np.pi, int(angle_cfg["table_points"]))
        r, V_0, F_0, P_target, hist_x, target_counts, support = calculate_dbi_potential(
            pool["dists"], bins, target_grid, jacobian_type="angle", periodic=False, settings=settings
        )


        filename = f"{args.outdir}/angle_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["angles"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "target_counts": target_counts, "hist_x": hist_x, "bins": bins, "support": support}

    for idx, a in enumerate(priors_data.get("angles", [])):
        if a.get("type") in ["ibi", "dbi"]:
            name = a.get("name", f"idx_{idx}")
            a["type"] = "tabulated"
            a["file"] = f"{args.outdir}/angle_tabulated_{name}.dat"
            a["min"] = 0.0
            a["max"] = np.pi

    # Process Dihedrals
    pooled_dihedrals = {}
    for idx, d in enumerate(priors_data.get("dihedrals", [])):
        d_type = d.get("type", "cosine")
        if d_type in ["ibi", "dbi"]:
            name = d.get("name", f"idx_{idx}")
            if name not in pooled_dihedrals: pooled_dihedrals[name] = {"dists": [], "type": d_type}
            pooled_dihedrals[name]["dists"].extend(dihedral_dists[idx])
            
    for name, pool in pooled_dihedrals.items():
        if len(pool["dists"]) == 0: continue
        bins = np.linspace(0.0, 2 * np.pi, int(settings["dihedral"]["hist_edges"]))
        target_values = np.array(pool["dists"])
        target_values = np.where(target_values < 0, target_values + 2 * np.pi, target_values)
        target_grid = np.linspace(0.0, 2.0 * np.pi, int(settings["dihedral"]["table_points"]))
        r, V_0, F_0, P_target, hist_x, target_counts, support = calculate_dbi_potential(
            target_values, bins, target_grid, jacobian_type="dihedral", periodic=True, settings=settings
        )
        filename = f"{args.outdir}/dihedral_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["dihedrals"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "target_counts": target_counts, "hist_x": hist_x, "bins": bins, "support": support}

    for idx, d in enumerate(priors_data.get("dihedrals", [])):
        if d.get("type") in ["ibi", "dbi"]:
            name = d.get("name", f"idx_{idx}")
            d["type"] = "tabulated"
            d["file"] = f"{args.outdir}/dihedral_tabulated_{name}.dat"
            d["min"] = 0.0
            d["max"] = 2.0 * np.pi
            
    # Save JSON in DBI-only mode too
    tmp_priors = "cg_priors_tmp_ibi.json"
    with open(tmp_priors, "w") as f:
        json.dump(priors_data, f, indent=4)
        
    if args.iterations == 0:
        final_out_priors = f"{args.outdir}/cg_priors_final.json"
        with open(final_out_priors, "w") as f:
            json.dump(priors_data, f, indent=4)
        print(f"[SUCCESS] DBI-only priors saved to {final_out_priors}")
        if os.path.exists(tmp_priors):
            os.remove(tmp_priors)
        return
        
    # ---------------------------------------------------------
    # STEP 2: Iterative Boltzmann Inversion (IBI)
    # ---------------------------------------------------------
    # Initial modified priors are already saved above
        
    script_name = "_tmp_ibi_md.py"
    traj_name = "_tmp_traj.npz"
    
    for it in range(1, args.iterations + 1):
        print(f"\n[INFO] --- IBI Iteration {it}/{args.iterations} ---")
        
        # Use subprocess to run run_cg_md.py
        # Since we are not providing --model, it acts as a priors-only MD.
        run_cg_md_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation", "run_cg_md.py")
        
        # Run equilibrate.py to generate a valid starting configuration for MD
        equilibrate_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation", "equilibrate.py")
        chk_name = "_tmp_equilibrated.npz"
        
        cmd_equil = [
            args.pypresso, equilibrate_script,
            "--priors_only",
            "--config", args.config,
            "--priors", tmp_priors,
            "--rb_info", args.rb_info,
            "--dataset", args.dataset,
            "--out_checkpoint", chk_name,
            "--steps_sd", str(settings["simulation"]["equilibration_sd_steps"]),
            "--steps_md", str(settings["simulation"]["equilibration_md_steps"]),
            "--dt", str(settings["simulation"]["dt"])
        ]
        
        print(f"[INFO] Running equilibration via equilibrate.py...")
        res = subprocess.run(cmd_equil, capture_output=True, text=True)
        if res.returncode != 0:
            print("[ERROR] ESPResSo equilibration failed!")
            print(res.stdout)
            print(res.stderr)
            sys.exit(1)
        
        cmd = [
            args.pypresso, run_cg_md_script,
            "--config", args.config,
            "--priors", tmp_priors,
            "--rb_info", args.rb_info,
            "--dataset", args.dataset,
            "--checkpoint", chk_name,
            "--steps", str(settings["simulation"]["steps"]),
            "--log_interval", str(settings["simulation"]["log_interval"]),
            "--dt", str(settings["simulation"]["dt"]),
            "--out_traj", traj_name,
            "--no_log"
        ]
        
        print(f"[INFO] Running ESPResSo MD simulation via run_cg_md.py...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("[ERROR] ESPResSo simulation failed!")
            print(res.stdout)
            print(res.stderr)
            sys.exit(1)
            
        print("[INFO] MD completed. Analyzing trajectory...")
        try:
            trajectory = np.load(traj_name)
            com_positions = trajectory["com"]
            site_positions = trajectory["sites"]
            site_molecule = trajectory["site_molecule"].astype(int)
            site_index = trajectory["site_index"].astype(int)
            box_dim = np.asarray(trajectory["box"], dtype=float)
        except Exception as e:
            print(f"[ERROR] Could not load site-aware trajectory: {e}")
            sys.exit(1)

        site_lookup = {
            (int(mol), int(site)): idx
            for idx, (mol, site) in enumerate(zip(site_molecule, site_index))
        }

        def coordinate(frame_idx, mol_idx, requested_site):
            if requested_site == -1:
                return com_positions[frame_idx, mol_idx]
            key = (int(mol_idx), int(requested_site))
            if key not in site_lookup:
                raise KeyError(f"Missing virtual site {key} in IBI trajectory")
            return site_positions[frame_idx, site_lookup[key]]

        def dihedral_angle(p0, p1, p2, p3):
            b0 = -1.0 * mic_vector(p0, p1, box_dim)
            b1 = mic_vector(p1, p2, box_dim)
            b2 = mic_vector(p2, p3, box_dim)
            b1 /= np.linalg.norm(b1)
            v = b0 - np.dot(b0, b1)*b1
            w = b2 - np.dot(b2, b1)*b1
            x = np.dot(v, w)
            y = np.dot(np.cross(b1, v), w)
            return np.arctan2(y, x)

        sim_bond_dists = {name: [] for name in ibi_tables.get("bonds", {}).keys()}
        sim_angle_dists = {name: [] for name in ibi_tables.get("angles", {}).keys()}
        sim_dihedral_dists = {name: [] for name in ibi_tables.get("dihedrals", {}).keys()}
        
        for frame_idx in range(com_positions.shape[0]):
            # Bonds
            for idx, b in enumerate(priors_data.get("bonds", [])):
                name = b.get("name", f"idx_{idx}")
                if name in sim_bond_dists:
                    i, j = b["mol_i"], b["mol_j"]
                    site_i, site_j = b.get("site_i", -1), b.get("site_j", -1)
                    pos_i = coordinate(frame_idx, i, site_i)
                    pos_j = coordinate(frame_idx, j, site_j)
                    r = np.linalg.norm(mic_vector(pos_i, pos_j, box_dim))
                    sim_bond_dists[name].append(r)
            
            # Angles
            for idx, a in enumerate(priors_data.get("angles", [])):
                name = a.get("name", f"idx_{idx}")
                if name in sim_angle_dists:
                    i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
                    site_i = a.get("site_i", -1)
                    site_j = a.get("site_j", -1)
                    site_k = a.get("site_k", -1)
                    pos_i = coordinate(frame_idx, i, site_i)
                    pos_j = coordinate(frame_idx, j, site_j)
                    pos_k = coordinate(frame_idx, k, site_k)
                    v1 = mic_vector(pos_j, pos_i, box_dim)
                    v2 = mic_vector(pos_j, pos_k, box_dim)
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-6 and n2 > 1e-6:
                        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                        sim_angle_dists[name].append(np.arccos(cos_theta))

            # Dihedrals
            for idx, d in enumerate(priors_data.get("dihedrals", [])):
                name = d.get("name", f"idx_{idx}")
                if name in sim_dihedral_dists:
                    i, j, k, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
                    pos_i = coordinate(frame_idx, i, d.get("site_i", -1))
                    pos_j = coordinate(frame_idx, j, d.get("site_j", -1))
                    pos_k = coordinate(frame_idx, k, d.get("site_k", -1))
                    pos_l = coordinate(frame_idx, l, d.get("site_l", -1))
                    phi = dihedral_angle(pos_i, pos_j, pos_k, pos_l)
                    sim_dihedral_dists[name].append(np.mod(phi, 2.0 * np.pi))
                
        # Update tabulated bonds
        print("[INFO] Updating tabulated bonds...")
        for name, table in ibi_tables.get("bonds", {}).items():
            sim_dists = np.asarray(sim_bond_dists[name], dtype=float)
            outside = (sim_dists < table["bins"][0]) | (sim_dists > table["bins"][-1])
            if np.any(outside):
                print(
                    f"[WARN] Bond {name}: {np.count_nonzero(outside)}/{sim_dists.size} "
                    f"samples outside the target histogram range; max={np.max(sim_dists):.4f} nm"
                )
            safety_limit = 0.95 * float(table["x"][-1])
            if np.max(sim_dists) >= safety_limit:
                raise RuntimeError(
                    f"Bond {name} reached {np.max(sim_dists):.4f} nm, too close to "
                    f"the table limit {table['x'][-1]:.4f} nm"
                )
            sim_counts, hist_sim, _ = histogram_density(sim_dists, table["bins"])
            _, V_next, F_next = update_ibi_potential(
                table["V"], hist_sim, table["P"], table["hist_x"], table["x"],
                table["target_counts"], sim_counts,
                periodic=False, target_type="bond", settings=settings,
                previous_force=table["F"]
            )
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/bond_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated angles
        print("[INFO] Updating tabulated angles...")
        for name, table in ibi_tables.get("angles", {}).items():
            sim_dists = sim_angle_dists[name]
            sim_counts, hist_sim, _ = histogram_density(sim_dists, table["bins"])
            _, V_next, F_next = update_ibi_potential(
                table["V"], hist_sim, table["P"], table["hist_x"], table["x"],
                table["target_counts"], sim_counts,
                periodic=False, target_type="angle", settings=settings,
                previous_force=table["F"]
            )
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/angle_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated dihedrals
        print("[INFO] Updating tabulated dihedrals...")
        for name, table in ibi_tables.get("dihedrals", {}).items():
            sim_dists = sim_dihedral_dists[name]
            sim_dists = np.mod(np.asarray(sim_dists), 2.0 * np.pi)
            sim_counts, hist_sim, _ = histogram_density(sim_dists, table["bins"])
            _, V_next, F_next = update_ibi_potential(
                table["V"], hist_sim, table["P"], table["hist_x"], table["x"],
                table["target_counts"], sim_counts,
                periodic=True, target_type="dihedral", settings=settings,
                previous_force=table["F"]
            )
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/dihedral_tabulated_{name}.dat", table["x"], V_next, F_next)
            
    print(f"\n[SUCCESS] IBI Converged after {args.iterations} iterations.")
    
    # Save updated priors back to the final file
    final_out_priors = f"{args.outdir}/cg_priors_final.json"
    print(f"[INFO] Saving updated priors with tabulated paths to {final_out_priors}")
    with open(final_out_priors, "w") as f:
        json.dump(priors_data, f, indent=4)
        
    # Cleanup temp files
    if os.path.exists(script_name): os.remove(script_name)
    if os.path.exists(traj_name): os.remove(traj_name)
    if os.path.exists(tmp_priors): os.remove(tmp_priors)

if __name__ == "__main__":
    main()
