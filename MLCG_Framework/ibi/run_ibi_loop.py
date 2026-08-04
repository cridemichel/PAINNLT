import os
import sys
import numpy as np
import argparse
import json
import struct
import subprocess
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "numpy",
#     "scipy"
# ]
# ///

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

def extrapolate_potential_and_force(x, V, F, hist, target_type='bond'):
    valid_idx = np.where(hist > 1e-5)[0]
    if len(valid_idx) == 0:
        return V, F
    first, last = valid_idx[0], valid_idx[-1]
    
    # Left extrapolation (repulsive core)
    base_left_force = F[first]
    x_left = x[first]
    V_left = V[first]
    
    if target_type == 'bond':
        # WCA Extrapolation for Bonds: F(x) = F_min * (x_min/x)^13
        if base_left_force > 0 and x_left > 0:
            for i in range(first - 1, -1, -1):
                if x[i] <= 0:
                    F[i] = F[i+1]
                    V[i] = V[i+1]
                    continue
                ratio = min(x_left / x[i], 100.0) # Prevent catastrophic overflow
                fi = base_left_force * (ratio**13)
                ui = V_left + (base_left_force * x_left / 12.0) * ((ratio**12) - 1.0)
                # Cap extremely large forces to prevent table precision issues
                F[i] = min(fi, 150.0)
                V[i] = ui
        else:
            # Fallback
            for i in range(first - 1, -1, -1):
                dx = x[i] - x_left
                F[i] = max(base_left_force, 10.0)
                V[i] = V_left - F[i] * dx
    else:
        # Constant Force for Angles / Dihedrals
        for i in range(first - 1, -1, -1):
            dx = x[i] - x_left
            F[i] = max(base_left_force, 10.0)
            V[i] = V_left - F[i] * dx
        
    # Right extrapolation (attractive tail / zero)
    base_right_force = F[last]
    x_right = x[last]
    V_right = V[last]
    
    for i in range(last + 1, len(x)):
        dx = x[i] - x_right
        F[i] = min(base_right_force, -10.0)
        V[i] = V_right - F[i] * dx
        
    return V, F

def enforce_consistency_and_cap(x, V_orig, F_orig, force_max=100.0):
    F_capped = np.clip(F_orig, -force_max, force_max)
    min_idx = np.argmin(V_orig)
    V_new = np.zeros_like(V_orig)
    V_new[min_idx] = V_orig[min_idx]
    
    for i in range(min_idx, len(x) - 1):
        dx = x[i+1] - x[i]
        avg_F = 0.5 * (F_capped[i] + F_capped[i+1])
        V_new[i+1] = V_new[i] - avg_F * dx
        
    for i in range(min_idx, 0, -1):
        dx = x[i] - x[i-1]
        avg_F = 0.5 * (F_capped[i] + F_capped[i-1])
        V_new[i-1] = V_new[i] + avg_F * dx
        
    V_new -= np.min(V_new)
    return V_new, F_capped

def normalize_density(hist, grid):
    """Return a clipped probability density normalized on *grid*."""
    hist = np.clip(np.asarray(hist, dtype=float), 1e-12, None)
    grid = np.asarray(grid, dtype=float)
    if hist.shape != grid.shape:
        raise ValueError(f"Histogram/grid shape mismatch: {hist.shape} vs {grid.shape}")
    norm = np.trapezoid(hist, grid)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize an empty or non-finite distribution")
    return hist / norm


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


def integrate_tabulated_force(x, force, target_type, reference_energy=None):
    """Integrate the exact force convention expected by ESPResSo tables.

    Distance/dihedral tables store -dU/dq; angle tables store +dU/dtheta.
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


def table_from_potential(target_grid, potential, target_type, periodic=False, force_max=150.0):
    """Create a table whose node energies are consistent with its capped forces."""
    target_grid = np.asarray(target_grid, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if target_grid.shape != potential.shape:
        raise ValueError("Potential and target grid must have matching shapes")

    if periodic:
        potential = potential.copy()
        potential[-1] = potential[0]
        spline = CubicSpline(target_grid, potential, bc_type="periodic")
    else:
        spline = CubicSpline(target_grid, potential, bc_type="not-a-knot")

    derivative = spline(target_grid, 1)
    force = derivative if target_type == "angle" else -derivative
    force = np.clip(force, -force_max, force_max)

    if periodic and target_type != "angle":
        # A periodic conservative force has zero integral over one period.
        period = target_grid[-1] - target_grid[0]
        force -= np.trapezoid(force, target_grid) / period
        force[-1] = force[0]

    energy = integrate_tabulated_force(target_grid, force, target_type, potential)
    if target_type == "angle":
        energy, force = add_angle_walls(target_grid, energy, force)
    elif periodic:
        energy[-1] = energy[0]
        force[-1] = force[0]
    return energy, force


def calculate_dbi_potential(values, bins, target_grid, kT=2.49, periodic=False, jacobian_type=None):
    values = np.asarray(values, dtype=float)
    if jacobian_type == "dihedral":
        values = np.mod(values, 2.0 * np.pi)

    raw_hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    target_hist = normalize_density(raw_hist, bin_centers)

    # Jacobians enter only the initial PMF. IBI updates compare the same raw
    # observable in target and simulated ensembles, so the Jacobian cancels.
    pmf_hist = target_hist.copy()
    if jacobian_type == "bond":
        pmf_hist /= np.clip(bin_centers**2, 1e-12, None)
    elif jacobian_type == "angle":
        pmf_hist /= np.clip(np.sin(bin_centers), 1e-6, None)
    pmf_hist = normalize_density(pmf_hist, bin_centers)

    potential_hist = -kT * np.log(np.clip(pmf_hist, 1e-12, None))
    potential_hist -= np.min(potential_hist)
    potential_hist = gaussian_filter1d(
        potential_hist,
        sigma=2.0,
        mode="wrap" if periodic else "reflect",
    )

    spline = make_spline(bin_centers, potential_hist, periodic=periodic)
    potential_grid = spline(target_grid)
    energy, force = table_from_potential(
        target_grid,
        potential_grid,
        target_type=jacobian_type,
        periodic=periodic,
    )
    return target_grid, energy, force, target_hist, bin_centers


def update_ibi_potential(
    V_i,
    P_i,
    P_target,
    bin_centers,
    target_grid,
    kT=2.49,
    alpha=0.5,
    periodic=False,
    target_type="bond",
):
    """Apply an IBI update on the histogram grid, then project it to the table grid."""
    V_i = np.asarray(V_i, dtype=float)
    target_grid = np.asarray(target_grid, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)
    if V_i.shape != target_grid.shape:
        raise ValueError(f"V_i/target_grid mismatch: {V_i.shape} vs {target_grid.shape}")

    P_i = normalize_density(P_i, bin_centers)
    P_target = normalize_density(P_target, bin_centers)
    delta_hist = alpha * kT * np.log(
        np.clip(P_i, 1e-12, None) / np.clip(P_target, 1e-12, None)
    )
    delta_hist = gaussian_filter1d(
        delta_hist,
        sigma=2.0,
        mode="wrap" if periodic else "reflect",
    )

    delta_spline = make_spline(bin_centers, delta_hist, periodic=periodic)
    potential_grid = V_i + delta_spline(target_grid)
    potential_grid -= np.min(potential_grid)
    potential_grid = gaussian_filter1d(
        potential_grid,
        sigma=2.0,
        mode="wrap" if periodic else "reflect",
    )
    if periodic:
        potential_grid[-1] = potential_grid[0]

    energy, force = table_from_potential(
        target_grid,
        potential_grid,
        target_type=target_type,
        periodic=periodic,
    )
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
    default_pypresso = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "espresso", "build", "pypresso")
    )
    parser.add_argument("--pypresso", type=str, default=default_pypresso, help="Path to the ESPResSo pypresso executable")
    args = parser.parse_args()
    
    print("[INFO] =========================================")
    print(f"[INFO] Starting Iterative Boltzmann Inversion")
    print(f"[INFO] Iterations: {args.iterations}")
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
        bins = np.linspace(0.01, 3.0, 300)
        target_grid = np.linspace(0.01, 3.0, 2001)
        r, V_0, F_0, P_target, hist_x = calculate_dbi_potential(pool["dists"], bins, target_grid, jacobian_type="bond")
        filename = f"{args.outdir}/bond_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["bonds"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "hist_x": hist_x, "bins": bins}

    for idx, b in enumerate(priors_data.get("bonds", [])):
        if b.get("type") in ["ibi", "dbi"]:
            name = b.get("name", f"idx_{idx}")
            b["type"] = "tabulated"
            b["file"] = f"{args.outdir}/bond_tabulated_{name}.dat"
            b["min"] = 0.01
            b["max"] = 3.0

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
        bins = np.linspace(0.0, np.pi, 300)
        target_grid = np.linspace(0.0, np.pi, 2001)
        r, V_0, F_0, P_target, hist_x = calculate_dbi_potential(pool["dists"], bins, target_grid, jacobian_type="angle", periodic=False)


        filename = f"{args.outdir}/angle_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["angles"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "hist_x": hist_x, "bins": bins}

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
        bins = np.linspace(0.0, 2 * np.pi, 300)
        target_values = np.array(pool["dists"])
        target_values = np.where(target_values < 0, target_values + 2 * np.pi, target_values)
        target_grid = np.linspace(0.0, 2.0 * np.pi, 2001)
        r, V_0, F_0, P_target, hist_x = calculate_dbi_potential(target_values, bins, target_grid, jacobian_type="dihedral", periodic=True)
        filename = f"{args.outdir}/dihedral_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["dihedrals"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "hist_x": hist_x, "bins": bins}

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
            "--steps_sd", "5000",
            "--steps_md", "2000"
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
            "--steps", "10000",
            "--log_interval", "10",
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
            sim_dists = sim_bond_dists[name]
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            hist_sim = normalize_density(hist_sim, table["hist_x"])
            _, V_next, F_next = update_ibi_potential(
                table["V"], hist_sim, table["P"], table["hist_x"], table["x"],
                periodic=False, target_type="bond"
            )
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/bond_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated angles
        print("[INFO] Updating tabulated angles...")
        for name, table in ibi_tables.get("angles", {}).items():
            sim_dists = sim_angle_dists[name]
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            hist_sim = normalize_density(hist_sim, table["hist_x"])
            _, V_next, F_next = update_ibi_potential(
                table["V"], hist_sim, table["P"], table["hist_x"], table["x"],
                periodic=False, target_type="angle"
            )
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/angle_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated dihedrals
        print("[INFO] Updating tabulated dihedrals...")
        for name, table in ibi_tables.get("dihedrals", {}).items():
            sim_dists = sim_dihedral_dists[name]
            sim_dists = np.mod(np.asarray(sim_dists), 2.0 * np.pi)
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            hist_sim = normalize_density(hist_sim, table["hist_x"])
            _, V_next, F_next = update_ibi_potential(
                table["V"], hist_sim, table["P"], table["hist_x"], table["x"],
                periodic=True, target_type="dihedral"
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
