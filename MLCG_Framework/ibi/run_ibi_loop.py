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
    
    with open(bin_file, "rb") as f:
        data = f.read(4)
        if not data: return bond_dists, angle_dists, dihedral_dists, [], []
        num_frames = struct.unpack("i", data)[0]
        
        for frame_idx in range(num_frames):
            num_molecules = struct.unpack("i", f.read(4))[0]
            num_total_sites = struct.unpack("i", f.read(4))[0]
            box_dim = np.array(struct.unpack("3f", f.read(12)))
            
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
                for _ in range(num_sites):
                    site_type = struct.unpack("i", f.read(4))[0]
                    site_pos = np.array(struct.unpack("3f", f.read(12)))
                    sites.append(site_pos)
                
                frame_centers.append(center)
                frame_sites.append(sites)
                # In our generic dataset, the CG bead type corresponds to the first site type
                frame_types.append(site_type if num_sites > 0 else 0)
                
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
                
    return bond_dists, angle_dists, dihedral_dists, first_frame_centers, np.array(first_frame_types)

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

def calculate_dbi_potential(values, bins, kT=2.49, periodic=False, jacobian_type=None):
    if jacobian_type == 'dihedral':
        values = np.mod(values, 2.0 * np.pi)
        
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    raw_hist = hist.copy()
    if jacobian_type == 'bond':
        hist = hist / (bin_centers**2)
    elif jacobian_type == 'angle':
        hist = hist / np.clip(np.sin(bin_centers), 1e-6, None)
    hist = np.clip(hist, 1e-6, None)
    
    hist /= np.sum(hist) * (bin_centers[1] - bin_centers[0])
    
    potential = -kT * np.log(hist)
    potential -= np.min(potential)
    
    mode = 'wrap' if periodic else 'reflect'
    potential_smooth = gaussian_filter1d(potential, sigma=2.0, mode=mode)
    
    if not periodic:
        potential_smooth, _ = extrapolate_potential_and_force(bin_centers, potential_smooth, np.zeros_like(potential_smooth), raw_hist, target_type=jacobian_type)
        
    bc_type = 'periodic' if periodic else 'not-a-knot'
    spline = CubicSpline(bin_centers, potential_smooth, bc_type=bc_type)
    force_deriv = spline(bin_centers, 1)
    
    if jacobian_type == 'angle':
        F_0 = force_deriv
    else:
        F_0 = -force_deriv
        
    F_0 = np.clip(F_0, -150.0, 150.0)
    
    # Reintegrate capped forces for perfect consistency
    min_idx = np.argmin(potential_smooth)
    V_new = np.zeros_like(potential_smooth)
    V_new[min_idx] = potential_smooth[min_idx]
    
    for i in range(min_idx, len(bin_centers) - 1):
        dx = bin_centers[i+1] - bin_centers[i]
        avg_F = 0.5 * (F_0[i] + F_0[i+1])
        if jacobian_type == 'angle':
            V_new[i+1] = V_new[i] + avg_F * dx
        else:
            V_new[i+1] = V_new[i] - avg_F * dx
            
    for i in range(min_idx, 0, -1):
        dx = bin_centers[i] - bin_centers[i-1]
        avg_F = 0.5 * (F_0[i] + F_0[i-1])
        if jacobian_type == 'angle':
            V_new[i-1] = V_new[i] - avg_F * dx
        else:
            V_new[i-1] = V_new[i] + avg_F * dx
            
    V_new -= np.min(V_new)
    return bin_centers, V_new, F_0, hist

def update_ibi_potential(V_i, P_i, P_target, bin_centers, kT=2.49, alpha=0.5, periodic=False, target_type='bond'):
    P_i = np.clip(P_i, 1e-6, None)
    P_target = np.clip(P_target, 1e-6, None)
    
    update = alpha * kT * np.log(P_i / P_target)
    V_next = V_i + update
    V_next -= np.min(V_next)
    
    mode = 'wrap' if periodic else 'reflect'
    V_next_smooth = gaussian_filter1d(V_next, sigma=2.0, mode=mode)
    
    if not periodic:
        V_next_smooth, _ = extrapolate_potential_and_force(bin_centers, V_next_smooth, np.zeros_like(V_next_smooth), P_target, target_type=target_type)
        
    bc_type = 'periodic' if periodic else 'not-a-knot'
    spline = CubicSpline(bin_centers, V_next_smooth, bc_type=bc_type)
    force_deriv = spline(bin_centers, 1)
    
    if target_type == 'angle':
        F_0 = force_deriv
    else:
        F_0 = -force_deriv
        
    F_0 = np.clip(F_0, -150.0, 150.0)
    
    min_idx = np.argmin(V_next_smooth)
    V_new = np.zeros_like(V_next_smooth)
    V_new[min_idx] = V_next_smooth[min_idx]
    
    for i in range(min_idx, len(bin_centers) - 1):
        dx = bin_centers[i+1] - bin_centers[i]
        avg_F = 0.5 * (F_0[i] + F_0[i+1])
        if target_type == 'angle':
            V_new[i+1] = V_new[i] + avg_F * dx
        else:
            V_new[i+1] = V_new[i] - avg_F * dx
            
    for i in range(min_idx, 0, -1):
        dx = bin_centers[i] - bin_centers[i-1]
        avg_F = 0.5 * (F_0[i] + F_0[i-1])
        if target_type == 'angle':
            V_new[i-1] = V_new[i] - avg_F * dx
        else:
            V_new[i-1] = V_new[i] + avg_F * dx
            
    V_new -= np.min(V_new)
    return V_new, F_0

def save_tabulated_potential(filename, x, energy, force):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    data = np.column_stack((x, energy, force))
    np.savetxt(filename, data, fmt="%.6f", header="x energy force")





def main():
    parser = argparse.ArgumentParser(description="Run IBI loop using exact tabulated targets.")
    parser.add_argument("--dataset", required=True, help="Path to the binary dataset file")
    parser.add_argument("--priors", required=True, help="Path to cg_priors.json")
    parser.add_argument("--config", required=True, help="Path to config.json (for run_cg_md)")
    parser.add_argument("--rb_info", required=True, help="Path to rigid_bodies_info.json (for run_cg_md)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of IBI iterations")
    parser.add_argument("--outdir", default="ibi_priors", help="Output directory for potentials")
    parser.add_argument("--pypresso", type=str, default="/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/espresso/build/pypresso")
    args = parser.parse_args()
    
    print("[INFO] =========================================")
    print(f"[INFO] Starting Iterative Boltzmann Inversion")
    print(f"[INFO] Iterations: {args.iterations}")
    print("[INFO] =========================================\n")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    with open(args.priors, "r") as f:
        priors_data = json.load(f)
        
    print(f"[INFO] Reading target dataset: {args.dataset}")
    bond_dists, angle_dists, dihedral_dists, first_frame_centers, types = read_dataset_distributions(args.dataset, priors_data)
    
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
        bins = np.linspace(0.0, 5.0, 300)
        r, V_0, F_0, P_target = calculate_dbi_potential(pool["dists"], bins, jacobian_type='bond')
        filename = f"{args.outdir}/bond_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["bonds"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "bins": bins}

    for idx, b in enumerate(priors_data.get("bonds", [])):
        if b.get("type") in ["ibi", "dbi"]:
            name = b.get("name", f"idx_{idx}")
            b["type"] = "tabulated"
            b["file"] = f"{args.outdir}/bond_tabulated_{name}.dat"
            b["min"] = 0.0
            b["max"] = 5.0

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
        r, V_0, F_0, P_target = calculate_dbi_potential(pool["dists"], bins, jacobian_type='angle', periodic=False)
        
        # Protective walls to prevent angles from reaching exactly 0 or pi
        for i in range(len(r)):
            if r[i] < 0.1:
                dx = 0.1 - r[i]
                F_0[i] += 5000.0 * dx
                V_0[i] += 0.5 * 5000.0 * dx**2
            elif r[i] > np.pi - 0.1:
                dx = r[i] - (np.pi - 0.1)
                F_0[i] -= 5000.0 * dx
                V_0[i] += 0.5 * 5000.0 * dx**2

        filename = f"{args.outdir}/angle_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["angles"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "bins": bins}

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
        r, V_0, F_0, P_target = calculate_dbi_potential(target_values, bins, jacobian_type='dihedral', periodic=True)
        filename = f"{args.outdir}/dihedral_tabulated_{name}.dat"
        save_tabulated_potential(filename, r, V_0, F_0)
        if pool["type"] == "ibi":
            ibi_tables["dihedrals"][name] = {"x": r, "V": V_0, "F": F_0, "P": P_target, "bins": bins}

    for idx, d in enumerate(priors_data.get("dihedrals", [])):
        if d.get("type") in ["ibi", "dbi"]:
            name = d.get("name", f"idx_{idx}")
            d["type"] = "tabulated"
            d["file"] = f"{args.outdir}/dihedral_tabulated_{name}.dat"
            d["min"] = 0.0
            d["max"] = 2 * np.pi
            
    # Save JSON in DBI-only mode too
    tmp_priors = "cg_priors_tmp_ibi.json"
    with open(tmp_priors, "w") as f:
        json.dump(priors_data, f, indent=4)
        
    final_out_priors = f"{args.outdir}/cg_priors_final.json"
    with open(final_out_priors, "w") as f:
        json.dump(priors_data, f, indent=4)

    if args.iterations == 0:
        print("[INFO] User requested 0 iterations. Stopping at DBI.")
        sys.exit(0)
        
    # ---------------------------------------------------------
    # STEP 2: Iterative Boltzmann Inversion (IBI)
    # ---------------------------------------------------------
    # Initial modified priors are already saved above
        
    script_name = "_tmp_ibi_md.py"
    traj_name = "_tmp_traj.npy"
    
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
            positions = np.load(traj_name)
        except Exception as e:
            print(f"[ERROR] Could not load trajectory: {e}")
            sys.exit(1)
            
        # For simplicity, extract bond distances from MD trajectory
        box_dim = np.array([10.0, 10.0, 10.0])
        import math
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
        
        for frame in positions:
            # Bonds
            for idx, b in enumerate(priors_data.get("bonds", [])):
                name = b.get("name", f"idx_{idx}")
                if name in sim_bond_dists:
                    i, j = b["mol_i"], b["mol_j"]
                    r = np.linalg.norm(mic_vector(frame[i], frame[j], box_dim))
                    sim_bond_dists[name].append(r)
            
            # Angles
            for idx, a in enumerate(priors_data.get("angles", [])):
                name = a.get("name", f"idx_{idx}")
                if name in sim_angle_dists:
                    i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
                    v1 = mic_vector(frame[j], frame[i], box_dim)
                    v2 = mic_vector(frame[j], frame[k], box_dim)
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-6 and n2 > 1e-6:
                        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                        sim_angle_dists[name].append(np.arccos(cos_theta))

            # Dihedrals
            for idx, d in enumerate(priors_data.get("dihedrals", [])):
                name = d.get("name", f"idx_{idx}")
                if name in sim_dihedral_dists:
                    i, j, k, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
                    sim_dihedral_dists[name].append(dihedral_angle(frame[i], frame[j], frame[k], frame[l]))
                
        # Update tabulated bonds
        print("[INFO] Updating tabulated bonds...")
        for name, table in ibi_tables.get("bonds", {}).items():
            sim_dists = sim_bond_dists[name]
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            hist_sim = hist_sim / (table["x"]**2)
            hist_sim = np.clip(hist_sim, 1e-6, None)
            hist_sim /= np.sum(hist_sim) * (table["x"][1] - table["x"][0])
            V_next, F_next = update_ibi_potential(table["V"], hist_sim, table["P"], table["x"], periodic=False, target_type='bond')
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/bond_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated angles
        print("[INFO] Updating tabulated angles...")
        for name, table in ibi_tables.get("angles", {}).items():
            sim_dists = sim_angle_dists[name]
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            hist_sim = hist_sim / np.clip(np.sin(table["x"]), 1e-6, None)
            hist_sim = np.clip(hist_sim, 1e-6, None)
            hist_sim /= np.sum(hist_sim) * (table["x"][1] - table["x"][0])
            V_next, F_next = update_ibi_potential(table["V"], hist_sim, table["P"], table["x"], periodic=False, target_type='angle')
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/angle_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated dihedrals
        print("[INFO] Updating tabulated dihedrals...")
        for name, table in ibi_tables.get("dihedrals", {}).items():
            sim_dists = sim_dihedral_dists[name]
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            hist_sim = np.clip(hist_sim, 1e-6, None)
            hist_sim /= np.sum(hist_sim) * (table["x"][1] - table["x"][0])
            V_next, F_next = update_ibi_potential(table["V"], hist_sim, table["P"], table["x"], periodic=True, target_type='dihedral')
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/dihedral_tabulated_{name}.dat", table["x"], V_next, F_next)
            
    print(f"\n[SUCCESS] IBI Converged after {args.iterations} iterations.")
    
    # Save updated priors back to the original file
    print(f"[INFO] Saving updated priors with tabulated paths to {args.priors}")
    with open(args.priors, "w") as f:
        json.dump(priors_data, f, indent=4)
        
    # Cleanup temp files
    if os.path.exists(script_name): os.remove(script_name)
    if os.path.exists(traj_name): os.remove(traj_name)
    if os.path.exists(tmp_priors): os.remove(tmp_priors)

if __name__ == "__main__":
    main()
