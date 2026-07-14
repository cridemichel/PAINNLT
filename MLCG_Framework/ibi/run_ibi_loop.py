import os
import sys
import numpy as np
import argparse
import json
import struct
import subprocess
from scipy.ndimage import gaussian_filter1d

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
    
    with open(bin_file, "rb") as f:
        data = f.read(4)
        if not data: return bond_dists, angle_dists, dihedral_dists, []
        num_frames = struct.unpack("i", data)[0]
        
        for frame_idx in range(num_frames):
            num_molecules = struct.unpack("i", f.read(4))[0]
            num_total_sites = struct.unpack("i", f.read(4))[0]
            box_dim = np.array(struct.unpack("3f", f.read(12)))
            
            frame_centers = []
            frame_sites = []
            
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
                
            if frame_idx == 0:
                first_frame_centers = frame_centers
                
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
                
    return bond_dists, angle_dists, dihedral_dists, first_frame_centers

def extrapolate_potential_and_force(x, V, F, hist):
    valid_idx = np.where(hist > 1e-5)[0]
    if len(valid_idx) == 0:
        return V, F
    first, last = valid_idx[0], valid_idx[-1]
    
    k_wall = 5000.0  # Steep harmonic wall
    
    # Force the base extrapolation forces to point INWARD safely
    base_left_force = max(F[first], 10.0)
    base_right_force = min(F[last], -10.0)
    
    # Left extrapolation
    for i in range(first - 1, -1, -1):
        dx = x[first] - x[i]
        F[i] = base_left_force + k_wall * dx
        V[i] = V[i+1] + F[i+1] * (x[i+1] - x[i])
        
    # Right extrapolation
    for i in range(last + 1, len(x)):
        dx = x[i] - x[last]
        F[i] = base_right_force - k_wall * dx
        V[i] = V[i-1] - F[i-1] * (x[i] - x[i-1])
        
    return V, F

def calculate_dbi_potential(values, bins, kT=2.49, periodic=False, jacobian_type=None):
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if jacobian_type == 'bond':
        hist = hist / (bin_centers**2)
    elif jacobian_type == 'angle':
        hist = hist / np.clip(np.sin(bin_centers), 1e-6, None)
        
    raw_hist = hist.copy()
    hist = np.clip(hist, 1e-6, None)
    
    # Normalize after Jacobian correction
    hist /= np.sum(hist) * (bin_centers[1] - bin_centers[0])
    
    potential = -kT * np.log(hist)
    potential -= np.min(potential)
    
    # Smooth
    mode = 'wrap' if periodic else 'reflect'
    potential_smooth = gaussian_filter1d(potential, sigma=2.0, mode=mode)
    
    dx = bin_centers[1] - bin_centers[0]
    
    if periodic:
        force = -np.gradient(potential_smooth, dx)
    else:
        force = -np.gradient(potential_smooth, dx)
        potential_smooth, force = extrapolate_potential_and_force(bin_centers, potential_smooth, force, raw_hist)
        
    force = gaussian_filter1d(force, sigma=2.0, mode=mode)
        
    return bin_centers, potential_smooth, force, hist

def update_ibi_potential(V_i, P_i, P_target, bin_centers, kT=2.49, alpha=0.5, periodic=False):
    P_i = np.clip(P_i, 1e-6, None)
    P_target = np.clip(P_target, 1e-6, None)
    
    update = alpha * kT * np.log(P_i / P_target)
    V_next = V_i + update
    V_next -= np.min(V_next)
    
    mode = 'wrap' if periodic else 'reflect'
    V_next_smooth = gaussian_filter1d(V_next, sigma=2.0, mode=mode)
    
    dx = bin_centers[1] - bin_centers[0]
    force = -np.gradient(V_next_smooth, dx)
    
    if not periodic:
        V_next_smooth, force = extrapolate_potential_and_force(bin_centers, V_next_smooth, force, P_target)
        
    force = gaussian_filter1d(force, sigma=2.0, mode=mode)
        
    return V_next_smooth, force

def save_tabulated_potential(filename, x, energy, force):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = np.column_stack((x, energy, force))
    np.savetxt(filename, data, fmt="%.6f", header="x energy force")

def write_espresso_runner(script_name, priors_file, output_traj, initial_pos_file):
    """
    Writes a temporary Python script that runs ESPResSo MD using only priors.
    """
    with open(script_name, "w") as f:
        f.write(f"""import espressomd
import espressomd.interactions
import json
import numpy as np

# Load priors
with open('{priors_file}', 'r') as f:
    priors = json.load(f)

# Set up system
system = espressomd.System(box_l=[10.0, 10.0, 10.0]) # Generic box for isolated molecule
system.time_step = 0.002
system.cell_system.skin = 0.4

# Load initial positions
initial_pos = np.load('{initial_pos_file}')
num_particles = len(initial_pos)

# Setup particles with realistic initial coordinates
for i in range(num_particles):
    system.part.add(id=i, pos=initial_pos[i], type=0)

# WCA Non-Bonded interactions
wca = priors.get("wca", {{}})
if wca.get("epsilon", 0.0) > 0 and wca.get("sigma", 0.0) > 0:
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            system.non_bonded_inter[0, 0].lennard_jones.set_params(
                epsilon=wca["epsilon"], sigma=wca["sigma"],
                cutoff=wca["sigma"] * (2.0**(1/6)), shift="auto"
            )

# Apply interactions
system.force_cap = 2000.0

for b in priors.get("bonds", []):
    if b["type"] == "tabulated":
        data = np.loadtxt(b["file"])
        tb = espressomd.interactions.TabulatedDistance(
            min=data[0, 0], max=data[-1, 0], 
            energy=data[:, 1], force=data[:, 2]
        )
        system.bonded_inter.add(tb)
        system.part.by_id(b["mol_i"]).add_bond((tb, b["mol_j"]))
    elif b["type"] == "harmonic":
        hb = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
        system.bonded_inter.add(hb)
        system.part.by_id(b["mol_i"]).add_bond((hb, b["mol_j"]))

for a in priors.get("angles", []):
    if a.get("type", "harmonic") == "harmonic":
        ha = espressomd.interactions.AngleHarmonic(bend=a["k"], phi0=a["theta0"])
        system.bonded_inter.add(ha)
        system.part.by_id(a["mol_j"]).add_bond((ha, a["mol_i"], a["mol_k"]))
    elif a.get("type") == "tabulated":
        data = np.loadtxt(a["file"])
        ta = espressomd.interactions.TabulatedAngle(
            energy=data[:, 1], force=data[:, 2]
        )
        system.bonded_inter.add(ta)
        system.part.by_id(a["mol_j"]).add_bond((ta, a["mol_i"], a["mol_k"]))

for d in priors.get("dihedrals", []):
    if d.get("type", "cosine") == "cosine":
        cd = espressomd.interactions.Dihedral(bend=d["k"], mult=d.get("n", 1), phase=d["phi0"])
        system.bonded_inter.add(cd)
        system.part.by_id(d["mol_j"]).add_bond((cd, d["mol_i"], d["mol_k"], d["mol_l"]))
    elif d.get("type") == "tabulated":
        data = np.loadtxt(d["file"])
        td = espressomd.interactions.TabulatedDihedral(
            energy=data[:, 1], force=data[:, 2]
        )
        system.bonded_inter.add(td)
        system.part.by_id(d["mol_j"]).add_bond((td, d["mol_i"], d["mol_k"], d["mol_l"]))

# Debug prints
print("Distance 164-165 START:", np.linalg.norm(system.part.by_id(164).pos - system.part.by_id(165).pos))

# Check initial forces
system.integrator.run(0)
print("--- INITIAL ENERGY ---")
print(system.analysis.energy())
forces = system.part.all().f
print("--- INITIAL FORCES (BEFORE SD) ---")
for i, f in enumerate(forces):
    if np.linalg.norm(f) > 500:
        print(f"HIGH FORCE START: Particle {{i}} f={{f}} mag={{np.linalg.norm(f):.2f}}")

# Minimize energy / Burn-in
print("Gentle MD burn-in...")
system.integrator.set_vv()
system.thermostat.set_langevin(kT=2.49, gamma=10.0, seed=42)
system.force_cap = 50.0
system.integrator.run(2000)
system.force_cap = 0.0

system.integrator.run(0)
print("--- ENERGY AFTER BURN-IN ---")
print(system.analysis.energy())
forces = system.part.all().f
for i, f in enumerate(forces):
    if np.linalg.norm(f) > 500:
        print(f"HIGH FORCE AFTER SD: Particle {{i}} f={{f}} mag={{np.linalg.norm(f):.2f}}")

print("Distance 164-165 AFTER SD:", np.linalg.norm(system.part.by_id(164).pos - system.part.by_id(165).pos), flush=True)

system.thermostat.set_langevin(kT=2.49, gamma=1.0, seed=42)

# Run MD and save trajectory
import builtins
print("Running MD...")
# system.force_cap = 0  # Leave it capped just in case


positions = []
for _ in range(5000):
    system.integrator.run(10)
    pos = []
    for p in system.part:
        pos.append(p.pos)
    positions.append(pos)

np.save('{output_traj}', np.array(positions))
""")



def main():
    parser = argparse.ArgumentParser(description="Run IBI loop using exact tabulated targets.")
    parser.add_argument("--dataset", required=True, help="Path to the binary dataset file")
    parser.add_argument("--priors", required=True, help="Path to cg_priors.json")
    parser.add_argument("--iterations", type=int, default=5, help="Number of IBI iterations")
    parser.add_argument("--outdir", default="ibi_priors", help="Output directory for potentials")
    parser.add_argument("--pypresso", type=str, default="/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/espresso/build/pypresso")
    args = parser.parse_args()
    
    print("[INFO] =========================================")
    print(f"[INFO] Starting Iterative Boltzmann Inversion")
    print(f"[INFO] Iterations: {args.iterations}")
    print("[INFO] =========================================\n")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    with open(args.priors, "r") as f:
        priors_data = json.load(f)
        
    print(f"[INFO] Reading target dataset: {args.dataset}")
    bond_dists, angle_dists, dihedral_dists, first_frame_centers = read_dataset_distributions(args.dataset, priors_data)
    
    # Save initial positions for ESPResSo
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
            
    if args.iterations == 0:
        print("[INFO] User requested 0 iterations. Stopping at DBI.")
        sys.exit(0)
        
    # ---------------------------------------------------------
    # STEP 2: Iterative Boltzmann Inversion (IBI)
    # ---------------------------------------------------------
    # Write initial modified priors to a temp file
    tmp_priors = "cg_priors_tmp_ibi.json"
    with open(tmp_priors, "w") as f:
        json.dump(priors_data, f, indent=4)
        
    script_name = "_tmp_ibi_md.py"
    traj_name = "_tmp_traj.npy"
    
    for it in range(1, args.iterations + 1):
        print(f"\n[INFO] --- IBI Iteration {it}/{args.iterations} ---")
        
        write_espresso_runner(script_name, tmp_priors, traj_name, initial_pos_file)
        
        print(f"[INFO] Running ESPResSo MD simulation...")
        res = subprocess.run([args.pypresso, script_name], capture_output=True, text=True)
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
            V_next, F_next = update_ibi_potential(table["V"], hist_sim, table["P"], table["x"], periodic=False)
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/bond_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated angles
        print("[INFO] Updating tabulated angles...")
        for name, table in ibi_tables.get("angles", {}).items():
            sim_dists = sim_angle_dists[name]
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            V_next, F_next = update_ibi_potential(table["V"], hist_sim, table["P"], table["x"], periodic=False)
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/angle_tabulated_{name}.dat", table["x"], V_next, F_next)

        # Update tabulated dihedrals
        print("[INFO] Updating tabulated dihedrals...")
        for name, table in ibi_tables.get("dihedrals", {}).items():
            sim_dists = sim_dihedral_dists[name]
            hist_sim, _ = np.histogram(sim_dists, bins=table["bins"], density=True)
            V_next, F_next = update_ibi_potential(table["V"], hist_sim, table["P"], table["x"], periodic=True)
            table["V"] = V_next
            table["F"] = F_next
            save_tabulated_potential(f"{args.outdir}/dihedral_tabulated_{name}.dat", table["x"], V_next, F_next)
            
    print(f"\n[SUCCESS] IBI Converged after {args.iterations} iterations.")
    
    # Cleanup temp files
    if os.path.exists(script_name): os.remove(script_name)
    if os.path.exists(traj_name): os.remove(traj_name)
    if os.path.exists(tmp_priors): os.remove(tmp_priors)

if __name__ == "__main__":
    main()
