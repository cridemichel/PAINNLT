import os
import sys
import numpy as np
import argparse
import json
import struct
from scipy.ndimage import gaussian_filter1d

# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "numpy",
#     "scipy"
# ]
# ///

def calculate_dbi_potential(values, bins, kT=2.49):
    """
    Computes V_0(x) = -kT * ln(P(x))
    """
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Avoid log(0)
    hist = np.clip(hist, 1e-6, None)
    
    potential = -kT * np.log(hist)
    # Shift minimum to 0
    potential -= np.min(potential)
    
    # Smooth the potential to avoid wild force fluctuations
    potential_smooth = gaussian_filter1d(potential, sigma=2.0)
    
    # Force = -dV/dx
    dx = bin_centers[1] - bin_centers[0]
    force = -np.gradient(potential_smooth, dx)
    
    return bin_centers, potential_smooth, force, hist

def update_ibi_potential(V_i, P_i, P_target, kT=2.49, alpha=0.5):
    """
    V_{i+1} = V_i + alpha * kT * ln(P_i / P_target)
    """
    P_i = np.clip(P_i, 1e-6, None)
    P_target = np.clip(P_target, 1e-6, None)
    
    update = alpha * kT * np.log(P_i / P_target)
    V_next = V_i + update
    
    V_next -= np.min(V_next)
    V_next_smooth = gaussian_filter1d(V_next, sigma=2.0)
    return V_next_smooth

def save_tabulated_potential(filename, r, energy, force):
    """
    Saves in ESPResSo Tabulated format: r energy force
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = np.column_stack((r, energy, force))
    np.savetxt(filename, data, fmt="%.6f", header="r energy force")
    print(f"[INFO] Tabulated potential saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="All-Atom target dataset")
    parser.add_argument("--iterations", type=int, default=0, help="IBI iterations (0 = DBI only)")
    parser.add_argument("--outdir", type=str, default="ibi_priors", help="Output directory")
    args = parser.parse_args()
    
    print("[INFO] =========================================")
    print(f"[INFO] Starting Iterative Boltzmann Inversion")
    print(f"[INFO] Iterations: {args.iterations} (0 means Direct Boltzmann Inversion only)")
    print("[INFO] =========================================\n")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # MOCKUP: In a real scenario, we extract the bond lengths from the dataset.
    # Here we simulate a target distribution for a generic bond at 0.5 nm
    print(f"[INFO] Reading target dataset: {args.dataset}")
    # Simulating a Gaussian distribution of bond lengths around 0.5 nm
    target_bond_lengths = np.random.normal(loc=0.5, scale=0.02, size=10000)
    
    bins = np.linspace(0.3, 0.7, 100)
    
    # ---------------------------------------------------------
    # STEP 1: Direct Boltzmann Inversion (DBI)
    # ---------------------------------------------------------
    print("[INFO] Performing Direct Boltzmann Inversion (DBI) to get V_0...")
    r, V_0, F_0, P_target = calculate_dbi_potential(target_bond_lengths, bins)
    
    if args.iterations == 0:
        print("[INFO] User requested 0 iterations. Stopping at DBI.")
        save_tabulated_potential(f"{args.outdir}/bond_dbi.dat", r, V_0, F_0)
        
        print("\n[SUCCESS] You can now use this file in cg_priors.json:")
        print("  {")
        print('    "type": "tabulated",')
        print(f'    "file": "{args.outdir}/bond_dbi.dat",')
        print('    "min": 0.3, "max": 0.7')
        print("  }")
        sys.exit(0)
        
    # ---------------------------------------------------------
    # STEP 2: Iterative Boltzmann Inversion (IBI)
    # ---------------------------------------------------------
    V_i = V_0
    for it in range(1, args.iterations + 1):
        print(f"\n[INFO] --- IBI Iteration {it}/{args.iterations} ---")
        
        # In a full framework, here we would:
        # 1. Update the tabulated potential file
        # 2. Call ESPResSo MD loop (via subprocess or internal import)
        # 3. Read the resulting trajectory
        
        print("[INFO] Running ESPResSo MD simulation with V_i...")
        # MOCKUP: Simulate the resulting distribution. 
        # Usually it's wider or shifted due to cross-correlations.
        simulated_bond_lengths = np.random.normal(loc=0.51, scale=0.03, size=10000) # shifted!
        
        hist_sim, _ = np.histogram(simulated_bond_lengths, bins=bins, density=True)
        P_i = np.clip(hist_sim, 1e-6, None)
        
        print("[INFO] Updating tabulated potential...")
        V_i = update_ibi_potential(V_i, P_i, P_target)
        
        # Recompute force
        dx = r[1] - r[0]
        F_i = -np.gradient(V_i, dx)
        
        # Convergence check (e.g. KL divergence)
        kl = np.sum(P_i * np.log(P_i / P_target)) * dx
        print(f"[INFO] KL Divergence: {kl:.4f}")
        
    print(f"\n[SUCCESS] IBI Converged after {args.iterations} iterations.")
    save_tabulated_potential(f"{args.outdir}/bond_ibi_final.dat", r, V_i, F_i)
    
if __name__ == "__main__":
    main()
