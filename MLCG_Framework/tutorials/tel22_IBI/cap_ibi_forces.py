import glob
import os
import numpy as np

def cap_table(filepath, max_force=300.0):
    try:
        data = np.loadtxt(filepath, comments='#')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    if len(data) < 2:
        return

    x = data[:, 0]
    energy = data[:, 1]
    force = data[:, 2]

    # Find energy minimum to anchor the integration
    min_idx = np.argmin(energy)
    
    # Clip forces
    new_force = np.clip(force, -max_force, max_force)
    
    # Re-integrate energy: F = -dU/dx => dU = -F dx
    new_energy = np.zeros_like(energy)
    new_energy[min_idx] = energy[min_idx]  # keep the same minimum baseline
    
    # Integrate right
    for i in range(min_idx + 1, len(x)):
        dx = x[i] - x[i-1]
        new_energy[i] = new_energy[i-1] - 0.5 * (new_force[i] + new_force[i-1]) * dx
        
    # Integrate left
    for i in range(min_idx - 1, -1, -1):
        dx = x[i+1] - x[i]
        new_energy[i] = new_energy[i+1] + 0.5 * (new_force[i] + new_force[i+1]) * dx

    # Write back
    with open(filepath, 'w') as f:
        f.write("# x energy force (Capped and Re-integrated)\n")
        for i in range(len(x)):
            f.write(f"{x[i]:.6f} {new_energy[i]:.6f} {new_force[i]:.6f}\n")
    
    print(f"Capped {os.path.basename(filepath)} at max_force={max_force}.")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Cap IBI forces to a maximum threshold and re-integrate energy.")
    parser.add_argument("--max_force", type=float, default=300.0, help="Maximum force threshold (default: 300.0)")
    args = parser.parse_args()

    priors_dir = "ibi_priors"
    if not os.path.exists(priors_dir):
        print(f"Directory {priors_dir} not found.")
        return

    # Cap all bonds and angles using the configured max_force.
    print(f"Applying force cap at max_force={args.max_force} kJ/(mol*nm)...")
    for dat_file in glob.glob(os.path.join(priors_dir, "*_tabulated_*.dat")):
        cap_table(dat_file, max_force=args.max_force)

if __name__ == "__main__":
    main()
