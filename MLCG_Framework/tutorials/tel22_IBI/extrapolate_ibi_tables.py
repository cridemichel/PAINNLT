import os
import glob
import numpy as np

def extrapolate_table(filepath, is_angle=False):
    # Read the data, skipping the header "# x energy force"
    try:
        data = np.loadtxt(filepath, comments='#')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    if len(data) < 2:
        print(f"File {filepath} has too few rows.")
        return

    x = data[:, 0]
    energy = data[:, 1]
    force = data[:, 2]

    x_min, u_min, f_min = x[0], energy[0], force[0]
    x_max, u_max, f_max = x[-1], energy[-1], force[-1]

    # Calculate slopes for force extrapolation
    # slope = df/dx. We want negative slopes for restoring forces.
    k_left = (force[1] - force[0]) / (x[1] - x[0])
    if k_left > 0:
        k_left = -1000.0  # Enforce stiff repulsion if not inherently repulsive
        
    k_right = (force[-1] - force[-2]) / (x[-1] - x[-2])
    if k_right > 0:
        k_right = -1000.0 # Enforce stiff restoring force if not inherently attractive

    # Define the padding grid
    dx = x[1] - x[0]
    
    # Left padding
    target_min = 0.0 if is_angle else 0.001
    num_left = int(np.floor((x_min - target_min) / dx))
    
    left_rows = []
    if num_left > 0:
        for i in range(num_left, 0, -1):
            xi = x_min - i * dx
            if xi < target_min: continue
            # Linear force extrapolation: F(x) = f_min + k_left * (x - x_min)
            fi = f_min + k_left * (xi - x_min)
            # Energy integration: U(x) = U_min - ( F_min*(x - x_min) + 0.5*k_left*(x - x_min)^2 )
            ui = u_min - (f_min * (xi - x_min) + 0.5 * k_left * (xi - x_min)**2)
            left_rows.append([xi, ui, fi])

    # Right padding
    target_max = np.pi if is_angle else 5.0
    num_right = int(np.floor((target_max - x_max) / dx))
    
    right_rows = []
    if num_right > 0:
        for i in range(1, num_right + 1):
            xi = x_max + i * dx
            if xi > target_max: continue
            # Linear force extrapolation: F(x) = f_max + k_right * (x - x_max)
            fi = f_max + k_right * (xi - x_max)
            # Energy integration: U(x) = U_max - ( F_max*(x - x_max) + 0.5*k_right*(x - x_max)^2 )
            ui = u_max - (f_max * (xi - x_max) + 0.5 * k_right * (xi - x_max)**2)
            right_rows.append([xi, ui, fi])

    # Combine
    new_data = []
    if left_rows:
        new_data.extend(left_rows)
    new_data.extend(data.tolist())
    if right_rows:
        new_data.extend(right_rows)

    new_data = np.array(new_data)

    # Save over the same file
    with open(filepath, 'w') as f:
        f.write("# x energy force (Extrapolated)\n")
        for row in new_data:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
            
    print(f"Extrapolated {os.path.basename(filepath)}: [{x_min:.3f}, {x_max:.3f}] -> [{new_data[0,0]:.3f}, {new_data[-1,0]:.3f}]")

def main():
    priors_dir = "ibi_priors"
    if not os.path.exists(priors_dir):
        print(f"Directory {priors_dir} not found.")
        return

    # Extrapolate bonds
    for bond_file in glob.glob(os.path.join(priors_dir, "bond_tabulated_*.dat")):
        extrapolate_table(bond_file, is_angle=False)

    # Extrapolate angles
    for angle_file in glob.glob(os.path.join(priors_dir, "angle_tabulated_*.dat")):
        extrapolate_table(angle_file, is_angle=True)

    print("All tables successfully extrapolated!")

if __name__ == "__main__":
    main()
