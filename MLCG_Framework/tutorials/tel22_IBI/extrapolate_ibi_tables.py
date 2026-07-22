import os
import glob
import numpy as np

def extrapolate_table(filepath, is_angle=False):
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

    x_min, u_min, f_min = x[0], energy[0], force[0]
    x_max, u_max, f_max = x[-1], energy[-1], force[-1]

    # Padding
    dx = x[1] - x[0]
    
    # Left padding
    target_min = 0.01 if is_angle else 0.001
    num_left = int(np.floor((x_min - target_min) / dx))
    
    left_rows = []
    if num_left > 0:
        for i in range(num_left, 0, -1):
            xi = x_min - i * dx
            if xi < target_min: continue
            
            if not is_angle:
                # WCA Extrapolation for Bonds: F(x) = F_min * (x_min/x)^13
                # We assume f_min is repulsive (positive). If not, fallback to constant force.
                if f_min > 0 and xi > 0:
                    ratio = x_min / xi
                    fi = f_min * (ratio**13)
                    # U(x) = U_min + F_min*x_min/12 * ( (x_min/x)^12 - 1 )
                    ui = u_min + (f_min * x_min / 12.0) * ((ratio**12) - 1.0)
                else:
                    f_left_extrap = max(f_min, 10.0)
                    fi = f_left_extrap
                    ui = u_min - fi * (xi - x_min)
            else:
                # Constant force extrapolation for angles
                f_left_extrap = max(f_min, 10.0)
                fi = f_left_extrap
                ui = u_min - fi * (xi - x_min)
                
            left_rows.append([xi, ui, fi])

    # Right padding
    target_max = np.pi if is_angle else 5.0
    num_right = int(np.floor((target_max - x_max) / dx))
    
    right_rows = []
    if num_right > 0:
        for i in range(1, num_right + 1):
            xi = x_max + i * dx
            if xi > target_max: continue
            
            # Right side extrapolation (bonds and angles): Constant force
            f_right_extrap = min(f_max, -10.0)
            fi = f_right_extrap
            ui = u_max - fi * (xi - x_max)
                
            right_rows.append([xi, ui, fi])

    # Combine
    new_data = []
    if left_rows:
        new_data.extend(left_rows)
    new_data.extend(data.tolist())
    if right_rows:
        new_data.extend(right_rows)

    new_data = np.array(new_data)

    # Save
    with open(filepath, 'w') as f:
        f.write("# x energy force (WCA/Constant Extrapolated)\n")
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

    print("All tables successfully WCA-extrapolated!")

if __name__ == "__main__":
    main()
