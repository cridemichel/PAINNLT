import argparse
import json
import numpy as np
import MDAnalysis as mda

def calculate_wca_force(r_vec, r_mag, epsilon, sigma):
    """Calcola la forza repulsiva WCA per usarla come Prior."""
    cutoff = sigma * (2.0 ** (1.0 / 6.0))
    if r_mag >= cutoff:
        return np.zeros(3)
    
    sr = sigma / r_mag
    sr6 = sr**6
    sr12 = sr6**2
    # Forza: - dE/dr = 4 * eps * (12*sr12/r - 6*sr6/r)
    f_mag = 4.0 * epsilon * (12.0 * sr12 - 6.0 * sr6) / r_mag
    return f_mag * (r_vec / r_mag)

def get_unwrapped_positions(positions, box_dim):
    unwrapped = np.copy(positions)
    ref = unwrapped[0]
    for i in range(1, len(unwrapped)):
        dvec = unwrapped[i] - ref
        dvec -= box_dim * np.round(dvec / box_dim)
        unwrapped[i] = ref + dvec
    return unwrapped

def main():
    parser = argparse.ArgumentParser(description="Convert GROMACS trajectory to TorchMD-Net NPZ")
    parser.add_argument("-c", "--topology", required=True, help="Topology file (.gro or .tpr)")
    parser.add_argument("-f", "--trajectory", required=True, help="Trajectory file with forces (.trr)")
    parser.add_argument("-m", "--mapping", required=True, help="JSON file with CG mapping")
    parser.add_argument("-p", "--priors", default=None, help="JSON file with Prior parameters")
    parser.add_argument("-o", "--output", default="dataset.npz", help="Output .npz file")
    args = parser.parse_args()

    # 1. Carica il mapping
    with open(args.mapping, "r") as f:
        mapping_data = json.load(f)
    mapping_by_resname = mapping_data.get("residues", {})
    site_types = mapping_data.get("site_types", {})

    # 2. Carica i Priors
    priors = {}
    if args.priors:
        with open(args.priors, "r") as f:
            priors = json.load(f)

    # 3. MDAnalysis
    u = mda.Universe(args.topology, args.trajectory)
    
    all_pos = []
    all_dy = []
    z_types = None
    
    def get_mass(atom_name):
        return 16.0 if "O" in atom_name.upper() else 1.0

    print(f"[INFO] Processando {len(u.trajectory)} frame...")
    for frame_idx, ts in enumerate(u.trajectory):
        if frame_idx % 100 == 0:
            print(f"       Frame {frame_idx}/{len(u.trajectory)}")
            
        box = ts.dimensions[:3] / 10.0 # nm
        
        frame_pos = []
        frame_dy = []
        if z_types is None:
            z_types = []
            
        for res in u.residues:
            resname = res.resname
            if resname not in mapping_by_resname:
                continue
                
            res_map = mapping_by_resname[resname]
            for site_name, atom_patterns in res_map.items():
                s_type = site_types[site_name]
                atoms = res.atoms
                pos = atoms.positions / 10.0 # da Angstrom a nm
                forces = ts.forces[atoms.indices] # kJ/(mol*nm)
                masses = np.array([get_mass(a.name) for a in atoms])
                
                unwrapped_pos = get_unwrapped_positions(pos, box)
                com = np.average(unwrapped_pos, weights=masses, axis=0)
                net_force = np.sum(forces, axis=0)
                
                frame_pos.append(com)
                frame_dy.append(net_force)
                
                if frame_idx == 0:
                    z_types.append(s_type)
        
        frame_pos = np.array(frame_pos)
        frame_dy = np.array(frame_dy)
        
        if "wca" in priors:
            eps = priors["wca"]["epsilon"]
            sig = priors["wca"]["sigma"]
            num_sites = len(frame_pos)
            for i in range(num_sites):
                for j in range(i+1, num_sites):
                    dvec = frame_pos[i] - frame_pos[j]
                    dvec -= box * np.round(dvec / box)
                    dist = np.linalg.norm(dvec)
                    f_wca_ij = calculate_wca_force(dvec, dist, eps, sig)
                    
                    frame_dy[i] -= f_wca_ij
                    frame_dy[j] += f_wca_ij
        
        all_pos.append(frame_pos)
        all_dy.append(frame_dy)

    z = np.array(z_types, dtype=np.int64)
    pos_array = np.array(all_pos, dtype=np.float32)
    dy_array = np.array(all_dy, dtype=np.float32)

    print(f"\n[INFO] Salvataggio in {args.output}...")
    np.savez(args.output, z=z, pos=pos_array, dy=dy_array)

if __name__ == "__main__":
    main()
