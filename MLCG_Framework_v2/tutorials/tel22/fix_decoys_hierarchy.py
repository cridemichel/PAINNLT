import sys

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()

# Replace the block that writes decoys to binary
old_write_block = """print("[INFO] Scrittura decoy nel binario...")
with open(args.output, "ab") as f:
    for d_idx, (d_sites, d_centers, d_forces, d_box) in enumerate(decoy_frames):
        # Flatten positions and types for the binary format
        flat_pos = []
        flat_forces = []
        flat_types = []
        
        for m_idx, sites in enumerate(d_sites):
            for (s_type, s_pos) in sites:
                flat_pos.append(s_pos)
                flat_types.append(s_type)
                
        # Need absolute forces flat array
        # Wait, d_forces was zeroed, we just need to flatten it
        flat_pos = np.array(flat_pos, dtype=np.float32)
        flat_forces = np.zeros((len(flat_pos), 3), dtype=np.float32)
        flat_types = np.array(flat_types, dtype=np.int32)
        
        # Write to bin file
        n_particles = len(flat_pos)
        f.write(struct.pack('Q', n_particles))
        f.write(struct.pack('3f', float(d_box[0]), float(d_box[1]), float(d_box[2])))
        f.write(flat_types.tobytes())
        f.write(flat_pos.tobytes())
        f.write(flat_forces.tobytes())"""

new_write_block = """print("[INFO] Scrittura decoy nel binario...")
with open(args.output, "ab") as f:
    for d_idx, (d_sites, d_centers, d_forces, d_box) in enumerate(decoy_frames):
        num_molecules = len(d_sites)
        num_total_sites = sum(len(sites) for sites in d_sites)
        
        f.write(struct.pack("i", num_molecules))
        f.write(struct.pack("i", num_total_sites))
        f.write(struct.pack("3f", float(d_box[0]), float(d_box[1]), float(d_box[2])))
        
        for mol_id in range(num_molecules):
            num_sites = len(d_sites[mol_id])
            f.write(struct.pack("i", mol_id))
            f.write(struct.pack("i", num_sites))
            f.write(struct.pack("3f", *d_centers[mol_id]))
            f.write(struct.pack("3f", *d_forces[mol_id]))
            f.write(struct.pack("3f", 0.0, 0.0, 0.0)) # torques are zero
            for site_type, site_pos in d_sites[mol_id]:
                f.write(struct.pack("i", int(site_type)))
                f.write(struct.pack("3f", *site_pos))"""

content = content.replace(old_write_block, new_write_block)

with open(filepath, "w") as f:
    f.write(content)
print("Fixed decoys hierarchical file writing logic!")
