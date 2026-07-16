import json

with open("tel22_topology.json", "r") as f:
    config = json.load(f)

part_id = 0
mol_com_parts = {}
mol_vs_parts = {}

# Reconstruct particle IDs just like equilibrate.py
for i in range(220):
    mol_com_parts[i] = part_id
    part_id += 1

site_idx = 0
for mol_i, (m_type, r_name) in enumerate(config.get("mol_resnames", [])):
    if r_name in config.get("rigid_bodies", {}):
        rb_info = config["rigid_bodies"][r_name]
        for s_idx, s_name in enumerate(rb_info.get("sites", {}).keys()):
            mol_vs_parts[(mol_i, s_idx)] = part_id
            if part_id in [652, 654, 661, 668, 376, 383]:
                print(f"Particle {part_id} is Mol {mol_i} (resname {r_name}), Site {s_name} (idx {s_idx})")
            part_id += 1
