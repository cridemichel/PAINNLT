import sys
sys.path.append("/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/espresso/build/src/python")
import espressomd
import numpy as np
import json
import torch
import espressomd.painn

# Minimal ESPResSo setup to check forces
with open("tel22_topology.json") as f:
    config = json.load(f)
priors = config["priors"]
nn = config["neural_network"]

system = espressomd.System(box_l=[15.0, 15.0, 15.0])
system.time_step = 0.001
system.cell_system.skin = 0.4

# Read frame 0
import struct
with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_mols = struct.unpack("i", f.read(4))[0]
    num_sites = struct.unpack("i", f.read(4))[0]
    box = struct.unpack("3f", f.read(12))
    
    centers = []
    for _ in range(num_mols): centers.append(struct.unpack("3f", f.read(12)))
    # skip forces, torques
    f.seek(num_mols * 3 * 4 * 2, 1)
    
    sites = []
    for _ in range(num_sites): sites.append(struct.unpack("i3f", f.read(16)))

system.box_l = box

mol_com_parts = {}
mol_vs_parts = {}
part_id = 0

for i, pos in enumerate(centers):
    system.part.add(id=part_id, pos=pos)
    mol_com_parts[i] = part_id
    part_id += 1

site_idx = 0
for mol_i, (m_type, r_name) in enumerate(config.get("mol_resnames", [])):
    if r_name in config.get("rigid_bodies", {}):
        rb_info = config["rigid_bodies"][r_name]
        for s_name, s_info in rb_info.get("sites", {}).items():
            _, x, y, z = sites[site_idx]
            system.part.add(id=part_id, pos=[x,y,z], virtual=1, vs_auto_relate_to=mol_com_parts[mol_i])
            mol_vs_parts[(mol_i, list(rb_info["sites"].keys()).index(s_name))] = part_id
            part_id += 1
            site_idx += 1
    else:
        for s_name in config["site_mapping"].keys():
            _, x, y, z = sites[site_idx]
            system.part.add(id=part_id, pos=[x,y,z], virtual=1, vs_auto_relate_to=mol_com_parts[mol_i])
            mol_vs_parts[(mol_i, list(config["site_mapping"].keys()).index(s_name))] = part_id
            part_id += 1
            site_idx += 1

# Activate ML potential to compute forces
for i in range(nn["num_species"] + 2):
    for j in range(i, nn["num_species"] + 2):
        system.non_bonded_inter[i, j].soft_sphere.set_params(a=0.0, n=1, cutoff=5.0, offset=0.0)

espressomd.painn.activate_painn_potential(
    model_path="tel22_model.pt", num_species=nn["num_species"],
    hidden_channels=nn["hidden_channels"], n_layers=nn["n_layers"],
    num_rbf=nn["num_rbf"], cutoff=nn["cutoff"], device="cpu"
)

system.integrator.run(0)

forces = [np.linalg.norm(p.f) for p in system.part]
print("Max force (ML only):", max(forces))
