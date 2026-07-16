import sys, os
sys.path.append("/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/espresso/build/src/python")
import espressomd
import numpy as np
import json
import espressomd.painn

with open("tel22_topology.json") as f: config = json.load(f)
priors = config["priors"]
nn = config["neural_network"]

system = espressomd.System(box_l=[15.0, 15.0, 15.0])
system.time_step = 0.001
system.cell_system.skin = 0.4

# Setup particles and ML Potential (no bonds/priors)
# We just want to see PaiNN forces on frame 0
import struct
with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_mols = struct.unpack("i", f.read(4))[0]
    num_sites = struct.unpack("i", f.read(4))[0]
    box = struct.unpack("3f", f.read(12))
    centers = [struct.unpack("3f", f.read(12)) for _ in range(num_mols)]

system.box_l = box
for i, pos in enumerate(centers):
    system.part.add(id=i, pos=pos)

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
print("Max PaiNN force:", max(forces))
print("Mean PaiNN force:", np.mean(forces))
