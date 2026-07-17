import espressomd
import espressomd.painn
import numpy as np
import struct

system = espressomd.System(box_l=[15.0, 15.0, 15.0])
system.time_step = 0.01
system.cell_system.skin = 0.4

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    system.box_l = [max(b, 11.0) for b in box_dim]
    
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        torque = struct.unpack("3f", f.read(12))
        
        system.part.add(pos=center, type=0)
        
        for s in range(num_sites):
            stype = struct.unpack("i", f.read(4))[0]
            spos = struct.unpack("3f", f.read(12))
            system.part.add(pos=spos, type=stype)

print("Particles added.")

import json
with open("tel22_training_config.json", "r") as f:
    nn_config = json.load(f)

espressomd.painn.activate_painn_potential(
    model_path="tel22_model.pt",
    num_species=nn_config["num_species"],
    hidden_channels=nn_config["hidden_channels"],
    n_layers=nn_config["n_layers"],
    num_rbf=nn_config["num_rbf"],
    cutoff=nn_config["cutoff"],
    device="auto"
)

system.integrator.run(0)

forces = system.part.all().f
f_norms = np.linalg.norm(forces, axis=1)
max_f = np.max(f_norms)
print("Max force from ML potential:", max_f)
print("Are there any NaNs?", np.isnan(forces).any())
if np.isnan(forces).any():
    print("NaNs found in particles:")
    for i, f in enumerate(forces):
        if np.isnan(f).any():
            print(f"Particle {i} has NaN force!")

