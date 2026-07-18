import numpy as np
import espressomd
import espressomd.painn
from espressomd import thermostat
import json

with open("tel22_training_config.json") as f:
    nn_config = json.load(f)

system = espressomd.System(box_l=[100.0, 100.0, 100.0])
system.time_step = 0.001
system.cell_system.skin = 0.4

chk = np.load("equilibrated.npz")
pos = chk["pos"]
vel = chk["v"]
quat = chk["quat"]
omega = chk["omega"]

for i in range(len(pos)):
    system.part.add(pos=pos[i], v=vel[i], quat=quat[i], omega_body=omega[i])

with open("rigid_bodies_info.json") as f:
    rb_info = json.load(f)

for core_id_str, rb in rb_info.items():
    core_id = int(core_id_str)
    system.part.by_id(core_id).mass = rb["mass"]
    system.part.by_id(core_id).rinertia = rb["rinertia"]
    system.part.by_id(core_id).rotation = [True, True, True]
    
    for vs in rb["virtual_sites"]:
        vs_id = vs["id"]
        rel_pos = vs["rel_pos"]
        system.part.by_id(vs_id).is_virtual = True
        system.part.by_id(vs_id).vs_auto_relate_to(core_id)

espressomd.painn.activate_painn_potential(
    model_path="tel22_model.pt",
    num_species=nn_config["num_species"],
    hidden_channels=nn_config["hidden_channels"],
    n_layers=nn_config["n_layers"],
    num_rbf=nn_config["num_rbf"],
    cutoff=nn_config["cutoff"],
    device="cpu"
)

system.integrator.run(0)
en = system.analysis.energy()
print("E_tot before any step:", en["total"])
print("E_kin before any step:", en["kinetic"])
print("max_f:", max([sum([f**2 for f in p.f])**0.5 for p in system.part]))
