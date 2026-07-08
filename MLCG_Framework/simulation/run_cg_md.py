import espressomd
import espressomd.painn
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Trained ML potential (.pt)")
parser.add_argument("--config", type=str, required=True, help="NN config JSON")
parser.add_argument("--priors", type=str, required=True, help="cg_priors.json")
parser.add_argument("--rb_info", type=str, required=True, help="rigid_bodies_info.json")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to get initial frame from (e.g. cg_dataset.bin)")
parser.add_argument("--dt", type=float, default=0.002, help="Time step (ps)")
parser.add_argument("--steps", type=int, default=10000, help="Simulation steps")
parser.add_argument("--device", type=str, default="auto", help="Device for ML (cpu/mps/cuda)")
args = parser.parse_args()

print("[INFO] Loading configurations...")
with open(args.config, "r") as f:
    nn_config = json.load(f)
with open(args.priors, "r") as f:
    priors = json.load(f)
with open(args.rb_info, "r") as f:
    rb_info = json.load(f)

# The dummy particle type for COMs should be higher than the max ML species type
DUMMY_COM_TYPE = nn_config["num_species"] + 1

# Setup ESPResSo System
print("[INFO] Initializing ESPResSo system...")
# For a real run, box_l should be read from the first frame of the dataset or config.
# Here we just set a large box and will resize it if needed.
system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.time_step = args.dt
system.cell_system.skin = 0.4
system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=42)

# Reading initial frame logic would go here. 
# For demonstration of the integration, we assume particles are added:
# For each molecule:
#  com_part = system.part.add(pos=..., type=DUMMY_COM_TYPE, mass=..., rinertia=..., rotation=[True,True,True])
#  for site in sites:
#      vs = system.part.add(pos=..., type=site_type, virtual=True)
#      vs.vs_auto_relate_to(com_part.id)

print("[INFO] Adding priors...")
# WCA
wca = priors.get("wca", {})
if wca.get("epsilon", 0.0) > 0 and wca.get("sigma", 0.0) > 0:
    for i in range(nn_config["num_species"]):
        for j in range(i, nn_config["num_species"]):
            system.non_bonded_inter[i, j].lennard_jones.set_params(
                epsilon=wca["epsilon"], sigma=wca["sigma"],
                cutoff=wca["sigma"] * (2.0**(1/6)), shift="auto"
            )

# Bonds (Harmonic, FENE, Morse)
for idx, b in enumerate(priors.get("bonds", [])):
    b_type = b.get("type", "harmonic")
    if b_type == "harmonic":
        bond = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
    elif b_type == "fene":
        bond = espressomd.interactions.FeneBond(k=b["k"], d_r_max=b["r_max"], r_0=b["r0"])
    elif b_type == "morse":
        # Note: ESPResSo Morse bond might have different parameter names depending on version
        # Usually: eps=D, alpha=a, rmin=r0
        bond = espressomd.interactions.MorseBond(eps=b["D"], alpha=b["a"], rmin=b["r0"], cutoff=b["r0"] + 3.0/b["a"])
    else:
        print(f"[WARNING] Unknown bond type: {b_type}")
        continue
    
    system.bonded_inter.add(bond)
    print(f"[INFO] Added {b_type} bond {idx}: mol {b['mol_i']} <-> mol {b['mol_j']}")
    # The actual application `system.part.by_id(p1).add_bond((bond, p2))` 
    # must be done after particles are added, mapping mol_i to its actual particle ID.

print("[INFO] Activating ML Potential...")
espressomd.painn.activate_painn_potential(
    model_path=args.model,
    num_species=nn_config["num_species"],
    hidden_channels=nn_config["hidden_channels"],
    n_layers=nn_config["n_layers"],
    num_rbf=nn_config["num_rbf"],
    cutoff=nn_config["cutoff"],
    device_str=args.device
)

print(f"[INFO] Running {args.steps} integration steps...")
system.integrator.run(args.steps)
print("[INFO] Simulation finished successfully.")

