import espressomd
import espressomd.painn
import json
import argparse
import numpy as np
import struct
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False, default=None, help="Trained ML potential (.pt)")
parser.add_argument("--config", type=str, required=True, help="NN config JSON")
parser.add_argument("--priors", type=str, required=True, help="cg_priors.json")
parser.add_argument("--rb_info", type=str, required=True, help="rigid_bodies_info.json")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to get initial frame from (e.g. cg_dataset.bin)")
parser.add_argument("--checkpoint", type=str, default=None, help="NPZ file with pos and v to load instead of dataset positions")
parser.add_argument("--dt", type=float, default=0.002, help="Time step (ps)")
parser.add_argument("--steps", type=int, default=10000, help="Simulation steps")
parser.add_argument("--no_log", action="store_true", help="Disable wandb logging")
parser.add_argument("--log_interval", type=int, default=10, help="Interval for writing trajectory (default: 10)")
parser.add_argument("--device", type=str, default="auto", help="Device for ML (cpu, mps, cuda, auto)")
parser.add_argument("--kT", type=float, default=2.49, help="Simulation temperature in kJ/mol (default 2.49 for 300K)")
parser.add_argument("--nve", action="store_true", help="Run NVE simulation (no thermostat)")

parser.add_argument("--toxvaerd_alpha", type=float, default=0.1, help="Toxvaerd smoothing dimensionless parameter")
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
# system.force_cap = 10000.0 # Rimosso! Limitava il WCA permettendo compenetrazioni. Abbiamo limitato le forze ML in C++.
if not args.nve:
    system.thermostat.set_langevin(kT=args.kT, gamma=1.0, gamma_rot=1.0, seed=42)
else:
    system.thermostat.turn_off()


print(f"[INFO] Running {args.steps} integration steps...")

def get_rb_data_by_sites(site_types, rb_info):
    for resname, data in rb_info.items():
        expected_types = [site["type"] for site in data["sites"].values()]
        if sorted(expected_types) == sorted(site_types):
            return data
    raise ValueError(f"Unknown site types {site_types}")

print("[INFO] Reading initial frame from dataset...")
mol_com_parts = {}
mol_vs_parts = {}

with open(args.dataset, "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    system.box_l = [b for b in box_dim]
    
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        torque = struct.unpack("3f", f.read(12))
        
        site_types = []
        site_positions = []
        for s in range(num_sites):
            stype = struct.unpack("i", f.read(4))[0]
            spos = struct.unpack("3f", f.read(12))
            site_types.append(stype)
            site_positions.append(spos)
            
        rb_data = get_rb_data_by_sites(site_types, rb_info)
        mass = rb_data["mass_amu"]
        inertia = rb_data["inertia_amu_nm2"]
        
        p_com = system.part.add(
            pos=center, type=DUMMY_COM_TYPE,
            mass=mass, rinertia=inertia,
            rotation=[True, True, True] if num_sites > 1 else [False, False, False],
            mol_id=mol_idx
        )
        mol_com_parts[mol_idx] = p_com.id
        
        for site_idx, (stype, spos) in enumerate(zip(site_types, site_positions)):
            # Virtual sites must have near-zero mass/inertia to not inflate the total system mass.
            # ESPResSo requires mass > 0, so we use 1e-5.
            p_vs = system.part.add(pos=spos, type=stype, mass=1e-5, rinertia=[1e-5, 1e-5, 1e-5], mol_id=mol_idx)
            p_vs.virtual = True
            p_vs.vs_auto_relate_to(p_com.id)
            p_vs.gamma = 0.0
            p_vs.gamma_rot = 0.0
            mol_vs_parts[(mol_idx, site_idx)] = p_vs.id



for i in range(10):
    p = system.part.by_id(i)
    print(f"ID: {i} | Type: {p.type} | Virtual: {p.is_virtual} | Mass: {p.mass} | Rot: {p.rotation}")
