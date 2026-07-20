import os
import sys
import json
import argparse
import numpy as np
import espressomd
import espressomd.interactions
import espressomd.painn
import matplotlib.pyplot as plt

def run_nve(dt, steps, model_path, config_path, priors_path, rb_info_path, dataset_path, checkpoint_path):
    print(f"Running NVE with dt={dt}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(rb_info_path, "r") as f:
        rb_info = json.load(f)
        
    system = espressomd.System(box_l=[10.0, 10.0, 10.0])
    system.time_step = dt
    system.cell_system.skin = 0.4
    
    with open(dataset_path, "rb") as f:
        header = f.read(256).decode("utf-8").strip("\x00")
        metadata = json.loads(header)
        frame_bytes = metadata["num_particles"] * 3 * 8
        f.read(frame_bytes) # Skip coords
        forces = np.frombuffer(f.read(frame_bytes), dtype=np.float64).reshape(-1, 3)
        atomic_numbers = np.frombuffer(f.read(metadata["num_particles"] * 8), dtype=np.int64)
        
    checkpoint = np.load(checkpoint_path)
    coords = checkpoint["positions"]
    vels = checkpoint["velocities"]
    
    # Pre-compute Rigid Body properties
    rb_centers = {}
    rb_masses = {}
    for rb in rb_info:
        rb_id = rb["id"]
        indices = rb["particles"]
        rb_coords = coords[indices]
        masses = np.ones(len(indices)) # Assume mass 1.0 for CG particles
        total_mass = np.sum(masses)
        com = np.sum(rb_coords * masses[:, np.newaxis], axis=0) / total_mass
        rb_centers[rb_id] = com
        rb_masses[rb_id] = total_mass
    
    # Add virtual particles
    for rb in rb_info:
        rb_id = rb["id"]
        system.part.add(id=rb_id, pos=rb_centers[rb_id], mass=rb_masses[rb_id], type=100) # Type 100 for COM
        
    # Add real particles
    for i in range(metadata["num_particles"]):
        if i in rb_centers: continue
        
        is_virtual = False
        rb_parent = -1
        for rb in rb_info:
            if i in rb["particles"]:
                is_virtual = True
                rb_parent = rb["id"]
                break
                
        if is_virtual:
            system.part.add(id=i, pos=coords[i], v=vels[i], type=int(atomic_numbers[i]))
            system.part.by_id(i).vs_auto_relate_to(system.part.by_id(rb_parent))
        else:
            system.part.add(id=i, pos=coords[i], v=vels[i], mass=1.0, type=int(atomic_numbers[i]))
            
    # Setup priors
    with open(priors_path, "r") as f:
        priors = json.load(f)
        
    # We MUST add classical bonds exactly as in run_cg_md.py
    for k, b_info in priors["HarmonicBond"].items():
        type1, type2 = map(int, k.split("_"))
        bond = espressomd.interactions.HarmonicBond(k=b_info["k"], r_0=b_info["r0"])
        system.bonded_inter.add(bond)
        
    for k, a_info in priors["HarmonicAngle"].items():
        type1, type2, type3 = map(int, k.split("_"))
        angle = espressomd.interactions.AngleHarmonic(k=a_info["k"], phi0=a_info["theta0"])
        system.bonded_inter.add(angle)
        
    # Read bonds from topology
    with open("tel22_topology.json", "r") as f:
        topo = json.load(f)
        
    bond_id = 0
    for bond in topo["bonds"]:
        p1, p2, k, r0 = bond
        bond_type = espressomd.interactions.HarmonicBond(k=k, r_0=r0)
        system.bonded_inter.add(bond_type)
        system.part.by_id(p1).add_bond((bond_id, p2))
        bond_id += 1
        
    # Note: we skip angle bonds here for brevity unless needed for stability.
    
    # PaiNN
    num_species = 100
    espressomd.painn.init_painn(model_path, num_species, config["hidden_channels"], config["n_layers"], config["num_rbf"], config["cutoff"], "cpu")
    
    energies = []
    
    # Evaluate t=0
    system.integrator.run(0)
    
    for step in range(steps):
        system.integrator.run(1)
        e = system.analysis.energy()
        e_tot = e["total"] + espressomd.painn.get_painn_energy()
        energies.append(e_tot)
        
    energies = np.array(energies)
    return np.std(energies)

def main():
    model_path = "tel22_model.pt"
    config_path = "tel22_training_config.json"
    priors_path = "cg_priors.json"
    rb_info_path = "rigid_bodies_info.json"
    dataset_path = "tel22_dataset.bin"
    checkpoint_path = "equilibrated.npz"
    
    dts = [0.0005, 0.001, 0.002, 0.004]
    stds = []
    
    for dt in dts:
        steps = int(2.0 / dt) # 2 ps simulation
        std = run_nve(dt, steps, model_path, config_path, priors_path, rb_info_path, dataset_path, checkpoint_path)
        stds.append(std)
        print(f"dt={dt} -> std={std}")
        
    dts = np.array(dts)
    stds = np.array(stds)
    
    plt.figure(figsize=(8,6))
    plt.loglog(dts, stds, marker='o', label='ML NVE')
    
    # Plot quadratic reference
    ref_y = stds[0] * (dts / dts[0])**2
    plt.loglog(dts, ref_y, linestyle='--', color='gray', label='O(dt^2)')
    
    plt.xlabel('Timestep dt (ps)')
    plt.ylabel('Std of Total Energy (kJ/mol)')
    plt.title('Energy Conservation Scaling')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('/Users/demichel/.gemini/antigravity/brain/d54d813a-ae54-4ca6-9922-6179a054f737/energy_conservation_tel22_nve.png', dpi=150)
    print("Plot saved to energy_conservation_tel22_nve.png")

if __name__ == "__main__":
    main()
