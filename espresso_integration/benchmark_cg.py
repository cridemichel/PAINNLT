import numpy as np
import struct
import argparse
import sys
import os
import time

try:
    import espressomd
    import espressomd.painn
    import json
except ImportError:
    print("[ERRORE] ESPResSo non trovato.")
    sys.exit(1)

def run_benchmark(model_path, config_path, rb_info_path, priors_path, bin_file, steps=5000):
    print(f"\n[INFO] Inizializzazione Benchmark su {steps} steps...")
    
    with open(bin_file, "rb") as f:
        f.read(4)
        num_molecules, num_total_sites = struct.unpack("ii", f.read(8))
        box = np.array(struct.unpack("3f", f.read(12)))
        initial_centers = []
        for mol_idx in range(num_molecules):
            mol_id, num_sites = struct.unpack("ii", f.read(8))
            cx, cy, cz, fx, fy, fz, tx, ty, tz = struct.unpack("9f", f.read(36))
            initial_centers.append([cx, cy, cz])
            for s in range(num_sites):
                f.read(16)
                
    system = espressomd.System(box_l=box)
    system.time_step = 0.005 # 5 fs per il benchmark
    system.cell_system.skin = 0.05
    system.cell_system.set_n_square()
    
    with open(rb_info_path, "r") as f:
        rb_info = json.load(f)
        
    resname = list(rb_info.keys())[0]
    data = rb_info[resname]
    mass = data["mass_amu"]
    rinertia = data["inertia_amu_nm2"]
    
    site_name = list(data["sites"].keys())[0]
    site_type = data["sites"][site_name]["type"]
    
    parts = []
    for pos in initial_centers:
        pos_wrapped = np.array(pos) % box
        p = system.part.add(pos=pos_wrapped, type=site_type, mass=mass, rinertia=rinertia, rotation=[True,True,True])
        parts.append(p)

    if priors_path and os.path.exists(priors_path):
        with open(priors_path, "r") as f:
            priors = json.load(f)
            if "wca" in priors:
                eps = priors["wca"].get("epsilon", 0)
                sig = priors["wca"].get("sigma", 0)
                if eps > 0 and sig > 0:
                    system.non_bonded_inter[site_type, site_type].lennard_jones.set_params(
                        epsilon=eps, sigma=sig, cutoff=sig*(2**(1/6)), shift="auto"
                    )
    
    with open(config_path, "r") as f:
        nn_config = json.load(f)
        
    cutoff = float(nn_config['cutoff'])
    system.min_global_cut = cutoff
    
    system.non_bonded_inter[site_type, site_type].soft_sphere.set_params(
        a=0.0, n=1, cutoff=cutoff, offset=0.0
    )

    espressomd.painn.activate_painn_potential(
        model_path=model_path, 
        num_species=int(nn_config['num_species']), 
        hidden_channels=int(nn_config['hidden_channels']), 
        n_layers=int(nn_config['n_layers']), 
        num_rbf=int(nn_config['num_rbf']), 
        cutoff=cutoff, 
        device="mps"
    )
    
    system.thermostat.set_langevin(kT=2.49, gamma=1.0, seed=42)
    
    print("[INFO] Warmup (100 steps)...")
    system.integrator.run(100)
    
    print("[INFO] Avvio misurazione benchmark...")
    start_time = time.time()
    
    system.integrator.run(steps)
    
    end_time = time.time()
    
    elapsed = end_time - start_time
    steps_per_sec = steps / elapsed
    ns_per_day = (steps_per_sec * system.time_step * 86400) / 1000.0
    
    print("\n" + "="*50)
    print(" RISULTATI BENCHMARK ESPRESSO + PaiNN (MPS)")
    print("="*50)
    print(f" Numero di siti CG: {num_molecules}")
    print(f" Timestep:          {system.time_step * 1000:.2f} fs")
    print(f" Tempo trascorso:   {elapsed:.2f} secondi")
    print(f" Steps al secondo:  {steps_per_sec:.1f} steps/s")
    print(f" Performance:       {ns_per_day:.2f} ns/day")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rb", type=str, required=True)
    parser.add_argument("--priors", type=str, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    
    args = parser.parse_args()
    
    run_benchmark(args.model, args.config, args.rb, args.priors, args.bin, steps=args.steps)
    sys.stdout.flush()
    os._exit(0)
