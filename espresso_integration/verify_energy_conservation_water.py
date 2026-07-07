import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
import sys
import os

try:
    import espressomd
    import espressomd.painn
    import json
except ImportError:
    print("[ERRORE] ESPResSo non trovato.")
    sys.exit(1)

def run_nve(model_path, config_path, rb_info_path, priors_path, bin_file, dt, device="cpu"):
    print(f"\n[INFO] Avvio test NVE con dt = {dt:.4f} ps ({dt*1000:.2f} fs) su {device}")
    
    # Generiamo esattamente 9 atomi su una griglia 3x3x1 in un box molto compatto
    box_size = 0.9  # Box size ridotto (0.9 nm) per alta densità
    initial_centers = []
    spacing = box_size / 3.0 # Spacing di 0.3 nm (esattamente uguale alla sigma WCA, urti fortissimi!)
    for x in range(3):
        for y in range(3):
            initial_centers.append([x*spacing + 0.1, y*spacing + 0.1, 0.45])
    box = np.array([box_size, box_size, box_size])
    system = espressomd.System(box_l=box)
    system.time_step = dt
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
    
    np.random.seed(42)
    initial_velocities = np.random.randn(len(initial_centers), 3) * 0.01 # vel bassa come in check_dt_scaling_cg

    parts = []
    for idx, pos in enumerate(initial_centers):
        pos_wrapped = np.array(pos) % box
        p = system.part.add(pos=pos_wrapped, type=site_type, mass=mass, rinertia=rinertia, rotation=[True,True,True], v=initial_velocities[idx])
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
        device=device
    )
    
    # NESSUNA TERMALIZZAZIONE! Per testare l'errore di integrazione O(dt^2) 
    # dobbiamo partire ESATTAMENTE dallo stesso microstato (stesse posizioni, stesse velocità).
    # Qualsiasi termostato (Langevin) dipendente dal dt altererebbe il microstato iniziale.
    system.thermostat.turn_off()
    
    # Produzione NVE con tempo fisso
    t_total = 0.2 # ps (velocizzato per il test dello scaling)
    steps_nve = max(10, int(round(t_total / dt)))
    print(f"       -> Esecuzione per {steps_nve} steps (t_tot = {t_total} ps)...")
    
    times = []
    e_tots = []
    e_kins = []
    e_pots = []
    
    for i in range(steps_nve):
        system.integrator.run(1)
        
        energies = system.analysis.energy()
        e_kin = energies['kinetic']
        e_wca = energies['total'] - e_kin
        e_painn = espressomd.painn.get_painn_energy()
        
        e_pot = e_wca + e_painn
        e_tot = e_kin + e_pot
        
        times.append(system.time)
        e_tots.append(e_tot)
        e_kins.append(e_kin)
        e_pots.append(e_pot)
        
    e_tots = np.array(e_tots)
    delta_e = np.std(e_tots) # Fluttuazione
    
    return np.array(times), e_tots, e_kins, e_pots, delta_e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rb", type=str, required=True)
    parser.add_argument("--priors", type=str, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    if args.dt is not None:
        times, e_tots, e_kins, e_pots, dE = run_nve(args.model, args.config, args.rb, args.priors, args.bin, args.dt, args.device)
        np.savez(f"nve_dt_{args.dt:.4f}.npz", times=times, e_tots=e_tots, e_kins=e_kins, e_pots=e_pots, dE=dE)
        sys.stdout.flush()
        import os; os._exit(0)
    
    import subprocess
    # dt_list = [0.0001, 0.001, 0.002, 0.004, 0.006] # in ps (0.1, 1, 2, 4, 6 fs)
    dt_list = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]
    delta_e_list = []
    
    plt.figure(figsize=(14, 5))
    target_times, target_e_tot, target_e_kin, target_e_pot = None, None, None, None
    
    for dt in dt_list:
        cmd = [
            sys.executable, sys.argv[0],
            "--bin", args.bin, "--model", args.model,
            "--config", args.config, "--rb", args.rb,
            "--priors", args.priors if args.priors else "",
            "--dt", str(dt),
            "--device", args.device
        ]
        if not args.priors:
            cmd.remove("--priors")
            cmd.remove("")
            
        print(f"[MASTER] Eseguo worker per dt = {dt}")
        subprocess.run(cmd, check=True)
        
        data = np.load(f"nve_dt_{dt:.4f}.npz")
        delta_e_list.append(data['dE'])
        
        if np.isclose(dt, 0.002):
            target_times = data['times']
            target_e_tot = data['e_tots']
            target_e_kin = data['e_kins']
            target_e_pot = data['e_pots']

    # Normalizziamo l'energia totale al valore iniziale per il plot temporale
    plt.subplot(1, 2, 1)
    plt.plot(target_times, target_e_tot - target_e_tot[0], label="E Totale (Shiftata)", linewidth=2, color='black')
    plt.plot(target_times, target_e_kin - target_e_kin[0], label="E Cinetica", alpha=0.7)
    plt.plot(target_times, target_e_pot - target_e_pot[0], label="E Potenziale", alpha=0.7)
    plt.xlabel("Tempo (ps)")
    plt.ylabel(r"$\Delta$ Energia (kJ/mol)")
    plt.title("Conservazione Energia (NVE, dt = 2 fs)")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Scaling log-log
    plt.subplot(1, 2, 2)
    dt_arr = np.array(dt_list) * 1000 # in fs per il plot
    dE_arr = np.array(delta_e_list)
    
    # Fit lineare in spazio log-log
    log_dt = np.log(dt_arr)
    log_dE = np.log(dE_arr)
    coeffs = np.polyfit(log_dt, log_dE, 1)
    slope = coeffs[0]
    
    plt.loglog(dt_arr, dE_arr, 'o-', markersize=8, label=f"Dati Misurati")
    
    # Plot della retta di riferimento quadratica (pendenza = 2) passante per il primo punto
    ref_dE = dE_arr[0] * (dt_arr / dt_arr[0])**2
    plt.loglog(dt_arr, ref_dE, '--', color='red', label="Scaling Teorico (dt$^2$)")
    
    plt.xlabel("Timestep dt (fs)")
    plt.ylabel(r"Fluttuazione E Totale, $\sigma(E)$ (kJ/mol)")
    plt.title(f"Scaling dell'errore (Pendenza misurata: {slope:.2f})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    out_img = "energy_conservation.png"
    plt.savefig(out_img, dpi=300)
    print(f"\n[INFO] Grafico salvato in {out_img}")
    
    sys.stdout.flush()
    import os; os._exit(0)
