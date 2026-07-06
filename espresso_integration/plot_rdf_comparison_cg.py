import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
import sys
import os

try:
    import espressomd
    import espressomd.painn
    import espressomd.observables
    import json
except ImportError:
    print("[ERRORE] ESPResSo non trovato. Questo script deve essere eseguito nell'ambiente con ESPResSo.")
    sys.exit(1)

def compute_target_rdf_from_bin(bin_file, r_max, n_bins):
    """
    Legge il dataset binario generato da convert_gro2bin.py 
    e calcola la Radial Distribution Function (RDF) target dai Centri di Massa.
    """
    print(f"[INFO] Lettura dataset {bin_file} per calcolare l'RDF target...")
    try:
        f = open(bin_file, "rb")
    except FileNotFoundError:
        print(f"[ERRORE] File '{bin_file}' non trovato.")
        sys.exit(1)

    num_frames = struct.unpack("i", f.read(4))[0]
    
    hist_total = np.zeros(n_bins)
    edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[1:] + edges[:-1])
    dr = r_max / n_bins
    
    total_density = 0.0
    valid_frames = 0
    
    for frame_idx in range(num_frames):
        data = f.read(8)
        if not data: break
        num_molecules, num_total_sites = struct.unpack("ii", data)
        
        box = np.array(struct.unpack("3f", f.read(12)))
        vol = box[0] * box[1] * box[2]
        density = num_molecules / vol
        
        centers = []
        for mol_idx in range(num_molecules):
            mol_id, num_sites = struct.unpack("ii", f.read(8))
            cx, cy, cz, fx, fy, fz, tx, ty, tz = struct.unpack("9f", f.read(36))
            centers.append([cx, cy, cz])
            
            # Salta i siti
            for s in range(num_sites):
                f.read(16)
                
        centers = np.array(centers)
        
        # Calcolo distanze con MIC (Minimum Image Convention) usando NumPy broadcasting
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        diff -= box * np.round(diff / box)
        dist = np.linalg.norm(diff, axis=-1)
        
        # Prendi triangolo superiore (esclude self ed evita doppio conteggio)
        idx = np.triu_indices(num_molecules, k=1)
        dists = dist[idx]
        
        hist, _ = np.histogram(dists, bins=edges)
        hist_total += hist
        total_density += density
        valid_frames += 1
        
        if frame_idx % 50 == 0:
            print(f"       Processato frame {frame_idx}/{num_frames}")

    f.close()
    
    if valid_frames == 0:
        return r_centers, np.zeros(n_bins)
        
    avg_density = total_density / valid_frames
    hist_avg = hist_total / valid_frames
    
    N = num_molecules
    shell_vol = 4.0 * np.pi * (r_centers**2) * dr
    ideal_counts = avg_density * shell_vol
    
    g_r = (2.0 * hist_avg / N) / ideal_counts
    
    return r_centers, g_r

def run_espresso_rdf(model_path, config_path, rb_info_path, priors_path, bin_file, r_max, n_bins, steps=1000):
    print(f"\n[INFO] Avvio simulazione ESPResSo per calcolo RDF...")
    
    # 0. Estraiamo il primo frame dal dataset target per inizializzare
    # ESPResSo nello stesso identico stato di densità e box!
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
                
    print(f"       Box rilevato: {box} nm")
    print(f"       Molecole rilevate: {num_molecules}")
    
    system = espressomd.System(box_l=box)
    system.time_step = 0.002
    system.cell_system.skin = 0.05
    system.cell_system.set_n_square() # Evita crash se cutoff + skin > box/2 in piccoli sistemi
    
    # Setup particelle e PaiNN
    with open(rb_info_path, "r") as f:
        rb_info = json.load(f)
        
    resname = list(rb_info.keys())[0] # Usiamo il primo resname
    data = rb_info[resname]
    mass = data["mass_amu"]
    rinertia = data["inertia_amu_nm2"]
    
    # Estraiamo il tipo esatto (es. 11 per CG_WAT) per l'embedding di PyTorch
    site_name = list(data["sites"].keys())[0]
    site_type = data["sites"][site_name]["type"]
    
    parts = []
    for idx, pos in enumerate(initial_centers):
        # Assicuriamoci che i COM siano wrapped nella box
        pos_wrapped = np.array(pos) % box
        p = system.part.add(pos=pos_wrapped, type=site_type, mass=mass, rinertia=rinertia, rotation=[True,True,True])
        parts.append(p)

    # Carica Priors WCA se esistono
    if priors_path and os.path.exists(priors_path):
        with open(priors_path, "r") as f:
            priors = json.load(f)
            if "wca" in priors:
                eps = priors["wca"].get("epsilon", 0)
                sig = priors["wca"].get("sigma", 0)
                if eps > 0 and sig > 0:
                    system.non_bonded_inter[site_type, site_type].lennard_jones.set_params(
                        epsilon=eps, sigma=sig, cutoff=sig*(2**(1/6)), shift=0.0
                    )
    
    with open(config_path, "r") as f:
        nn_config = json.load(f)
        
    cutoff = float(nn_config['cutoff'])
    # Forziamo il neighbor list di ESPResSo ad arrivare al cutoff della rete neurale
    system.min_global_cut = cutoff
    
    # IMPORTANTE: ESPResSo costruisce il neighbor list per una coppia SOLO se il cutoff 
    # della loro interazione è grande abbastanza! WCA ha un cutoff minuscolo (0.33 nm).
    # Per costringere ESPResSo a passarci tutte le distanze fino a `cutoff` (es. 0.9 nm), 
    # aggiungiamo una interazione fantasma a energia zero (soft_sphere con a=0.0):
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
        device="cpu"
    )
    
    system.thermostat.set_langevin(kT=2.49, gamma=1.0, seed=42)
    
    print("[INFO] Equilibrazione termica dal primo frame...")
    for i in range(50):
        system.integrator.run(10)
        e_pot = espressomd.painn.get_painn_energy()
        max_f = max([np.max(np.abs(p.f)) for p in parts])
        print(f"Equilibration step {i*10:3d} | E_pot: {e_pot:.2f} | Max Force: {max_f:.2f} | Pos 0: {parts[0].pos}")
    
    # Assicuriamoci che r_max non superi metà della scatola più piccola
    actual_rmax = min(r_max, min(box) / 2.0)
    if actual_rmax != r_max:
         print(f"[WARNING] Rmax {r_max} supera L/2. Verrà ridotto a {actual_rmax} nm")
    
    rdf_obs = espressomd.observables.RDF(ids1=[p.id for p in parts], 
                                         ids2=[p.id for p in parts], 
                                         min_r=0.0, max_r=actual_rmax, n_r_bins=n_bins)
    
    g_r_sum = np.zeros(n_bins)
    samples = 0
    
    print("[INFO] Produzione e campionamento RDF...")
    for i in range(steps // 10):
        system.integrator.run(10)
        g_r_sum += np.array(rdf_obs.calculate())
        samples += 1
        if i % 10 == 0:
            print(f" -> Campionato frame RDF {i*10}/{steps}...")
        
    g_r_avg = g_r_sum / samples
    r_centers = rdf_obs.bin_centers()
    
    return r_centers, g_r_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, required=True, help="Percorso del dataset binario (.bin)")
    parser.add_argument("--model", type=str, required=True, help="Modello PaiNN (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Config modello (.json)")
    parser.add_argument("--rb", type=str, required=True, help="Rigid bodies info (.json)")
    parser.add_argument("--priors", type=str, default=None, help="Priors (.json)")
    parser.add_argument("--rmax", type=float, default=1.5, help="Rmax per l'RDF in nm")
    parser.add_argument("--bins", type=int, default=100, help="Numero di bin per l'RDF")
    
    args = parser.parse_args()
    
    # Estraiamo box dal file
    with open(args.bin, "rb") as f:
        f.read(12)
        box_l = np.array(struct.unpack("3f", f.read(12)))
        
    r_max = min(args.rmax, min(box_l) / 2.0)
    
    # 1. Target GROMACS
    r_tgt, g_tgt = compute_target_rdf_from_bin(args.bin, r_max, args.bins)
    
    # 2. ESPResSo
    r_esp, g_esp = run_espresso_rdf(args.model, args.config, args.rb, args.priors, args.bin, r_max, args.bins, steps=1000)
    
    # 3. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(r_tgt, g_tgt, label="GROMACS (Target Dataset)", linewidth=2)
    plt.plot(r_esp, g_esp, '--', label="ESPResSo (PaiNN + Priors)", linewidth=2)
    plt.xlabel("Distanza r (nm)")
    plt.ylabel("RDF g(r)")
    plt.title("Confronto Radial Distribution Function (CG)")
    plt.legend()
    plt.grid(True)
    
    out_img = "rdf_comparison.png"
    plt.savefig(out_img, dpi=300)
    print(f"\n[INFO] Grafico salvato in {out_img}")
