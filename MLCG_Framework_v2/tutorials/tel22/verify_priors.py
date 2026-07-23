import espressomd
import espressomd.interactions
import numpy as np
import json
import os

# 1. Carica configurazione iniziale
chk = np.load("equilibrated.npz")
pos = chk["pos"]

with open("tel22_training_config.json", "r") as f:
    nn_config = json.load(f)
box_l = nn_config.get("box_l", [10.0, 10.0, 10.0]) # fallback se non c'è, in tel22 è noto

# 2. Inizializza ESPResSo
system = espressomd.System(box_l=box_l)
system.time_step = 0.01
system.cell_system.skin = 0.4

# Crea particelle (assumiamo tel22: 219 particelle, tutte reali per semplicità di test, 
# oppure leggiamo rigid_bodies_info)
# Più semplice: copiamo run_cg_md.py logica
with open("tel22_training_config.json", "r") as f:
    nn_config = json.load(f)
with open("cg_priors.json", "r") as f:
    priors = json.load(f)
with open("rigid_bodies_info.json", "r") as f:
    rb_info = json.load(f)

# Troviamo il mapping dai priors
mol_types = []
for m in priors.get("molecules", []):
    mol_types.append((m["type"], m.get("resname", "")))

# Aggiungi particelle
for i in range(len(pos)):
    # type fallback a 0, non è importante per i bond armonici, ma lo è per WCA
    # Mappiamo i tipi basandoci su mol_types se disponibili (assumiamo atomi singoli per semplicità)
    t = mol_types[i][0] if i < len(mol_types) else 0
    system.part.add(id=i, pos=pos[i], type=t)

# Aggiungi WCA
import math
wca = priors.get("wca", {})
has_wca = wca.get("sigma", 0.0) > 0 or len(wca.get("overrides", {})) > 0
if wca.get("epsilon", 0.0) > 0 and has_wca:
    for i in range(nn_config["num_species"]):
        sigma_i = wca.get("overrides", {}).get(str(i), {}).get("sigma", wca["sigma"])
        eps_i = wca.get("overrides", {}).get(str(i), {}).get("epsilon", wca["epsilon"])
        for j in range(i, nn_config["num_species"]):
            sigma_j = wca.get("overrides", {}).get(str(j), {}).get("sigma", wca["sigma"])
            eps_j = wca.get("overrides", {}).get(str(j), {}).get("epsilon", wca["epsilon"])
            sigma_mix = (sigma_i + sigma_j) / 2.0
            eps_mix = math.sqrt(eps_i * eps_j)
            system.non_bonded_inter[i, j].lennard_jones.set_params(
                epsilon=eps_mix, sigma=sigma_mix,
                cutoff=sigma_mix * (2.0**(1/6)), shift="auto"
            )

# Esclusioni WCA intra-molecolari
wca_exclusions = set()
for b in priors.get("bonds", []):
    m1, m2 = min(b["mol_i"], b["mol_j"]), max(b["mol_i"], b["mol_j"])
    wca_exclusions.add((m1, m2))
for a in priors.get("angles", []):
    m1, m2 = min(a["mol_i"], a["mol_k"]), max(a["mol_i"], a["mol_k"])
    wca_exclusions.add((m1, m2))

for (m1, m2) in wca_exclusions:
    try:
        system.part.by_id(m1).add_exclusion(m2)
    except Exception:
        pass

# Aggiungi Legami Armonici
for b in priors.get("bonds", []):
    if b.get("type", "harmonic") == "harmonic":
        bond = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
        system.bonded_inter.add(bond)
        system.part.by_id(b["mol_j"]).add_bond((bond, b["mol_i"]))

# Aggiungi Angoli Armonici
for a in priors.get("angles", []):
    if a.get("type", "harmonic") == "harmonic":
        angle = espressomd.interactions.AngleHarmonic(bend=a["k"], phi0=a["theta0"])
        system.bonded_inter.add(angle)
        # ESPResSo angle bond: applied to central particle, with arguments (p1, p3)
        system.part.by_id(a["mol_j"]).add_bond((angle, a["mol_i"], a["mol_k"]))

# Calcola forze ESPResSo
system.integrator.run(0)
f_espresso = np.array([p.f for p in system.part])

# =========================================================
# Calcola forze Python (stesso codice di build_cg_dataset.py)
# =========================================================
f_python = np.zeros_like(pos)

def mic_vector(p1, p2, box):
    d = p1 - p2
    d = d - box * np.round(d / box)
    return d

# WCA Python
if wca.get("epsilon", 0.0) > 0 and has_wca:
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            if (min(i, j), max(i, j)) in wca_exclusions:
                continue
            r_vec = mic_vector(pos[i], pos[j], box_l)
            r = np.linalg.norm(r_vec)
            # Parametri mix (assumendo tipi tutti uguali a 0 per semplicità o leggendo da wca)
            t_i = mol_types[i][0] if i < len(mol_types) else 0
            t_j = mol_types[j][0] if j < len(mol_types) else 0
            sigma_i = wca.get("overrides", {}).get(str(t_i), {}).get("sigma", wca["sigma"])
            eps_i = wca.get("overrides", {}).get(str(t_i), {}).get("epsilon", wca["epsilon"])
            sigma_j = wca.get("overrides", {}).get(str(t_j), {}).get("sigma", wca["sigma"])
            eps_j = wca.get("overrides", {}).get(str(t_j), {}).get("epsilon", wca["epsilon"])
            
            s_ij = (sigma_i + sigma_j) / 2.0
            e_ij = math.sqrt(eps_i * eps_j)
            
            r_cut = s_ij * (2.0**(1/6))
            if r < r_cut and r > 1e-6:
                f_scalar = 24.0 * e_ij * (2.0 * (s_ij/r)**12 - (s_ij/r)**6) / r
                f_vec = f_scalar * (r_vec / r)
                # Forza di i su j: i viene spinto indietro, j spinto in avanti
                # r_vec è pos_i - pos_j
                # Quindi repulsione: pos_i viene spinto lungo r_vec, pos_j lungo -r_vec
                f_python[i] += f_vec
                f_python[j] -= f_vec

# Bond Python
for b in priors.get("bonds", []):
    if b.get("type", "harmonic") == "harmonic":
        i, j = b["mol_i"], b["mol_j"]
        r_vec = mic_vector(pos[i], pos[j], box_l)
        r = np.linalg.norm(r_vec)
        if r > 1e-6:
            f_scalar = - b["k"] * (r - b["r0"])
            f_vec = - f_scalar * (r_vec / r)
            f_python[i] += f_vec
            f_python[j] -= f_vec

# Angle Python
for a in priors.get("angles", []):
    if a.get("type", "harmonic") == "harmonic":
        i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
        r_ji = mic_vector(pos[j], pos[i], box_l)
        r_jk = mic_vector(pos[j], pos[k], box_l)
        d_ji = np.linalg.norm(r_ji)
        d_jk = np.linalg.norm(r_jk)
        if d_ji > 1e-6 and d_jk > 1e-6:
            cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            if sin_theta > 1e-6:
                dV_dtheta = a["k"] * (theta - a["theta0"])
                grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji**2)
                grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk**2)
                
                scalar_force = dV_dtheta / sin_theta
                f_i = scalar_force * grad_i_cos
                f_k = scalar_force * grad_k_cos
                f_j = -(f_i + f_k)
                
                f_python[i] += f_i
                f_python[j] += f_j
                f_python[k] += f_k

# 3. Confronto
diff = f_espresso - f_python
max_err = np.max(np.abs(diff))
mae = np.mean(np.abs(diff))

idx_max = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
p_idx, dim = idx_max
print(f"Max Absolute Error between ESPResSo and Python Priors: {max_err:.6e} kJ/mol/nm at atom {p_idx} dim {dim}")
print(f"F_espresso = {f_espresso[p_idx]}, F_python = {f_python[p_idx]}")
print(f"Position: {pos[p_idx]}")
print(f"Mean Absolute Error: {mae:.6e} kJ/mol/nm")

if max_err < 1e-4:
    print("[SUCCESS] Le forze applicate da ESPResSo coincidono ESATTAMENTE con quelle sottratte dal dataset Python!")
else:
    print("[ERROR] Discrepanza rilevata tra ESPResSo e Python!")
