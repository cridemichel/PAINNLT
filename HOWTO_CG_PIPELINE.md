# 🚀 Tutorial Completo: Da GROMACS a ESPResSo (Pipeline Coarse-Graining ibrida con Priors)

Questo documento guida passo-passo attraverso l'intero flusso di lavoro: dalla generazione dei dati All-Atom in GROMACS, alla costruzione del modello Coarse-Grained (CG) con Priors Fisici, fino all'addestramento della Rete Neurale PaiNN e alla simulazione finale in ESPResSo (anche su cluster HPC come Leonardo).

---

## 🏗️ Fase 1: Simulazione All-Atom (GROMACS)
Tutto inizia con una traiettoria All-Atom classica. Assicurati che GROMACS salvi le **Forze** oltre alle coordinate.

*   **File richiesti:** 
    *   `topologia.tpr` (o `conf.gro`): Contiene la definizione degli atomi e le masse.
    *   `traiettoria.trr`: Traiettoria ad alta frequenza di salvataggio (es. ogni 10-100 fs) contenente posizioni (`x`) e forze (`f`).

---

## 🗺️ Fase 2: Definizione del Mapping e dei Priors
Prima di convertire i dati per PyTorch, dobbiamo definire come mappare gli atomi (Mapping) e quali vincoli fisici classici sottrarre all'Intelligenza Artificiale (Priors).

### 2a. Il file `cg_mapping.json`
Definisce come raggruppare gli atomi in "Siti Virtuali" e Corpi Rigidi.
```json
{
  "mapping_method": "COM", 
  "site_types": {
      "G_COM": 0,
      "SUGAR": 1,
      "PHOSPHATE": 2
  },
  "residues": {
      "GUA": {
          "G_COM": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
          "SUGAR": ["C1'", "C2'", "C3'", "O3'", "C4'", "O4'", "C5'", "O5'"],
          "PHOSPHATE": ["P", "OP1", "OP2"]
      }
  }
}
```
*Note: `mapping_method` può essere `COM` (Centro di Massa), `COG` (Centro Geometrico) o `ATOM` (Atomo specifico).*

**Esempio 2: Mapping per nucleotidi AAT (G4 DNA Linkers)**
Se vuoi mappare interi nucleotidi Adenina e Timina in singole particelle CG, puoi usare un file come `cg_mapping_aat.json`:
```json
{
    "mapping_method": "COM",
    "site_types": {
        "A_COM": 10,
        "T_COM": 11
    },
    "residues": {
        "DA": { "A_COM": ["*"] },
        "DT": { "T_COM": ["*"] },
        "A":  { "A_COM": ["*"] },
        "T":  { "T_COM": ["*"] }
    }
}
```
In questo modo ogni residuo `DA` (Deossiadenosina) o `DT` verrà convertito in un'unica particella CG al suo Centro di Massa, e la rete neurale riceverà i `site_type` 10 e 11 per riconoscerli.

### 2b. Il file `priors.json` (Il "Delta-Learning")
Per impedire collisioni (WCA) e mantenere intatti i legami covalenti flessibili, calcoliamo analiticamente queste forze e le *sottraiamo* dal dataset. La rete neurale dovrà così imparare solo le interazioni complesse residue.
```json
{
  "wca": {
      "epsilon": 1.0,
      "sigma": 0.3
  },
  "bonds": [
      {
          "type": "harmonic",
          "mol_i": 0,
          "mol_j": 1,
          "k": 1000.0,
          "r0": 0.2
      },
      {
          "type": "morse",
          "mol_i": 1,
          "mol_j": 2,
          "D": 400.0,
          "a": 15.0,
          "r0": 0.2
      }
  ]
}
```

---

## 💾 Fase 3: Generazione del Dataset
Usa lo script Python `convert_gro2bin.py` per leggere la topologia e la traiettoria, calcolare le posizioni CG (risolvendo le condizioni periodiche tramite Minimum Image Convention), sottrarre le forze Priors classiche (Delta-Learning) e scrivere il tutto in un dataset binario altamente compresso.

```bash
python python_scripts/convert_gro2bin.py \
    -c GROMACS/conf.gro \
    -f GROMACS/traiettoria.trr \
    -m python_scripts/cg_mapping.json \
    -p GROMACS/priors.json \
    -o cg_dataset.bin
```

**Parametri dello script:**
*   `-c / --topology`: Il file di topologia (`.tpr` o `.gro`). Si consiglia caldamente di usare un `.gro` (es. `conf.gro`) perché l'attuale parser di `MDAnalysis` potrebbe fallire nel leggere versioni troppo recenti di GROMACS TPR (es. versione 138).
*   `-f / --trajectory`: La traiettoria contenente posizioni e **forze** (`.trr`).
*   `-m / --mapping`: Il file JSON di mapping (es. `cg_mapping.json` o `cg_mapping_aat.json` discusso sopra). Lo script è completamente generale e si adatterà automaticamente al tuo JSON.
*   `-p / --priors`: (Opzionale) Il file JSON con i priors da sottrarre.
*   `-o / --output`: Il file binario di output per PyTorch.

**Output:**
1.  `cg_dataset.bin`: Il dataset binario per l'addestramento (leggerissimo da caricare in C++).
2.  `rigid_bodies_info.json`: Un file fondamentale generato in automatico che contiene per ogni molecola la `mass_amu` e il tensore d'inerzia `inertia_amu_nm2` (necessari poi per ESPResSo).

---

## 🧠 Fase 4: Addestramento del Modello (LibTorch C++)
L'addestramento è delegato a un'applicazione C++ ottimizzata (PyTorch C++ API / LibTorch).

```bash
./build/cg_painn_train cg_dataset.bin best_cg_model.pt
```
*La rete ignorerà automaticamente le interazioni interne ai corpi rigidi e addestrerà il modello a prevedere Forza Totale e Torques (Momenti Torcenti) sui Centri di Massa (o forze residue se hai sottratto i priors).*
Verrà salvato il file `best_cg_model.pt` (o il nome specificato) e il suo config `best_cg_model_config.json`.

---

## ⚛️ Fase 5: Simulazione Ibrida in ESPResSo
Questo è il culmine del progetto: mettere assieme le forze ML e i Priors classici.
Crea uno script (es. `simulate.py`):

```python
import espressomd
import espressomd.painn
import espressomd.interactions
import json
import numpy as np

system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.time_step = 0.001

# 1. Caricamento della topologia generata nello Step 3
with open("rigid_bodies_info.json") as f:
    rb_info = json.load(f)

# (Esempio) Creazione dei corpi rigidi (Particella Centrale + Virtual Sites)
center_parts = []
for idx, (resname, data) in enumerate(rb_info.items()):
    # Estrarre il tipo di particella atteso dalla Rete Neurale
    site_name = list(data["sites"].keys())[0]
    site_type = data["sites"][site_name]["type"]

    # Creiamo il Centro di Massa (Corpo Rigido reale che risponde a Newton/Euler)
    center = system.part.add(
        pos=[2.0 * idx, 5.0, 5.0], type=site_type,
        mass=data["mass_amu"], rinertia=data["inertia_amu_nm2"], rotation=[True, True, True]
    )
    center_parts.append(center)
    
    # Colleghiamo i Virtual Sites
    for s_name, s_data in data["sites"].items():
        v_part = system.part.add(pos=center.pos + s_data["relative_pos_nm"], type=s_data["type"], virtual=True)
        v_part.vs_auto_relate_to(center.id)

# 2. Ripristino dei Priors Esatti usati in addestramento
with open("priors.json") as f:
    priors = json.load(f)

if "wca" in priors:
    # Aggiungiamo WCA tra tutte le particelle dello stesso tipo
    eps = priors["wca"]["epsilon"]
    sig = priors["wca"]["sigma"]
    system.non_bonded_inter[site_type, site_type].lennard_jones.set_params(
        epsilon=eps, sigma=sig, cutoff=sig*(2**(1/6)), shift=0.0
    )

if "bonds" in priors:
    for b in priors["bonds"]:
        if b["type"] == "harmonic":
            hb = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
            system.bonded_inter.add(hb)
            system.part.by_id(center_parts[b["mol_i"]].id).add_bond((hb, center_parts[b["mol_j"]].id))
            
        elif b["type"] == "morse":
            mb = espressomd.interactions.MorseBond(eps=b["D"], alpha=b["a"], rmin=b["r0"])
            system.bonded_inter.add(mb)
            system.part.by_id(center_parts[b["mol_i"]].id).add_bond((mb, center_parts[b["mol_j"]].id))

# 3. Attivazione del Modello Neurale
with open("best_cg_model_config.json") as f:
    config = json.load(f)

# Forziamo il neighbor list ad arrivare al cutoff della rete neurale
cutoff = float(config['cutoff'])
system.min_global_cut = cutoff

# IMPORTANTE: ESPResSo passa le coppie di vicini a PaiNN SOLO se il loro cutoff
# di interazione è sufficientemente grande! Siccome WCA ha un cutoff minuscolo (0.33 nm),
# dobbiamo aggiungere un'interazione "fantasma" a energia zero (es. soft_sphere) 
# per costringere il motore C++ a tracciare le particelle fino al cutoff di PaiNN (es. 0.9 nm).
system.non_bonded_inter[site_type, site_type].soft_sphere.set_params(
    a=0.0, n=1, cutoff=cutoff, offset=0.0
)

espressomd.painn.activate_painn_potential(
    model_path="best_cg_model.pt",
    num_species=int(config['num_species']),
    hidden_channels=int(config['hidden_channels']),
    n_layers=int(config['n_layers']),
    num_rbf=int(config['num_rbf']),
    cutoff=float(config['cutoff']),
    device="cuda" # o cpu
)

# 4. Run!
system.thermostat.set_langevin(kT=2.49, gamma=1.0, seed=42)
system.integrator.run(1000)
```

---

## 🖥️ Fase 6: Deploy su CINECA Leonardo (Apptainer/SLURM)
Usa il container Apptainer fornito per avere PyTorch, ESPResSo e CUDA già pronti.

Crea un file `submit.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=painn_espresso
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=TUO_ACCOUNT

# Carica Apptainer
module load apptainer

# Esegui ESPResSo tramite il container, "bindando" la directory corrente
apptainer exec --nv --bind $PWD:/workspace /percorso/del/tuo/container.sif python /workspace/simulate.py
```
Invio del job: `sbatch submit.slurm`
