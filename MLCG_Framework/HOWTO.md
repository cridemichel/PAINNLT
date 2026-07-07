# MLCG Framework - Manuale d'Uso

Benvenuto nel **Machine Learning Coarse-Graining (MLCG) Framework**!
Questa pipeline è progettata per estrarre interazioni Coarse-Grained (CG) da traiettorie molecolari All-Atom (AA) utilizzando la *Direct Boltzmann Inversion* per calcolare i prior termodinamici e reti neurali Graph (PaiNN) in C++ (PyTorch) per fittare le forze residue.

La struttura è divisa in tre fasi principali, ognuna contenuta nella sua rispettiva cartella:

1. **Preprocessing:** Costruzione del dataset e sottrazione dei prior fisici.
2. **Training:** Addestramento della rete neurale C++.
3. **Simulation:** Esecuzione della MD Coarse-Grained in ESPResSo.

---

## Fase 0: Setup dell'Ambiente (Self-Contained)

Per garantire la massima riproducibilità, è consigliato creare un ambiente virtuale Python dedicato per questo framework e installare i pacchetti necessari, come `MDAnalysis` e `numpy`.

```bash
# Entra nella cartella del framework
cd MLCG_Framework

# Crea un ambiente virtuale chiamato "mlcg_venv"
python3 -m venv mlcg_venv

# Attiva l'ambiente virtuale
source mlcg_venv/bin/activate

# Installa i pacchetti necessari
pip install -r requirements.txt
```
*Nota: ricordati di attivare l'ambiente virtuale (`source mlcg_venv/bin/activate`) ogni volta che apri un nuovo terminale prima di lanciare gli script Python del framework.*

---

## Fase 1: Preprocessing e Inversione di Boltzmann

Tutto il codice per generare il dataset di addestramento si trova in `preprocessing/`.

### 1.1 Configurare la Topologia
Prima di lanciare lo script, devi definire come gli atomi All-Atom vengono mappati sui siti Coarse-Grained e quali siti CG sono legati covalentemente.
Apri il file `preprocessing/topology_config.json` e definisci:
- `temperature`: La temperatura (in Kelvin) della tua MD All-Atom (necessaria per calcolare $k_B T$).
- `mapping`: Liste di indici degli atomi AA (0-indexed) che compongono ciascun sito CG.
- `bonds`: Coppie di indici di siti CG che sono legati da un prior armonico.
- `wca_sigma` / `wca_epsilon`: Parametri globali (opzionali) per il prior di volume escluso (WCA).

### 1.2 Generare il Dataset
Una volta configurato, lancia lo script fornendo in input la tua traiettoria e la tua topologia GROMACS/All-Atom:
```bash
cd preprocessing
python3 build_cg_dataset.py --traj /path/to/traj.xtc --top /path/to/conf.gro
```
Lo script farà tre cose:
1. Calcolerà le costanti elastiche ottimali ($k, r_0$) usando la Boltzmann Inversion e le salverà in `cg_priors.json`.
2. Sottrarrà analiticamente queste forze armoniche (e il WCA) dalle forze CG mappate, generando il dataset sui residui `cg_dataset.bin` in `training/`.
3. Calcolerà le masse e i tensori d'inerzia per i siti/molecole CG e li salverà in `rigid_bodies_info.json`.

---

## Fase 2: Training della Rete Neurale (C++)

Una volta che il dataset `cg_dataset.bin` è stato generato, l'addestramento sui *residui* avviene puramente in C++.

### 2.1 Compilazione
Assicurati di aver scaricato LibTorch e compilato il binario:
```bash
cd training
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j4
```

### 2.2 Configurare i Parametri della Rete
Nella cartella `training/` troverai il file `cg_model_config.json`. Questo file contiene gli iperparametri della rete PaiNN (es. numero di layer, cutoff radiale, features, numero di epoche). Assicurati di modificarlo secondo le necessità del tuo sistema.

### 2.3 Esecuzione del Training
Lancia l'addestramento. Il programma cercherà automaticamente il file `cg_dataset.bin`.
```bash
cd training
./train_painn
```
Il training salverà il modello PyTorch JIT compilato come `best_cg_model.pt`.

---

## Fase 3: Integrazione e Simulazione in ESPResSo

L'ultima fase è l'utilizzo del modello e dei prior per la produzione MD. Per fare questo, devi prima installare il plugin C++ di PaiNN all'interno del codice sorgente di ESPResSo.

### 3.1 Installazione del Plugin in ESPResSo
Nella cartella `simulation/espresso_plugin/` troverai i 3 file necessari per l'integrazione.
Copia questi file nel codice sorgente di ESPResSo:

```bash
# Sostituisci "/path/to/espresso" con la tua directory dei sorgenti di ESPResSo
cp simulation/espresso_plugin/PaiNN_ML_Potential.cpp /path/to/espresso/src/core/machine_learning/
cp simulation/espresso_plugin/PaiNN_ML_Potential.hpp /path/to/espresso/src/core/machine_learning/
cp simulation/espresso_plugin/painn.pyx /path/to/espresso/src/python/espressomd/
```

Dopodiché, ricompila ESPResSo assicurandoti di avere PyTorch abilitato nella tua toolchain Cmake:
```bash
cd /path/to/espresso/build
make -j4
```

### 3.2 Eseguire la Dinamica
Troverai lo script template `run_cg_md.py` nella cartella `simulation/`.

```bash
cd simulation
/path/to/espresso/build/pypresso run_cg_md.py
```
Lo script si occuperà di:
1. Caricare le masse e i tensori d'inerzia dal file `rigid_bodies_info.json` (generato allo Step 1) per configurare le particelle in ESPResSo.
2. Caricare i prior topologici dal file `cg_priors.json` e configurare i legami armonici nativi di ESPResSo.
3. Caricare il WCA in ESPResSo.
4. Inizializzare la rete neurale `best_cg_model.pt` in ESPResSo tramite il plugin C++ ML Potential.
5. Lanciare la Dinamica Molecolare NVE o NVT (Langevin) combinando le forze analitiche con le predizioni neurali!
