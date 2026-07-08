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
- `bonds`: Coppie di indici o configurazioni FENE/Morse per l'inserimento dei prior.

Questo script supporta il mapping flessibile basato su file JSON. Puoi generare file `.bin` contenenti posizioni, forze e momenti torcenti (torques) sui singoli siti o centri di massa.

Esempio di esecuzione:
```bash
python build_cg_dataset.py \
    --traj ../GROMACS/ethanol.trr \
    --topol ../GROMACS/ethanol.gro \
    --config topology_config.json \
    --output cg_dataset.bin
```

#### Esempio di `topology_config.json`
Il file di configurazione controlla temperature, potenziali WCA, legami a molla (priors) e le regole di mapping (Multi-Bead, COM, ATOM, COG):

```json
{
    "temperature": 300.0,
    "wca_sigma": 0.0,
    "wca_epsilon": 0.0,
    "bonds": [
        [0, 1]
    ],
    "mapping": {
        "mapping_method": "COM",
        "residues": {
            "ETH": {
                "CG_CH3": ["C1", "H1", "H2", "H3"],
                "CG_CH2": ["C2", "H4", "H5"],
                "CG_OH":  ["O1", "H6"]
            }
        },
        "site_types": {
            "CG_CH3": 0, "CG_CH2": 1, "CG_OH": 2
        }
    }
}
```

#### Priors e Inversione di Boltzmann
Se includi l'array `"bonds"` (come `[[0, 1]]`), lo script effettuerà l'**Inversione di Boltzmann** statistica sulle distanze della traiettoria per ricavare la costante armonica $k$ e la distanza di equilibrio $r_0$.
In alternativa, puoi disattivare del tutto i priors lasciando `bonds: []`, oppure definire esplicitamente **FENE** o **Morse** (supporto anche per legami tra siti specifici invece che Centri di Massa):

```json
"bonds": [
    {
        "mol_i": 0, "mol_j": 1,
        "site_i": 2, "site_j": 0,
        "type": "fene",
        "k": 1000.0,
        "r0": 0.2,
        "r_max": 0.3
    }
]
```

Lo script calcolerà le forze risultanti (e i relativi **momenti torcenti**) e le sottrarrà dai target prima di salvare il dataset binario, esportando i parametri esatti nel file `cg_priors.json`.

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
Nella cartella `training/` troverai il file `cg_model_config.json`. Questo file è la **centralina di controllo** della rete: prima di lanciare l'addestramento, puoi modificare qui dentro parametri come `hidden_channels`, `n_layers`, `cutoff`, `learning_rate` e le `epochs`. Il codice C++ leggerà questo file in tempo reale senza bisogno di ricompilare!

### 2.3 Esecuzione del Training
Lancia l'addestramento. Di default, il programma cercherà i file `cg_dataset.bin`, `best_cg_model.pt` (per il salvataggio) e `cg_model_config.json` (per la configurazione).
```bash
cd training
./train_painn
```
*Nota: Puoi passare percorsi personalizzati da riga di comando:*
`./train_painn <dataset.bin> <output_model.pt> <config.json>`

Il training salverà il modello PyTorch JIT compilato e ottimizzato per ESPResSo.

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

L'integrazione ML+Priors è gestita elegantemente nel framework. La rete neurale (Plugin C++) si occupa **esclusivamente** della predizione complessa. I Priors (WCA, Harmonic, FENE, Morse) sono aggiunti nativamente nel motore MD di ESPResSo.

Per simulare, usa lo script `run_cg_md.py` che:
1. Instanzia le molecole e i Virtual Sites.
2. Legge `cg_priors.json` e applica i legami di ESPResSo sui siti/particelle.
3. Attiva il potenziale PaiNN C++ che inietterà la sua forza predetta su ogni sito.

```bash
python run_cg_md.py \
    --model best_model.pt \
    --config best_cg_model_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset cg_dataset.bin \
    --steps 10000 \
    --dt 0.002
```

### 3.4 Dinamica dei Corpi Rigidi e Filtro delle Particelle
Nel framework, la simulazione di molecole a più siti (Multi-Bead) sfrutta i **Virtual Sites** di ESPResSo:
1. **La Particella Reale (Centro di Massa)**: Per ogni molecola rigida, ESPResSo richiede una singola particella centrale dotata di massa e tensore d'inerzia. Questa è l'unica particella che l'integratore muove fisicamente nello spazio.
2. **I Siti Virtuali (Le Bead CG)**: I siti di interazione (es. `CH3`, `OH`) vengono istanziati come particelle senza massa, la cui posizione è rigidamente ancorata alla particella reale. Qualsiasi forza applicata a un sito virtuale viene automaticamente tradotta da ESPResSo in forza netta e **momento torcente (torque)** sulla particella reale.

**Il problema della Rete Neurale:**
La rete neurale PaiNN viene addestrata *esclusivamente* sui siti (es. tipi `0, 1, 2`). Non conosce l'esistenza del "Centro di Massa". Se passassimo il Centro di Massa al Modello ML, quest'ultimo andrebbe in errore cercando di interpretare una specie chimica sconosciuta.

**La Soluzione (Il Filtro `num_species`):**
Il plugin C++ (`PaiNN_ML_Potential.cpp`) include un filtro elegante: accetta in ingresso il parametro `num_species` (il numero totale di tipi noti alla rete neurale). 
Durante la simulazione in ESPResSo:
- Ai siti virtuali vengono assegnati i tipi standard (es. `0`, `1`, `2`). La Rete Neurale li vede, calcola le distanze, e prevede le forze.
- Alla particella "Reale" (il Centro di Massa) viene assegnato intenzionalmente un tipo "fantasma", ovvero un ID maggiore o uguale a `num_species` (ad esempio `type = 100`).
- Il plugin C++ **ignora attivamente** tutte le particelle con `type >= num_species`. 

Il Centro di Massa diventa così "invisibile" al Machine Learning, ma rimane perfettamente attivo per la meccanica di ESPResSo!
Se si desidera far interagire la particella del Centro di Massa:
- **Classicamente (es. WCA tra molecole)**: Basta definire in ESPResSo l'interazione per il `type 100`.
- **Con la Rete Neurale**: Basterà includere il Centro di Massa come un sito esplicito nel `topology_config.json` in fase di preprocessing, addestrare il modello includendolo (avrà un suo `type` come `3`), e assegnargli quel tipo in simulazione.

> [!NOTE]
> **Particelle "Reali" vs "Virtuali" in ESPResSo**
> ESPResSo non usa il `type` per capire se una particella è Reale o Virtuale. Il `type` è solo un'etichetta per la Rete Neurale o per i parametri di Lennard-Jones. La vera natura della particella viene decisa dal flag `virtual=True` o `virtual=False` durante la creazione in Python.
> 
> * Esempio: `system.part.add(pos=..., type=100, virtual=False)` crea la particella reale. Newton la muoverà, ma la rete neurale (fermandosi ai tipi < 100) la ignorerà.
> * Esempio: `system.part.add(pos=..., type=0, virtual=True)` crea un sito fantasma attaccato al corpo rigido. La Rete Neurale lo vedrà, applicherà la forza su di esso, ed ESPResSo farà leva trasferendo forza e momento torcente al corpo rigido reale a cui è legato.

### 4. Validazione dell'Energia (Scaling Quadratico)
Per assicurarti che l'integrazione di PyTorch e dei Prior all'interno di ESPResSo conservi l'energia (simulazione NVE simplettica), puoi usare lo script di test dedicato:
```bash
cd simulation
/path/to/espresso/build/pypresso verify_energy_scaling.py
```
Lo script ridurrà iterativamente il time-step `dt` e calcolerà la deviazione standard dell'energia totale ($E_{kin} + E_{bonded} + E_{ML}$). Dato che l'algoritmo di integrazione è *Velocity Verlet*, l'errore deve scalare con $O(dt^2)$, il che significa che dimezzando il time-step la fluttuazione si ridurrà esattamente di un fattore $\sim 0.25$!
