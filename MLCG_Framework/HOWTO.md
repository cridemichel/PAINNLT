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

#### Priors e Inversione di Boltzmann (DBI vs IBI)

Il framework supporta due filosofie fondamentali per estrarre le energie a priori dalla traiettoria All-Atom: la **Direct Boltzmann Inversion (DBI)** e l'**Iterative Boltzmann Inversion (IBI)**.

**1. Funzioni Analitiche (DBI, FENE, Morse, Angoli, Diedri)**
Se includi l'array `"bonds"` come liste di indici (es. `[[0, 1]]`), lo script di preprocessing effettuerà una statistica (DBI classica) per ricavare la costante armonica $k$ e la distanza di equilibrio $r_0$.
In alternativa, puoi disattivare l'inferenza automatica e definire esplicitamente parametri analitici molto più complessi per diversi gradi di libertà. Puoi usare:
- **Harmonic Bond** (`"type": "harmonic"`): la classica molla di Hooke.
- **FENE Bond** (`"type": "fene"`): utilissimo per catene polimeriche dove i monomeri non devono allontanarsi oltre un certo $R_{max}$.
- **Morse Bond** (`"type": "morse"`): essenziale per legami non lineari che devono potersi rompere (come lo stacking dei tetrad o i legami idrogeno).
- **Angoli Armonici** (nell'array `"angles"`): per stabilizzare l'angolo tra tre siti.
- **Diedri** (nell'array `"dihedrals"`): per stabilizzare la conformazione torsionale tra quattro siti.
Questo approccio parametrico è ultra-veloce da valutare, ma si basa su equazioni chiuse ideali.

**2. Iterative Boltzmann Inversion (IBI) [Curve Tabulate Esatte]**
Se il tuo sistema è altamente anarmonico o soffre di interferenze incrociate (es. la repulsione sterica modifica le distanze di legame), l'approssimazione armonica della DBI non è sufficiente. In questo caso, puoi usare la pipeline IBI integrata:
- Usa gli script nella cartella `ibi/` per estrarre matematicamente i potenziali esatti. Lo script `run_ibi_loop.py` calcola le curve (spline) e, tramite cicli MD in ESPResSo, le corregge iterativamente finché la distribuzione simulata non matcha matematicamente il target All-Atom.
- Successivamente, usa `generate_residual_dataset.py` per creare un dataset in cui la rete neurale viene addestrata solo sulle forze residue.
- Per simulare, nel tuo file `cg_priors.json`, ti basterà impostare `"type": "tabulated"` e fornire il percorso al file generato dall'IBI:
```json
{
    "mol_i": 0, "mol_j": 1,
    "type": "tabulated",
    "file": "ibi_priors/bond_ibi_final.dat",
    "min": 0.3, "max": 0.7
}
```
ESPResSo leggerà la tabella numerica (sia per legami `TabulatedDistance` che per angoli `TabulatedAngle`) iniettando il potenziale IBI esatto. Questa scelta garantisce una retrocompatibilità nativa e permette di mischiare liberamente molle DBI e tabelle IBI per gradi di libertà diversi!

> [!TIP]
> **Auto-calcolo della Distanza di Equilibrio (`r0`)**
> Per i legami espliciti (FENE, Morse, ecc.), se ometti il parametro numerico e imposti `"r0": "auto"`, lo script estrarrà automaticamente la distanza media esatta per quella coppia di atomi direttamente dalla traiettoria molecolare! Questo previene esplosioni termodinamiche e risolve elegantemente i mismatch di scala tra all-atom e coarse-grained.

> [!NOTE]
> **Potenziale di Morse e Force Capping**
> In ESPResSo, i legami di Morse espliciti sono iniettati sotto il cofano come `TabulatedDistance` (estesi oltre la dimensione del box). Il framework applica automaticamente un **Force Capping** (limite rigido) per prevenire le esplosioni di integrazione causate dal muro repulsivo esponenzialmente ripido quando i monomeri si avvicinano troppo, garantendo il perfetto equilibrio tra la frangibilità del legame e la stabilità termodinamica.

Esempi di definizione esplicita:
```json
"bonds": [
    {
        "mol_i": 0, "mol_j": 1,
        "site_i": 2, "site_j": 0,
        "type": "fene",
        "k": 1000.0,
        "r0": "auto",
        "r_max": 0.3
    },
    {
        "mol_i": 2, "mol_j": 3,
        "type": "morse",
        "D": 20.0,
        "a": 3.0,
        "r0": "auto"
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

## Uso del Potenziale di Morse per le Interazioni di Stacking (Esempio TEL22)
Le reti neurali grafiche a volte faticano a modellare forze non lineari e "fragili" a lungo raggio come l'impilamento (stacking) dei tetrad di Guanina o le forze di Van der Waals intra-catena in modo nativo, specialmente con pochi dati di training. Un modo elegante e veloce per risolvere questo problema è introdurre un **Potenziale di Morse** esplicito come prior.

Il potenziale di Morse modella perfettamente la "buca" energetica dello stacking biologico:
- Offre una profonda stabilità all'equilibrio (regolata dal parametro `D`, profondità della buca).
- Permette all'interazione di "rompersi" dolcemente a distanze maggiori (regolato dal parametro `a` o $\alpha$, larghezza della buca), a differenza dei legami armonici che genererebbero forze infinite impedendo fisicamente fenomeni come il melting termico o l'unfolding.

**Caso d'uso (Tutorial TEL22):**
Nel caso dei G-Quadruplex (TEL22), lo stacking planare tra le guanine è essenziale per la compattezza della struttura. Piuttosto che far imparare questa forza complessa interamente al Modello di Machine Learning, iniettiamo esplicitamente dei legami di Morse "scaffold" (impalcatura) tra le guanine impilate:
```json
{
    "mol_i": 2, "mol_j": 8,
    "type": "morse",
    "D": 50.0,
    "a": 3.0,
    "r0": "auto"
}
```
In questo setup:
- `D` a `50.0` kJ/mol garantisce che la struttura rimanga stabilmente foldata a temperature fisiologiche (300K). Valori più bassi (es. `20.0`) faciliterebbero un unfolding termico visibile.
- `"r0": "auto"` permette al framework di leggere l'esatta distanza di stacking direttamente dalla traiettoria atomistica (evitando esplosioni termodinamiche causate da un `r0` immesso manualmente e non perfettamente allineato con le dimensioni del CG).
- ESPResSo applicherà automaticamente un "Force Capping" su questi legami tabulati per impedire esplosioni di integrazione qualora i monomeri subiscano urti termici severi a brevissima distanza.

## 4. Esecuzione della Simulazione (ESPResSo)

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
