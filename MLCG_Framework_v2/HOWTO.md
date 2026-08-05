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
cd MLCG_Framework_v2

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
    --trajectory ../GROMACS/ethanol.trr \
    --topology ../GROMACS/ethanol.gro \
    --config topology_config.json \
    --dbi \
    --output cg_dataset.bin
```

**Opzioni supportate da `build_cg_dataset.py`:**
- `-c`, `--topology`: File di topologia (es. `.tpr` o `.gro`).
- `-f`, `--trajectory`: File di traiettoria (es. `.trr` o `.xtc`).
- `-j`, `--config`: File JSON con topologia CG e regole di mapping (default: `topology_config.json`).
- `--dbi`: Abilita la Direct Boltzmann Inversion (DBI) globale per estrarre le costanti di forza dei legami direttamente dalle distribuzioni termiche.
- `-p`, `--priors`: File JSON con i prior (es. `cg_priors.json`). Se fornito, lo script salta il calcolo statistico e applica direttamente i prior indicati al dataset. Necessario in fase di IBI.
- `-o`, `--output`: Nome del file binario di output (default: `../training/cg_dataset.bin`).

#### Esempio di `topology_config.json`
Il file di configurazione controlla temperature, potenziali WCA, legami a molla (priors) e le regole di mapping (Multi-Bead, COM, ATOM, COG):

```json
{
    "temperature": 300.0,
    "wca_sigma": 0.0,
    "wca_epsilon": 0.0,
    "wca_overrides": [
        {"type_i": 2, "type_j": 2, "sigma": 0.8, "epsilon": 2.5}
    ],
    "bonds": [
        [0, 1]
    ],
    "angles": [
        {"mol_i": 0, "mol_j": 1, "mol_k": 2, "site_i": 0, "site_j": 0, "site_k": 0, "theta0": "auto", "k": "auto"}
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
    },
    "rigid_bodies": {
        "ETH": {
            "auto_align_sites": true,
            "sites": {
                "CG_CH3": {"type": 0, "relative_pos_nm": [0.0, 0.0, 0.0]},
                "CG_CH2": {"type": 1, "relative_pos_nm": [0.15, 0.0, 0.0]},
                "CG_OH":  {"type": 2, "relative_pos_nm": [0.25, 0.1, 0.0]}
            }
        }
    }
}
```

> [!IMPORTANT]
> **Corpi Rigidi e Auto-Allineamento (Algoritmo di Kabsch)**
> Se definisci `"rigid_bodies"` nella tua topologia, lo script mapperà quelle molecole come Corpi Rigidi in ESPResSo (composti da una particella centrale con massa/inerzia reali e siti virtuali).
> 
> Di default (`"auto_align_sites": true`), lo script `build_cg_dataset.py` calcola la geometria "media" ideale di questi siti estraendo tutti gli snapshot dalla traiettoria GROMACS e allineandoli usando l'**Algoritmo di Kabsch**. Le geometrie risultanti vengono salvate in `rigid_bodies_info.json`.
> Durante la generazione del dataset (Pass 2), lo script usa esattamente questa rotazione di Kabsch per ricostruire matematicamente i siti rigidi ideali su ogni frame prima di valutare le forze prior (WCA/DBI/Harmonic). Questo garantisce una **rigorosa consistenza fisica** con ESPResSo (che non deforma i corpi rigidi), impedendo al modello ML di imparare contro-forze enormi e non fisiche causate dalle vibrazioni termiche istantanee.
> 
> Se preferisci fornire manualmente le coordinate ideali perfette (es. da un PDB) e non vuoi che lo script le sovrascriva con la media della traiettoria, imposta `"auto_align_sites": false`. Lo script utilizzerà esattamente le `relative_pos_nm` che hai digitato nel JSON!


> [!WARNING]
> **Rigenerazione obbligatoria dopo questa patch**
> I vecchi `rigid_bodies_info.json` non dichiarano il frame degli assi principali e i vecchi checkpoint non contengono provenienza. Rigenera nell'ordine dataset, `rigid_bodies_info.json`, modello/manifest e checkpoint. Gli override legacy servono solo a diagnosi controllate, non alla certificazione NVE.

### 1.2 Architettura termodinamica (prior analitici + ML residuale)

Il framework v2 usa una decomposizione esplicita dell'Hamiltoniana:

```text
U_tot = U_priors + U_PaiNN
```

Le forze dei prior vengono sottratte dai target durante il preprocessing e gli stessi prior vengono ricreati in ESPResSo. Il plugin PaiNN applica il gradiente esatto dell'energia della rete, senza clipping nascosto di energia o forza. `toxvaerd_alpha` fa parte dell'architettura e deve coincidere tra training e runtime.

#### Corpi rigidi e frame principale

`rigid_bodies_info.json` salva i momenti principali d'inerzia e le coordinate dei virtual site nello stesso frame degli assi principali. All'avvio, lo script ricostruisce la quaternion del COM allineando la geometria body-frame alla configurazione iniziale. I virtual site hanno massa numerica `1e-5`, mentre massa e inerzia fisiche restano sul COM.

#### WCA e PBC

Il mixing WCA usa Lorentz-Berthelot. Quando `wca_sigma` è `"auto"`, le distanze minime vengono calcolate con la minimum-image convention, inclusi i contatti attraverso le facce periodiche.

#### Cutoff PaiNN

La base radiale usa il cutoff Toxvaerd implementato in `PaiNN_Architecture.hpp`. Non esistono opzioni runtime `use_bias` o `apply_envelope`: training, parity e plugin condividono lo stesso header e la stessa parametrizzazione.

#### Fisica Avanzata: Priors Site-Dependent
Di default, i legami Armonici, Angoli e Diedri agiscono sui Centri di Massa. Tuttavia, puoi applicarli a Virtual Sites specifici usando i parametri `site_i`, `site_j`, `site_k`, `site_l` (0-indexed rispetto alla definizione del mapping della molecola).
Quando applicate ai Virtual Sites, le forze sono geometricamente esatte e il framework calcola automaticamente il **momento torcente** $\tau = \vec{r}_{site} \times \vec{F}_{site}$ per trasferire il momento rotazionale al Centro di Massa principale.



> [!TIP]
> **Auto-calcolo della Distanza di Equilibrio (`r0`)**
> Per i legami espliciti (FENE, Morse, ecc.), se ometti il parametro numerico e imposti `"r0": "auto"`, lo script estrarrà automaticamente la distanza media esatta per quella coppia di atomi direttamente dalla traiettoria molecolare! Questo previene esplosioni termodinamiche e risolve elegantemente i mismatch di scala tra all-atom e coarse-grained.

> [!NOTE]
> **Morse e tabelle in NVE**
> Morse viene ancora rappresentato come `TabulatedDistance`. Poiché ESPResSo interpola separatamente energia e forza, `run_cg_md.py --nve` rifiuta Morse e altri prior tabulati per default. L'override `--allow_nonconservative_tables` è destinato esclusivamente a test diagnostici e non certifica lo scaling energetico.

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

> [!TIP]
> **Manifest del modello**
> `train_painn` scrive `<modello>.manifest.json` con architettura effettiva, split e dimensioni degli input. Per aggiungere anche SHA-256 di modello, dataset e config, esegui `python3 training/create_model_manifest.py --model MODEL.pt --config CONFIG.json --dataset DATASET.bin`. Equilibrazione, produzione e parity validano il manifest prima di caricare i pesi.

> [!TIP]
> **Regolarizzazione di Lipschitz**
> Nel file `.json` puoi aggiungere o modificare il parametro `"lipschitz_lambda": 0.001`. Questo parametro introduce una penalità L2 sulla magnitudine delle forze durante l'addestramento (ispirata a CGnet). Attivandola, il modello imparerà a prevedere superfici di energia più dolci, prevenendo gradienti enormi ed esplosioni durante la simulazione in ESPResSo. Se lo imposti a `0.0`, l'overhead aggiuntivo viene completamente bypassato garantendo la totale retrocompatibilità.

### 2.3 Esecuzione del Training
Lancia l'addestramento. Di default, il programma cercherà i file `cg_dataset.bin`, `best_cg_model.pt` (per il salvataggio) e `cg_model_config.json` (per la configurazione).
```bash
cd training
./train_painn
```
*Nota: Puoi passare percorsi personalizzati da riga di comando:*
`./train_painn <dataset.bin> <output_model.pt> <config.json> [--resume]`

Il training salva un archivio LibTorch dei pesi PaiNN, caricato dalla stessa architettura C++ usata dal plugin ESPResSo.

Il trainer non riprende implicitamente un file esistente: usa un nuovo percorso o elimina il vecchio modello per un training pulito. `--resume` è esplicito e richiede un manifest compatibile; dopo ogni training esegui nuovamente `create_model_manifest.py` per aggiornare gli hash.

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
- Il Morse è realizzato come legame tabulato e non riceve alcun force cap automatico in produzione. `run_cg_md.py --nve` lo rifiuta per default perché energia e forza tabulate non costituiscono una certificazione conservativa; l'override è solo diagnostico.

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

### 3.2 Equilibrazione
Prima di avviare la simulazione di produzione, è fondamentale rilassare il sistema per rimuovere eventuali "clash" sterici (compenetrazioni tra atomi/bead) derivanti dalla topologia iniziale, specialmente quando si usano potenziali repulsivi rigidi.

Per l'equilibrazione, usa lo script `equilibrate.py`. Lo script esegue una procedura in più fasi:
1. **Steepest Descent classico**: rilassamento con WCA e prior analitici.
2. **NVT classica capped**: warm-up Langevin con force cap globale progressivo.
3. **NVT ML capped**: attivazione PaiNN con force cap globale progressivo.
4. **NVT ML uncapped**: equilibration finale con esattamente l'Hamiltoniana di produzione e force cap disattivato.

```bash
python equilibrate.py \
    --model best_cg_model.pt \
    --config best_cg_model_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset cg_dataset.bin \
    --out_checkpoint equilibrated.npz \
    --dt 0.002 \
    --kT 2.49 \
    --steps_sd 5000 \
    --steps_md 2000 \
    --device auto
```
**Opzioni supportate da `equilibrate.py`:**
- `--model`, `--config`, `--priors`, `--rb_info`, `--dataset`: File di input richiesti.
- `--out_checkpoint`: checkpoint versionato con stato dinamico, box, identità delle particelle e hash SHA-256 degli input.
- `--dt`: Time-step per la fase di MD (default: 0.002 ps).
- `--kT`: Temperatura in kJ/mol (default: 2.49 per 300K).
- `--steps_sd`: Numero di passi per la fase 1 di Steepest Descent (default: 5000).
- `--steps_md`: Numero di passi per la fase 2 di warm-up classico capped.
- `--steps_ml_capped`: Passi NVT con PaiNN e force cap globale.
- `--steps_ml_uncapped`: Passi NVT finali senza force cap; questi definiscono il checkpoint produttivo.
- `--warmup_chunk`: Intervallo dei messaggi di avanzamento.
- `--allow_missing_model_manifest`: override esplicito per modelli legacy.
- `--allow_unsafe_mpi`: abilita soltanto esperimenti MPI non certificati.
- `--device`: Dispositivo per PyTorch (`cpu`, `cuda`, `mps`, `auto`).

### 3.3 Esecuzione della Dinamica di Produzione
Troverai lo script template `run_cg_md.py` nella cartella `simulation/`.

L'integrazione ML+Priors è gestita elegantemente nel framework. La rete neurale (Plugin C++) si occupa **esclusivamente** della predizione complessa. I Priors (WCA, Harmonic, FENE, Morse) sono aggiunti nativamente nel motore MD di ESPResSo.

Per simulare, usa lo script `run_cg_md.py` che caricherà le coordinate equilibrate dal checkpoint e avvierà la produzione:

```bash
python run_cg_md.py \
    --model best_cg_model.pt \
    --config best_cg_model_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset cg_dataset.bin \
    --checkpoint equilibrated.npz \
    --steps 10000 \
    --dt 0.002 \
    --kT 2.49 \
    --device auto
```

**Opzioni supportate da `run_cg_md.py`:**
- `--model`, `--config`, `--priors`, `--rb_info`, `--dataset`: File di input richiesti.
- `--checkpoint`: File `.npz` prodotto da `equilibrate.py` contenente coordinate e velocità di partenza. Se omesso, partirà dalle coordinate del frame 0 del dataset.
- `--steps`: Numero di passi di simulazione (default: 10000).
- `--dt`: Time-step in picosecondi (default: 0.002 ps).
- `--kT`: Temperatura in kJ/mol (default: 2.49).
- `--device`: Dispositivo per PyTorch.
- `--nve`: Esegue la simulazione nell'ensemble NVE (nessun termostato).
- `--allow_missing_model_manifest`: override esplicito per modelli legacy senza manifest.
- `--allow_legacy_checkpoint` / `--allow_checkpoint_mismatch`: override espliciti per checkpoint legacy o non coerenti.
- `--allow_unsafe_mpi`: abilita soltanto test sperimentali multi-rank; il percorso PaiNN MPI non è certificato.
- `--allow_nonconservative_tables`: consente Morse/tabelle in NVE solo come diagnostica.

> [!TIP]
> **Provenienza del checkpoint**
> I checkpoint patchati contengono hash SHA-256 di dataset, modello, configurazione, prior e rigid-body info, oltre a box e identità delle particelle. Un mismatch interrompe il run prima dell'integrazione, salvo override esplicito.

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

### 3.5 Conservatività e limiti del plugin C++

Il plugin non applica clipping nascosto: la forza PaiNN è il gradiente della stessa energia riportata. Il caricamento del checkpoint PyTorch è fail-fast. Il percorso validato è single-rank; gli script rifiutano per default esecuzioni PaiNN multi-rank perché la comunicazione many-body dell'halo e la contabilizzazione energetica MPI richiedono ancora una parity dedicata 1/2/4 rank.

### 4. Validazione dell'Energia (Scaling Quadratico)
Per assicurarti che l'integrazione di PyTorch e dei Prior all'interno di ESPResSo conservi l'energia (simulazione NVE simplettica), puoi usare lo script di test dedicato:
```bash
cd simulation
/path/to/espresso/build/pypresso verify_energy_scaling.py
```
Lo script del tutorial salva ogni serie energetica, rimuove il drift lineare, stima l'autocorrelazione, usa un moving-block bootstrap e riporta pendenza, intervallo di confidenza, $R^2$, drift e rapporti tra timestep successivi. Per Velocity-Verlet ci si attende una pendenza prossima a 2 e, dimezzando `dt`, una deviazione circa quattro volte più piccola.
