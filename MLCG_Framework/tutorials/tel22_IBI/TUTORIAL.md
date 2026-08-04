# Tutorial Avanzato: TEL22 G-Quadruplex (Pipeline IBI + ML)

Benvenuto in questo tutorial avanzato! Qui imparerai come generare un modello Coarse-Grained per una molecola complessa come il DNA telomerico umano (TEL22) sfruttando l'architettura ibrida all'avanguardia del framework: **Iterative Boltzmann Inversion (IBI) accoppiata a Reti Neurali Grafiche (PaiNN)**.

Questa cartella (`TEL22_IBI/`) contiene una sequenza di script specificamente progettata per estrarre potenziali tabulati esatti e addestrare il modello ML a predire esclusivamente le forze residue.

---

> [!WARNING]
> **Prerequisiti GROMACS**
>
> Questo tutorial assume che tu abbia già generato i file della traiettoria All-Atom (GROMACS). I file necessari (`md.gro` e `md_whole.trr`) sono pesanti e non vengono inclusi nativamente.
> 
> Se questi file non sono presenti, **devi prima generarli seguendo lo script `01_run_gromacs.sh` contenuto nel tutorial `tel22` originale.**
> Una volta che hai lasciato girare GROMACS in quella cartella, i file verranno automaticamente collegati qui!

---

## Esecuzione della Pipeline IBI

In questa cartella troverai i singoli script numerati da eseguire uno dopo l'altro (o puoi lanciare `run_full_ibi_pipeline.sh` per farli eseguire tutti in sequenza automatica).

### 01_build_dataset.sh
Questo script utilizza il nostro pre-processore per trasformare la traiettoria All-Atom in un binario `tel22_dataset.bin` mappando ogni nucleotide e calcolando la statistica per una *Direct Boltzmann Inversion* (DBI) preliminare, necessaria come starting point per il passo successivo. Genera inoltre il file base `cg_priors.json`.
*(Nota: come configurato in `tel22_topology.json`, le interazioni angolari e diedrali agiranno sui Virtual Sites. Il framework scalerà in automatico forze e torques per riprodurre esattamente la fisica dello scheletro del filamento.)*

### 02_run_ibi.sh
Questo è il cuore dell'approccio tabulato. Lo script lancia il motore matematico `run_ibi_loop.py` che esegue la **Iterative Boltzmann Inversion**:
1. Legge le coordinate e i target.
2. Effettua iterazioni reali di Dinamica Molecolare simulata in ESPResSo.
3. Calcola la divergenza e corregge le curve tramite l'equazione di Henderson.
4. Sovrascrive automaticamente i legami nel file `cg_priors.json` impostando `"type": "tabulated"` e salvando le spline perfette in `ibi_priors/`.

> [!TIP]
> **La Strategia Ibrida Chirurgica (Evitare le Cross-Correlazioni)**
>
> Noterai che lo script `02_run_ibi.sh` è configurato per calcolare le tabelle numeriche (IBI) **esclusivamente ai legami** (`--bonds IBI`), lasciando gli **Angoli e i Diedri intatti** (`--angles DBI --dihedrals DBI`).
> 
> Perché questa scelta? In molecole giganti come il TEL22, ottimizzare iterativamente centinaia di gradi di libertà in contemporanea porta quasi sempre a instabilità numerica e interferenze incrociate.
> Il framework ci permette di adottare un approccio ibrido:
> - **Legami**: Trattati tramite IBI per gestire asimmetrie anarmoniche.
> - **Angoli e Diedri**: Gestiti tramite formule analitiche (DBI). Le loro sottili imperfezioni verranno assorbite in seguito dalla Rete Neurale!
> 
> **Fisica dello Jacobiano**: Quando lo script esegue la statistica DBI iniziale, il framework protegge fisicamente l'inversione applicando il volume dello spazio delle fasi (Jacobiano matematico). Questo si traduce nella divisione dell'istogramma per $r^2$ per i legami e per $\sin(\theta)$ per gli angoli. Questo step previene l'introduzione di bias entropici geometrici e fornisce un *initial guess* termodinamico perfetto.

### 03_subtract_ibi.sh [LA NOVITÀ]
Adesso che abbiamo le curve IBI perfette, richiamiamo `build_cg_dataset.py` passandogli il flag `--priors`. Invece di calcolare la statistica (DBI), lo script caricherà i potenziali tabulati esatti e li sottrarrà per generare il VERO dataset residuo: **`tel22_dataset_ibi_v2.bin`**.

### 04_train_model.sh
Passa il nuovo binario residuo al programma C++. Addestrerà la rete Graph Neural Network in C++ (tramite LibTorch). A differenza dell'approccio DBI classico, qui la rete dovrà fare molta meno fatica, dovendo imparare solo il rumore (le forze residue non lineari), mentre i muri sterici sono gestiti matematicamente dalle tabelle IBI.

### extrapolate_ibi_tables.py [LA SOLUZIONE SOTA]
Prima di avviare la simulazione, è vitale estrapolare le tabelle IBI! Le tabelle estratte coprono solo il dominio campionato (es. 0.1 nm -> 2.9 nm). Se la rete neurale spinge gli atomi anche solo di poco fuori da questo limite, ESPResSo crasherà lanciando un errore `bond broken`. Seguendo le *best practices* (SOTA, come in VOTCA), esegui `../../../espresso/build/pypresso extrapolate_ibi_tables.py` per estendere artificialmente e in modo sicuro le tabelle fino a distanze grandissime ($5.0$ nm) agganciando una molla lineare per le code. Questo garantisce stabilità incondizionata.

### 05_equilibrate.sh
Genera due checkpoint distinti allo stesso timestep della produzione (`0.0001 ps`):
- `equilibrated_priors.npz`, equilibrato con i soli prior, per il controllo priors-only;
- `equilibrated_ml.npz`, ottenuto attivando il modello PaiNN e completando un warmup NVT sotto l'Hamiltoniana finale IBI+ML.

### 06_run_espresso.sh
Carica il modello C++ e parte da `equilibrated_ml.npz`, cioè da una configurazione rilassata sotto la stessa Hamiltoniana usata in produzione. Le tabelle `.dat` e PaiNN vengono sommate in tempo reale.

> [!NOTE]
> Le colonne energia e forza delle interazioni tabulate sono coerenti ai nodi. ESPResSo le interpola però separatamente tra i nodi; per una conservatività matematica esatta servirebbe un interpolatore condiviso nel core.

---

## Analisi e Visualizzazione
Una volta conclusa la simulazione, verrà generato un file di traiettoria Coarse-Grained (`cg_trajectory.vtf`).
Per visualizzarlo o analizzarne lo stacking e la stabilità, puoi utilizzare esattamente gli stessi identici script di analisi termodinamica descritti nel tutorial base `tel22`:
- **`vtf_to_pymol.py`** per l'animazione PDB
- **`analyze_unfolding_exact.py`** per valutare le distanze di raggio di girazione.


## IBI tail extrapolation for TEL22

The IBI loop uses `ibi_extrapolation_config.json`.  Histogram bins with too few
samples are excluded from the update, short internal gaps are bridged, and the
IBI correction is tapered smoothly to zero at the reliable interval edges.
Unsupported bond tails are rebuilt after DBI and after every IBI iteration with
C1-continuous exponential curvature.  The left tail is repulsive and the right
tail is restoring.  The ESPResSo bond table domain is extended to 5.0 nm, while
a restoring guard force is already imposed at 2.6 nm.  Angles keep their exact
physical domain `[0, pi]` and conservative endpoint walls; dihedrals remain
periodic.

Do not run `extrapolate_ibi_tables.py` or another postprocessor on the generated
tables: extrapolation is part of the IBI update and postprocessing would destroy
the energy/force consistency.

Run the synthetic regression test with:

```bash
uv run python test_ibi_extrapolation.py
```
