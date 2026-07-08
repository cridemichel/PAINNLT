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
Questo script utilizza il nostro pre-processore per trasformare la traiettoria All-Atom in un binario `tel22_dataset.bin` mappando ogni nucleotide.
Genera inoltre il file base `cg_priors.json` che stabilisce la topologia strutturale.

### 02_run_ibi.sh [LA NOVITÀ]
Questo è il cuore dell'approccio tabulato. Lo script lancia il nostro motore matematico `run_ibi_loop.py` che:
1. Legge le coordinate bersaglio dal dataset.
2. Ricava le energie tramite Direct Boltzmann Inversion ($V_0$).
3. Effettua iterazioni di Dinamica Molecolare simulata per aggiornare i potenziali splinati fino alla convergenza (esportandoli in `ibi_priors/`).

Successivamente, uno script al volo modifica il tuo `cg_priors.json` impostando `"type": "tabulated"` sui legami, in modo che ESPResSo legga le curve splinate.
Infine, calcola esplicitamente la forza esercitata da quelle curve splinate su ogni frame, sottraendole dal dataset originale e generando il **`tel22_residual_dataset.bin`**.

### 03_train_model.sh
Passa il binario al programma C++. Addestrerà la rete Graph Neural Network in C++ (tramite LibTorch). A differenza dell'approccio DBI classico, qui la rete dovrà fare molta meno fatica, dovendo imparare solo il rumore (le forze residue non lineari), mentre i muri sterici sono gestiti matematicamente dall'IBI.

### 04_run_espresso.sh
Carica il modello C++ appena addestrato all'interno del motore di ESPResSo. Quando ESPResSo andrà ad applicare i legami, non userà semplici molle di Hooke, ma interpolerà in tempo reale i valori dalle tabelle numeriche `.dat` precedentemente calcolate!

---

## Analisi e Visualizzazione
Una volta conclusa la simulazione, verrà generato un file di traiettoria Coarse-Grained (`cg_trajectory.vtf`).
Per visualizzarlo o analizzarne lo stacking e la stabilità, puoi utilizzare esattamente gli stessi identici script di analisi termodinamica descritti nel tutorial base `tel22`:
- **`vtf_to_pymol.py`** per l'animazione PDB
- **`analyze_unfolding_exact.py`** per valutare le distanze di raggio di girazione.
