# Tutorial: TEL22 G-Quadruplex (End-to-End Pipeline)

Benvenuto in questo tutorial avanzato! Qui imparerai come generare un modello Coarse-Grained per una molecola complessa come il DNA telomerico umano (TEL22, che forma una struttura a G-quadruplex).

Rispetto a Etanolo o Acqua, qui partiremo da zero: genereremo la simulazione All-Atom in GROMACS, estrarremo le forze e addestreremo il modello!

In questa cartella troverai i singoli script numerati da eseguire uno dopo l'altro (o puoi lanciare `run_full_pipeline.sh` per farli eseguire tutti in sequenza automatica).

---

## 01_run_gromacs.sh
Questo script si collega al database PDB (RCSB), scarica in automatico tramite `curl` la struttura **143D** (il G-quadruplex determinato via NMR), la quale contiene nativamente **6 copie (catene)** della molecola TEL22 già ripiegate, perfette per una simulazione multi-molecolare.

> [!IMPORTANT]
> **Importanza degli Ioni per il DNA G-Quadruplex**
> I G-quadruplex necessitano obbligatoriamente di ioni **Potassio (K+)** incastonati all'interno del canale tetradico per non denaturarsi durante la dinamica. Lo script usa il comando GROMACS `genion` per inserire in automatico una quantità neutralizzante di K+ e Cl- (circa 0.15M) prima di avviare la simulazione.

Se apri lo script `01_run_gromacs.sh` vedrai esattamente tutti i comandi necessari per generare una simulazione All-Atom da zero. Lo script esegue:
1. `curl`: Scarica il file PDB originale.
2. `pdb2gmx`: Genera la topologia del DNA usando il Forcefield **AMBER99SB-ILDN** ignorando gli idrogeni NMR.
3. `editconf`: Centra le 6 molecole all'interno di un box cubico lasciando 1.5 nm di margine.
4. `solvate`: Riempie il box di molecole d'acqua (modello TIP3P).
5. `genion`: Sostituisce parte dell'acqua con 154 ioni K+ e 28 ioni Cl-.
6. **Minimizzazione dell'energia** (`mdrun` su `minim.mdp`).
7. **Equilibrazione NVT e NPT** (`mdrun` su `nvt.mdp` e `npt.mdp`) per stabilizzare temperatura (300K) e densità.
8. **MD Production (1 ns)**: Lancia la vera e propria simulazione molecolare salvando posizioni, velocità e, cosa fondamentale per il Machine Learning, le **forze** (`md.trr`) ogni singolo picosecondo.

---

## 02_build_dataset.sh
Una volta ottenuti i file `md.trr` e `md.gro`, questo script lancia il nostro pre-processore.
Utilizza il file `tel22_topology.json` in cui abbiamo specificato di mappare ogni singolo nucleotide (DA, DT, DG, DC) al suo Centro di Massa. Genererà il dataset `tel22_dataset.bin` pronto per la rete neurale.

---

## 03_train_model.sh
Passa il binario al programma C++. Addestrerà una rete Graph Neural Network in C++ (tramite LibTorch) per prevedere le interazioni residenziali. Per il tutorial è impostato su sole 5 epoche, ma sentiti libero di aumentarle in `tel22_training_config.json`.

---

## 04_run_espresso.sh
Carica il modello C++ appena addestrato all'interno del motore di ESPResSo per validare la stabilità strutturale e la conservazione dell'energia!
