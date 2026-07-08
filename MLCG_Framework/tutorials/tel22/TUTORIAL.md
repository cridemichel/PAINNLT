# Tutorial: TEL22 G-Quadruplex (End-to-End Pipeline)

Benvenuto in questo tutorial avanzato! Qui imparerai come generare un modello Coarse-Grained per una molecola complessa come il DNA telomerico umano (TEL22, che forma una struttura a G-quadruplex).

Rispetto a Etanolo o Acqua, qui partiremo da zero: genereremo la simulazione All-Atom in GROMACS, estrarremo le forze e addestreremo il modello!

In questa cartella troverai i singoli script numerati da eseguire uno dopo l'altro (o puoi lanciare `run_full_pipeline.sh` per farli eseguire tutti in sequenza automatica).

---

## 01_run_gromacs.sh
Questo script si collega al PDB (RCSB), scarica la struttura **143D** e costruisce un ambiente acquoso completo con 10 molecole di TEL22.
> [!IMPORTANT]
> **Importanza degli Ioni**
> I G-quadruplex necessitano di ioni **Potassio (K+)** all'interno del canale tetradico per non denaturarsi. Lo script usa `gmx genion` per inserire una quantità neutralizzante di K+ e Cl- (circa 0.15M) prima di avviare la dinamica.

Lo script esegue:
1. `pdb2gmx` (AMBER99SB-ILDN)
2. Inserimento di 10 molecole in un box da 8 nm
3. Solvatazione e Ionizzazione
4. Minimizzazione
5. Equilibrazione NVT e NPT
6. **MD Production (1 ns)** salvando le **forze** ogni 1 ps.

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
