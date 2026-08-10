# Tutorial: TEL22 G-Quadruplex (End-to-End Pipeline)

Benvenuto in questo tutorial avanzato! Qui imparerai come generare un modello Coarse-Grained per una molecola complessa come il DNA telomerico umano (TEL22, che forma una struttura a G-quadruplex).

Rispetto a Etanolo o Acqua, qui partiremo da zero: genereremo la simulazione All-Atom in GROMACS, estrarremo le forze e addestreremo il modello!

In questa cartella troverai i singoli script numerati da eseguire uno dopo l'altro (o puoi lanciare `run_full_pipeline.sh` per farli eseguire tutti in sequenza automatica).

---

## 01_run_gromacs.sh
Questo script si collega al database PDB (RCSB) e scarica tramite `curl` la struttura **143D** (il G-quadruplex NMR). Poiché è un ensemble NMR contenente modelli sovrapposti, lo script estrae accuratamente solo il **primo modello** per evitare infiniti picchi di energia.

> [!IMPORTANT]
> **Importanza degli Ioni per il DNA G-Quadruplex**
> I G-quadruplex necessitano obbligatoriamente di ioni **Potassio (K+)** incastonati all'interno del canale tetradico per non denaturarsi durante la dinamica. Lo script usa il comando GROMACS `genion` per inserire in automatico una quantità neutralizzante di K+ e Cl- (circa 0.15M) prima di avviare la simulazione.

Se apri lo script `01_run_gromacs.sh` vedrai esattamente tutti i comandi necessari per generare una simulazione All-Atom da zero. Lo script esegue:
1. `curl` ed `awk`: Scarica e pulisce il PDB.
2. `pdb2gmx`: Genera la topologia del singolo filamento di DNA usando il Forcefield **AMBER99SB-ILDN** ignorando gli idrogeni NMR.
3. `insert-molecules`: Replica la singola molecola di DNA **10 volte** e le inserisce randomicamente in un box cubico da 8 nm.
4. `solvate`: Riempie il box vuoto di molecole d'acqua (modello TIP3P).
5. `genion`: Sostituisce l'acqua in eccesso inserendo ioni K+ e Cl- per neutralizzare le 10 catene e creare la soluzione salina.
6. **Minimizzazione dell'energia** (`mdrun` su `minim.mdp`) per risolvere eventuali sovrapposizioni steriche.
7. **Equilibrazione NVT e NPT** (`mdrun` su `nvt.mdp` e `npt.mdp`) per stabilizzare temperatura (300K) e densità.
8. **MD Production (1 ns)**: Lancia la vera simulazione salvando le **forze** (`md.trr`) necessarie per il ML.

---

## 02_build_dataset.sh
Una volta ottenuti i file `md.trr` e `md.gro`, questo script lancia il nostro pre-processore.
Utilizza il file `tel22_topology.json` in cui abbiamo specificato di mappare ogni singolo nucleotide (DA, DT, DG, DC) al suo Centro di Massa, ma con regole fisiche **avanzate**:
1. **Virtual Sites & WCA Overrides**: Abbiamo disaccoppiato il nucleo fosfato-zucchero dalla base azotata (es. Guanina) assegnando a quest'ultima un *Virtual Site* (site_type 2). Usando `wca_overrides` nel file JSON, la Guanina possiede un repulsore sterico molto più "morbido" ed elastico ($\epsilon = 2.5$) rispetto agli altri siti duri ($\epsilon = 50.0$).
2. **Priors Site-Dependent**: Gli angoli del DNA e i legami vengono geometricamente misurati e forzati *esattamente* sui Virtual Sites specificati (`site_i: 0`), non sui Centri di Massa generici. Il framework calcolerà automaticamente le forze di torsione per conservare il momento angolare!

Genererà il dataset `tel22_dataset.bin` pronto per la rete neurale.

---

## 03_train_model.sh
Passa il dataset binario al trainer PaiNN C++/LibTorch e ottimizza simultaneamente forze traslazionali e torque dei rigid body con loss normalizzata sugli RMS del train set.

Per TEL22 il profilo di produzione selezionato dalle ablation tiny-set è:
- `hidden_channels = 128`
- `n_layers = 3`
- `num_rbf = 64`
- `cutoff = 1.6 nm`
- `torque_weight = 0.5`
- `batch_size = 4`
- massimo `200` epoche, con riduzione del learning rate su plateau ed early stopping.

Il limite di 200 epoche è un massimo: il trainer conserva il miglior checkpoint di validation e può arrestarsi prima. Il file `tel22_training_config.json` non viene riscritto se esiste già, così eventuali configurazioni deliberate non vengono perse.

---

## 04_run_espresso.sh
Carica il modello C++ appena addestrato all'interno del motore di ESPResSo per validare la stabilità strutturale e la conservazione dell'energia!

> [!TIP]
> [!TIP]
> **Stabilità NVE: Il Trade-off tra Bias ed Envelope**
> I modelli Graph Neural Networks (come PaiNN) generano tipicamente discontinuità al raggio di cutoff a causa dei *bias* nei layer lineari, distruggendo lo scaling nativo $\mathcal{O}(dt^2)$ dell'integratore Verlet nelle simulazioni NVE. Per risolvere questo problema, il framework usa nativamente lo Smoothing di Toxvaerd ($\mathcal{C}^3$). Nel tutorial puoi scegliere due approcci:
> 
> 1. **Approccio Rigoroso (Default nel Tutorial):** `use_bias=false` nel training (nessun envelope in MD). La rete non ha gradienti discontinui. Lo scaling è numericamente perfetto ($\approx 1.99$), ideale per NVE di precisione, anche se l'errore assoluto a corto raggio cresce leggermente per la perdita dei parametri di bias.
> 2. **Approccio Termodinamico:** Se decidi di impostare `"use_bias": true` nel file di configurazione del training per abbassare l'errore assoluto, è fortemente raccomandato abilitare contemporaneamente `"apply_envelope": true`. I tuoi script di simulazione (`04_equilibrate.sh` e `05_run_espresso.sh`) non necessitano più di flag manuali: essi andranno a leggere automaticamente il file di configurazione (`tel22_training_config.json`) e applicheranno l'envelope di Toxvaerd se la rete è stata addestrata per richiederlo. Lo scaling sarà leggermente sub-ottimale ($\approx 1.89$), ma le fluttuazioni dell'energia resteranno molto più piccole in valore assoluto!
> *Nota*: I parametri numerici avanzati (es. `toxvaerd_alpha`) possono ancora essere passati manualmente, ma il flag strutturale sull'envelope non è più necessario a CLI.

---

## 05. Visualizzazione (PyMOL)
Una volta conclusa la simulazione ESPResSo, verrà generato un file di traiettoria Coarse-Grained (`cg_trajectory.vtf`).
Per visualizzare questa traiettoria in modo elegante e animato, abbiamo incluso uno script utility:
```bash
python vtf_to_pymol.py
```
Puoi anche decidere come renderizzare le molecole passando il flag opzionale `--style`:
- `python vtf_to_pymol.py --style tube` (Default: disegna filamenti continui)
- `python vtf_to_pymol.py --style spheres` (Disegna le singole sfere coarse-grained disconnesse)

Questo script eseguirà due operazioni:
1. Analizzerà la traiettoria `vtf` scartando i Virtual Sites e manterrà solo i veri Centri di Massa, generando una traiettoria PDB pulita (`cg_trajectory_clean.pdb`).
2. Genererà uno script di setup per PyMOL (`load_tel22_pymol.pml`).

Per visualizzare l'animazione, apri il terminale e digita:
```bash
pymol load_tel22_pymol.pml
```
PyMOL si aprirà automaticamente applicando lo stile da te scelto (tubi o sfere) e colorando ogni filamento in modo distinto. Premi "Play" in basso a destra nell'interfaccia di PyMOL per osservare la dinamica termica!

---

## 06. Analisi e Validazione del Modello
Per valutare quantitativamente se il modello ML è stato in grado di mantenere il TEL22 ripiegato nel tempo, abbiamo incluso due script di analisi geometrica nella cartella:

1. **`analyze_unfolding_exact.py`**:
   Questo script calcola il **Raggio di Girazione ($R_g$)** e la **Distanza End-to-End (E2E)** misurando la differenza esatta tra il primo frame (di partenza) e l'ultimo frame della simulazione per ogni filamento. Si usa per certificare se qualche molecola si è "srotolata" (unfolded).
   ```bash
   uv run analyze_unfolding_exact.py
   ```

2. **`plot_rg_timeseries.py`**:
   Questo script estrae i dati per **tutti i frame** della traiettoria e genera un elegante grafico a linee (`output/tel22_rg_timeseries.png`). Mostra l'andamento del Raggio di Girazione ($R_g$) di ciascuno dei 10 strand nel tempo e calcola una curva di stabilità media, permettendo di accertarsi visivamente che il G-Quadruplex non si stia gonfiando a causa del solvente.
   ```bash
   uv run plot_rg_timeseries.py
   ```

---

## 07. Esperimento Avanzato: Modello Euristico per l'Unfolding (Pure ML)

Se il tuo obiettivo non è solo simulare il G-Quadruplex a 300 K, ma studiarne lo **srotolamento termico (unfolding)** a temperature molto elevate (es. 1000 K), incontrerai un problema: i potenziali classici tabulati per gli Angoli e i Diedri andranno in crash. Quando la spina dorsale si srotola, gli atomi possono allinearsi a 180°, generando una singolarità matematica nei diedri IBI (`bond broken between particles`).

Per risolvere questo problema e simulare l'unfolding, possiamo creare un **Modello Euristico Pure ML**, delegando il 100% della geometria alla Rete Neurale e mantenendo solo le molle `FENE` per l'integrità della spina dorsale e i legami `Morse` per i legami a idrogeno.

Ecco la procedura passo-passo per attivare l'unfolding sicuro:

1. **Eliminare Angoli e Diedri alla fonte:**
   Lo script `02_build_dataset.sh` rigenera ogni volta il file `cg_priors.json` leggendo dal file principale `tel22_topology.json`. 
   > [!WARNING]
   > Non modificare a mano `cg_priors.json`, perché verrà sovrascritto!
   Apri `tel22_topology.json`, cerca in fondo al file e **svuota** le liste `"angles"` e `"dihedrals"` facendole diventare matrici vuote (`[]`). Assicurati inoltre che il `"wca_sigma"` sia impostato a `0.6` (o maggiore) per evitare compenetrazioni durante le collisioni ad alta temperatura.

2. **Ricalcolare il Dataset e Addestrare:**
   Rilancia `./02_build_dataset.sh`. Ora lo script non sottrarrà più le forze dei diedri. 
   Lancia `./03_train_model.sh`. La Rete Neurale dovrà imparare da sola l'intera rigidità torsionale della molecola! (Consiglio: alza le `epochs` a 100 o 200 in `tel22_training_config.json` per aiutarla).

3. **Alzare la Temperatura in ESPResSo:**
   Il nostro script di esecuzione supporta la variazione dinamica della temperatura tramite l'argomento `--kT`.
   Apri `04_run_espresso.sh` e aggiungi il parametro `--kT 8.31` (che corrisponde a circa 1000 K, contro i classici 2.49 di 300 K).
   
Quando avvierai la simulazione, il modello estrapolerà dolcemente a grandi distanze e il tuo DNA si denaturerà in un perfetto polimero *random-coil* senza mai andare in crash!

### Diagnostic: global symmetry projection (03g)

If full physical validation remains at the zero-predictor baseline, use
`03g_symmetry_projection_test.sh` before increasing PaiNN capacity again.  It
compares the same fast 64x2 PaiNN on the raw force-matching targets and on a
diagnostic copy where the frame-wise net force and residual global generalized
torque have been removed.  The original `tel22_dataset.bin` is not modified;
legacy zero-target OOD decoys, if present in an old dataset, remain unchanged but
are excluded from optimization by default because the binary schema has no
per-molecule loss mask.

### Training-safety corrections (canonical PaiNN / unmasked decoys)

The current TEL22 profile uses the canonical PaiNN interatomic context MLP
(`D -> D -> 3D` with SiLU), a stabilized vector norm, deterministic per-epoch
shuffle, physical-only validation, and `include_decoys_in_train=false`.  The
legacy whole-frame zero-target OOD decoys are disabled at preprocessing time
(`decoy_target_fraction=0`) because only a local contact is perturbed while the
old format labels every molecule in that synthetic frame as zero residual.

DA and DT are one-site CG molecules.  Their sole site is mapped with `['*']`,
so it coincides exactly with the residue COM; this is required because one-site
bodies have no rotational degree of freedom in the ESPResSo runtime.

These changes alter both the mapped dataset and the PaiNN parameterization.
After applying the corresponding patch, rebuild `tel22_dataset.bin` with
`./02_build_dataset.sh`, recompile `training/train_painn`, delete/rename old
`.pt` files and manifests, then retrain from scratch.
