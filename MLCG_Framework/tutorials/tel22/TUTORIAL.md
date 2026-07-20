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
Passa il binario al programma C++. Addestrerà una rete Graph Neural Network in C++ (tramite LibTorch) per prevedere le interazioni residenziali. Per il tutorial è impostato su sole 5 epoche, ma sentiti libero di aumentarle in `tel22_training_config.json`.

---

## 04_run_espresso.sh
Carica il modello C++ appena addestrato all'interno del motore di ESPResSo per validare la stabilità strutturale e la conservazione dell'energia!

> [!TIP]
> **Stabilità dell'Energia NVE e Bias**
> Se il tuo modello utilizza reti neurali con layer lineari provvisti di Bias (es. `PaiNN`), potresti riscontrare fluttuazioni ed esplosioni dell'energia termodinamica dovute a discontinuità di forza al raggio di cutoff.
>
> Per risolvere matematicamente questo problema, abbiamo introdotto il flag `--apply_envelope` nello script `run_cg_md.py`. Passando questo flag, il motore C++ moltiplicherà a posteriori l'output del layer per una funzione *cosine envelope*, forzando l'interazione dolcemente a zero ai bordi del raggio di cutoff. Questo ti garantirà simulazioni NVE stabili in eterno!
> Esempio di esecuzione: `python ../../simulation/run_cg_md.py --apply_envelope ...`

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
