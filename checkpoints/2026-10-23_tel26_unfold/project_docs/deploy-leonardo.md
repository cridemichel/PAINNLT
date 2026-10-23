# Deploy di MLCG_Framework_v2 su Leonardo (CINECA)

Stato al 2026-09-22. **Installazione completata, verificata e allineata.
Tutorial TEL26 in repo (commit `94f11df`).** Resta la generazione della
topologia sui dati veri e il dataset dalle traiettorie di Giulia.

## Cosa funziona

`pypresso` con il plugin PaiNN, `train_painn` e un venv con MDAnalysis, in
ambiente nativo (niente container), installato in
`/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT`. Verifiche passate:

- `import espressomd.painn` riesce e `activate_painn_potential` espone tutti i
  parametri, ordered-geometry compresi;
- `train_painn` parte, rileva il device e si ferma sulla config mancante;
- su un nodo Booster: `torch.cuda.is_available()` → `True`,
  `NVIDIA A100-SXM-64GB`, driver 535.274.02 (la *minor version compatibility*
  di CUDA fa girare un binario 12.6 su un driver che espone la 12.2);
- ESPResSo e trainer ricompilati con CMake che vede il toolkit 12.6 in tutta
  la catena.

## La combinazione di versioni

Non è libera: ogni voce è imposta da un vincolo diverso. Sono i default degli
script; la guida completa è in `MLCG_Framework_v2/hpc/HOWTO_LEONARDO.md`.

| componente | versione | vincolo |
|---|---|---|
| gcc | 12.2.0 | ESPResSo rifiuta < 12.2; il sistema ha 8.5.0 |
| cuda | 12.6, con `module swap` e caricata **per ultima** | torch 2.10 usa API da CUDA 12.5; `openmpi` carica 12.2 come dipendenza e la riporterebbe indietro |
| torch | 2.10.0+cu126, dai wheel | glibc 2.28 esclude lo zip LibTorch; la ABI `cxx11` esclude i wheel ≤ 2.5; 2.10 è anche la versione del Mac |
| boost/fftw/openmpi | default dei moduli | compilati con gcc 12.2.0 |
| ESPResSo | commit `84cc1d924` | il plugin innesta file in percorsi che altre versioni spostano |

## Quello che abbiamo scoperto sul repo

L'innesto del plugin **non era completo nel repository**: tre modifiche
vivevano solo nell'albero locale del Mac.

1. `PaiNN_ML_Potential.cpp` non era in `target_sources()` → non compilato.
2. Nel `CMakeLists.txt` del core mancavano `find_package(Torch)`, il link a
   `${TORCH_LIBRARIES}` e la define `ESPRESSO_PAINN`.
3. In `forces.cpp` mancava la chiamata a
   `global_painn_potential->calculate_forces()`.

I primi due danno errori espliciti; **il terzo no**: senza, ESPResSo compila,
importa e simula in silenzio con i soli prior. Ora tutto è automatizzato in
`simulation/espresso_plugin/install_painn_core_sources.py` (idempotente), che
forza anche la visibilità dei simboli sui target di Kokkos.

**Alla prima simulazione verificare che `E_ML` non sia identicamente zero**: è
il controllo che distingue un modello attivo da uno che non viene mai chiamato.

## Struttura degli stadi

`hpc/submit_leonardo.sh <stadio> [VAR=valore ...]` — unico posto con account,
partizioni, risorse. `AFTER=<jobid>` incatena i job con `--dependency=afterok`,
ma la dipendenza va dichiarata mentre il job precedente è ancora in coda.

Il **sistema** è una variabile: `SYSTEM=tel26` sposta tutta la pipeline sulla
tutorial directory `tutorials/tel26` e sui nomi `tel26_*`. Default `tel22`.

| stadio | dove | perché |
|---|---|---|
| `setup` | lrd_all_serial | scarica: torch, pacchetti, ESPResSo |
| `configure` | lrd_all_serial | ESPResSo scarica heFFTe, Kokkos, Cabana |
| `build` | DCGP 32 core | solo compilazione |
| `dataset`, `noisefloor`, `analysis` | DCGP 16 core | MDAnalysis, numpy |
| `train`, `select`, `production` | Booster, 1 GPU | l'unico lavoro che usa la GPU |

`select` e `analysis` si fermano con un errore esplicito fuori dal TEL22:
importano `_tel22_cv`, che codifica 22 nucleotidi e le loro coordinate
collettive. Vanno riscritti per TEL26.

Vincoli che determinano questa forma: niente `--fakeroot` (quindi niente
container), 600 s di CPU per processo sul login, rete solo sui nodi di login
(e `lrd_all_serial` gira lì), due account ISCRA (Booster e DCGP), quota di
`$HOME`.

## Strumenti nuovi

- `preprocessing/build_g4_topology.py` — genera la topologia CG di un
  G-quadruplex a più copie: eredita dal template il mapping per residuo e i
  parametri (WCA, prior, forma dei Morse), rigenera bond di backbone, angoli e
  Morse K4, e **ricava il registro delle tetradi dalla geometria**.
  Le tetradi si cercano come gruppi *complanari* lungo l'asse di impilamento,
  non come quartine più vicine: in un G4 le guanine impilate distano ~0,34 nm
  e quelle complanari della stessa tetrade ~0,6 nm, quindi un criterio di
  prossimità trova le pile invece dei piani (verificato su struttura
  sintetica, dove sbagliava sistematicamente). Lo script riporta asse,
  spessore dei piani, separazione, planarità e lati di ogni tetrade, e
  controlla che ogni tetrade prenda una guanina da ciascuno dei quattro
  tratti — criterio strutturale, indipendente dalla geometria.
- `preprocessing/extract_solute_topology.py` — topologia ridotta ai
  `non-Water` per leggere un `.xtc` scritto con `compressed-x-grps`.
  Conteggio e coordinate del primo frame si leggono **dal reader**, non da un
  `Universe(tpr, xtc)`: MDAnalysis rifiuta quella coppia proprio quando i
  conteggi differiscono, cioè nel caso d'uso per cui lo script esiste.
- `build_cg_dataset.py --forces-trajectory` — posizioni e forze da due
  traiettorie diverse, appaiate per tempo, con verifica che gli atomi
  corrispondano per numero e nome.

## Tutorial TEL26

`MLCG_Framework_v2/tutorials/tel26/`, con `TUTORIAL.md` che documenta i
controlli da fare. Stadi `01`–`05`; `01_prepare_topology.sh` è quello che il
TEL22 non aveva e fa i due passi preliminari (topologia ridotta + topologia
CG). Niente `validate_antiparallel_topology.py`: assume la piega del 143D.

Tre controlli sull'output dello stadio `01`, in ordine di gravità:

1. **260 residui mappabili** (10 × 26). 240 = terminali AMBER non riconosciuti.
2. **Lati di tetrade 0,6–1,1 nm**, scarto dal piano < ~0,15 nm. Lati di
   0,03–0,20 nm = struttura senza coordinate.
3. **Un tratto controcorrente**: `min(su, giù)` vale 0 per la piega parallela,
   **1 per l'ibrida 3+1**, 2 per l'antiparallela. Per il 2JPZ deve dare 1.

Il terzo è il controllo che conta: un registro sbagliato produce un campo di
forza che gira e vincola le basi sbagliate — non fallisce, sbaglia in silenzio.

## Prossimi passi

1. **Topologia TEL26** sui dati veri, sul nodo di login:
   ```
   source hpc/env_leonardo.sh
   cd MLCG_Framework_v2/tutorials/tel26
   AA_TPR=<.../run/md_tel26_hybrid_2jpz_box12nm.tpr> \
   AA_XTC=<.../run/prod-1.part0001.xtc> \
       bash 01_prepare_topology.sh
   ```
2. **Dataset** dalle traiettorie di Giulia
   (`/leonardo_work/IscrB_G4MES/GiuliaC/machine_learning/Tel26_Hybrid_2jpz_10_monomeri/box_12nm_spce/run/`):
   il `.trr` contiene le sole forze (`nstxout = 0`), le posizioni stanno
   nell'`.xtc` ristretto ai `non-Water`, entrambi a 20 ps. ~10 000 frame in
   prod-1 più ~4 700 in prod-2.
   ```
   bash hpc/submit_leonardo.sh dataset SYSTEM=tel26 \
        AA_TOPOLOGY=.../tel26_solute.gro AA_TRAJECTORY=.../prod-1.part0001.xtc \
        AA_FORCES_TRAJECTORY=.../prod-1.part0001.trr AA_FORCES_TOPOLOGY=.../prod-1.tpr
   ```
   **Misurare il costo su un sottoinsieme** (500 frame) prima di lanciare tutto.
3. `noisefloor` prima del training: decide se la validation loss ordina i
   modelli o se la selezione va fatta sulla struttura.
4. Prima produzione CG: controllare `E_ML` diversa da zero.
5. Riscrivere `_tel22_cv` per 26 residui, per riabilitare `select` e `analysis`.

## Promemoria: cosa chiedere a Giulia

Le produzioni attuali salvano le forze **ogni 20 ps** (`nstfout = 10000`,
`dt = 0.002`). Se girerà altri 10-20 ns, chiederle `nstxout = nstfout = 500`
(coordinate *e* forze ogni 1 ps): da lì si possono mediare le forze su finestre
e la varianza del target scende come il numero di campioni per finestra. Sono
poche ore di GPU e cambiano la qualità del training più di qualunque intervento
a valle. Da chiedere **prima** che la produzione riparta.

## Costi stimati su A100

Regola di scala: **~0,04 s per frame per epoca** (dai tempi su M3 Max/MPS con
un fattore 3-5×, da verificare sul primo job). Training C3 40 epoche su 801
frame ~21 min (~0,09 nodo-ore). Oltre ~50 000 frame si sfora il limite di 24 h
del Booster: il trainer ha `--resume`. Il rischio vero non è la GPU ma lo
stadio `dataset`, CPU e lineare nei frame: misurarlo su 500 frame.

## Nota storica

Il 21/09 l'utenza è stata inutilizzabile per un giorno — aree di lavoro negate
e ogni `sbatch` rifiutato. Segnalato a `superc@cineca.it`, risolto il giorno
dopo senza spiegazioni. Le prove fallite erano tutte su `login01`, quelle
riuscite su `login05`: **se il sintomo torna, provare prima un altro nodo di
login**.
