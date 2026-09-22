# Installare MLCG_Framework_v2 su Leonardo (CINECA)

Procedura completa e riproducibile: da un'utenza nuova al framework compilato.
Ricavata da un'installazione reale, quindi ogni vincolo qui elencato è stato
incontrato davvero.

**Il risultato**: `pypresso` con il plugin PaiNN, il trainer `train_painn` e un
ambiente Python con MDAnalysis. Tutto nativo, senza container.

Procedura verificata il 22/09/2026 su un'installazione completa: l'import di
`espressomd.painn` riesce e la firma di `activate_painn_potential` elenca tutti
i parametri, ordered-geometry compresi.

---

## 0. La combinazione di versioni (leggere prima di tutto)

Non è libera: ciascuna voce è imposta da un vincolo diverso, e insieme
identificano una sola configurazione funzionante.

| componente | versione | perché proprio questa |
|---|---|---|
| `gcc` | **12.2.0** | ESPResSo rifiuta compilatori < 12.2; il gcc di sistema di RHEL 8 è 8.5.0 |
| `cuda` | **12.6** | torch 2.10 usa simboli CUDA introdotti in 12.5; il default di Leonardo è 12.2 |
| `torch` | **2.10.0+cu126**, dai wheel | glibc 2.28 (esclude lo zip LibTorch), ABI `cxx11` di libstdc++ (esclude i wheel ≤ 2.5), e stessa versione del Mac (checkpoint TorchScript compatibili) |
| `boost` / `fftw` / `openmpi` | i default dei moduli | sono compilati con gcc 12.2.0, quindi coerenti |
| ESPResSo | commit **`84cc1d924`** (`5.0.0-58`) | il plugin innesta file in `src/core/nonbonded_interactions` e `src/python/espressomd`: un'altra versione può averli spostati |
| Kokkos / Cabana / heFFTe | quelle che ESPResSo scarica | fissate nel suo `CMakeLists.txt` |

Sono già i default degli script: `LIBTORCH_VERSION`, `LIBTORCH_CUDA`,
`MODULE_CUDA` servono solo per deviare da qui.

---

## 1. I cinque vincoli di Leonardo

Spiegano perché la procedura ha la forma che ha.

**1.1 Niente container.** SingularityPRO 4.3 c'è (`/usr/bin/singularity`, non
esiste un modulo `apptainer`), ma mancano le mappe subuid/subgid:

```
$ singularity build --fakeroot prova.sif docker://alpine:latest
FATAL: could not use fakeroot: no valid mapping entry found for <user> (<uid>)
```

Senza `--fakeroot` un `.def` con `%post` non è costruibile qui. Si potrebbe
costruire l'immagine altrove e trasferirla, ma l'ambiente nativo è più veloce
da iterare e usa MPI e Boost ottimizzati del cluster.

**1.2 Il login node uccide i processi lunghi.** `ulimit -t` = 600 secondi di
CPU. Ogni cosa lunga va in un job.

**1.3 La rete c'è solo sui nodi di login.** I nodi di calcolo non raggiungono
internet; `lrd_all_serial` gira *sui* nodi di login (`login08`, `login13`). Da
qui la separazione fra stadi che scaricano e stadi che compilano. Nota che
anche la **configurazione** di ESPResSo scarica (heFFTe, Kokkos, Cabana, con
`FetchContent`), non solo il setup.

**1.4 glibc 2.28, e l'ABI di libstdc++.** RHEL 8. Lo zip ufficiale di LibTorch
è costruito contro una glibc più recente e non linka (`undefined reference to
log2@GLIBC_2.29`). I wheel PyTorch sono `manylinux_2_28` e vanno bene — ma solo
quelli con `_GLIBCXX_USE_CXX11_ABI=True`: `TorchConfig.cmake` propaga quel flag
via `TORCH_CXX_FLAGS` a tutto ciò che linka Torch, e con la ABI vecchia il core
di ESPResSo finisce con una `std::string` diversa da quella di Kokkos e Boost.

**1.5 Due account.** Un progetto ISCRA B ha `<PROGETTO>` sul Booster (ore GPU)
e `<PROGETTO>_0` su DCGP (ore CPU). Compilare e leggere traiettorie non usano
la GPU. `lrd_all_serial` non accetta l'associazione DCGP, quindi gli stadi che
ci girano non passano `--account`.

| stadio | dove | perché |
|---|---|---|
| `setup` | `lrd_all_serial` | scarica torch, i pacchetti, ESPResSo |
| `configure` | `lrd_all_serial` | ESPResSo scarica heFFTe, Kokkos, Cabana |
| `build` | `dcgp_usr_prod`, 32 core | solo compilazione |
| `dataset`, `noisefloor`, `analysis` | `dcgp_usr_prod`, 16 core | MDAnalysis e numpy |
| `train`, `select`, `production` | `boost_usr_prod`, 1 GPU | l'unico lavoro che usa la GPU |

---

## 2. Prerequisiti

### L'utenza deve essere abilitata

```bash
id
ls -ld /leonardo_work/<PROGETTO>
sbatch --test-only -p lrd_all_serial -t 00:10:00 --wrap=hostname
```

Se l'area di progetto dà *Permission denied* mentre quelle di altri progetti
si vedono, o se ogni `sbatch` risponde *Invalid account or account/partition
combination* benché `sacctmgr show assoc user=$USER` elenchi associazioni
valide, l'abilitazione non è completa: scrivere a `superc@cineca.it`. **Prima
però provare da un altro nodo di login**: in un caso reale il blocco c'era solo
su `login01` e non su `login05`.

### Dove installare

`$HOME` ha quota piccola, lo scratch viene ripulito periodicamente:

```bash
BASE=/leonardo_work/<PROGETTO>/$USER
mkdir -p "$BASE"
```

---

## 3. Installazione

```bash
cd "$BASE"
git clone https://github.com/cridemichel/PAINNLT.git
cd PAINNLT/MLCG_Framework_v2

bash hpc/submit_leonardo.sh setup        # ~5 min
bash hpc/submit_leonardo.sh configure    # ~2 min, dopo che setup è finito
bash hpc/submit_leonardo.sh build        # ~15 min, dopo che configure è finito
```

**Gli stadi vanno in sequenza.** Per incatenarli senza sorvegliare la coda:

```bash
S=$(bash hpc/submit_leonardo.sh setup | awk '/Submitted/{print $4}')
C=$(AFTER=$S bash hpc/submit_leonardo.sh configure | awk '/Submitted/{print $4}')
AFTER=$C bash hpc/submit_leonardo.sh build
```

`AFTER` usa `--dependency=afterok`: se uno stadio fallisce i successivi non
partono e restano in coda come `DependencyNeverSatisfied`, da rimuovere con
`scancel`. Attenzione: la dipendenza va dichiarata **mentre** il job precedente
è ancora in coda, altrimenti SLURM risponde `Job dependency problem`.

Cosa fa ciascuno stadio:

- **`setup`** — virtualenv con numpy, scipy, matplotlib, MDAnalysis, h5py,
  Cython; `torch` dal wheel; clone di ESPResSo al commit fissato. Stampa
  `cxx11 ABI`, che **deve** dire `True`.
- **`configure`** — configura CMake, **poi** innesta il plugin, **poi**
  riconfigura. L'ordine conta: `install_switched_morse_nonbonded.py` scrive nel
  file di configurazione dentro `espresso/build`, che quindi deve già esistere,
  e l'innesto aggiunge sorgenti che CMake deve raccogliere.
- **`build`** — ESPResSo (senza CUDA e senza waLBerla) e i due eseguibili del
  trainer.

### Ricompilare dopo un cambio di versione

Se cambia torch, il modulo CUDA o il compilatore, **cancellare entrambe le
build** prima di riconfigurare: mescolare oggetti compilati contro ABI o
toolkit diversi produce errori oscuri a valle.

```bash
rm -rf espresso/build training/build
```

---

## 4. Verifica

Il `build` si chiude con:

```
[bootstrap] verifica
  ok        .../espresso/build/pypresso
  ok        .../espresso/build/src/python/espressomd/painn.so
  ok        .../training/build/train_painn
  ok        painn.so include il supporto ordered-geometry
```

Poi, l'unica prova che conta davvero — **l'ambiente va sempre caricato prima**,
altrimenti `pypresso` non trova `libboost_mpi`:

```bash
source hpc/env_leonardo.sh
espresso/build/pypresso -c "
import espressomd.painn as p, inspect
print(sorted(inspect.signature(p.activate_painn_potential).parameters))"
```

Se stampa l'elenco dei parametri, l'installazione è completa.

---

### Le feature di ESPResSo si decidono a compilazione

ESPResSo compila solo le interazioni dichiarate in `myconfig.hpp`; le altre non
esistono nel binario. Il suo default **non include `MORSE`**, su cui la
pipeline CG fonda i contatti fra guanine di una tetrade — e l'estensione
switched-Morse del framework vive dentro `#ifdef MORSE`, quindi sparisce
insieme a lei anche quando l'innesto e' andato a buon fine.

`MORSE` veniva gia' abilitata da `install_switched_morse_nonbonded.py`, che
pero' scriveva il suo `myconfig.hpp` in `espresso/build/`. La build directory
viene **cancellata e rigenerata** quando cambiano compilatore o
`MLCG_TORCH_ROOT`, e il file se ne va con lei; `copy_plugin_files.sh`, che lo
riscriverebbe, gira solo nello stadio `configure`. Una ricompilazione senza
riconfigurare lasciava quindi il binario senza `MORSE`, e il difetto emergeva
solo alla **prima produzione** — dopo dataset e training, ore di calcolo piu'
tardi, con l'errore

```text
RuntimeError: The ESPResSo build does not expose the non-bonded Morse interaction.
```

Il file sta ora in `simulation/espresso_plugin/myconfig.hpp`, cioe' sotto
controllo di versione, e `copy_plugin_files.sh` lo installa nella **radice dei
sorgenti**, che nessuno cancella; se una copia nella build directory esiste e
differisce, ha la precedenza e viene riallineata con un avviso.

La verifica del bootstrap controlla otto feature e prova un `set_params` con
`switch_start`, cosi' il difetto si vede in fondo alla build e non alla prima
produzione. La riga da cercare:

```text
  ok        feature ESPResSo complete (MORSE + switched-Morse)
```

## 5. Cosa significa "innestare il plugin"

Copiare i file non basta, e nemmeno compilarli.
`simulation/espresso_plugin/copy_plugin_files.sh` copia i sorgenti e chiama tre
installer idempotenti; quello del core (`install_painn_core_sources.py`) fa
quattro cose:

1. aggiunge `PaiNN_ML_Potential.cpp` a `target_sources()` — ESPResSo elenca i
   sorgenti esplicitamente, un file copiato nella directory non viene
   compilato;
2. aggiunge `find_package(Torch)`, il link a `${TORCH_LIBRARIES}` e la define
   `ESPRESSO_PAINN` nel `CMakeLists.txt` del core;
3. inserisce in `forces.cpp`, dentro `System::calculate_forces()`, la chiamata
   `global_painn_potential->calculate_forces(...)`;
4. forza la visibilità di default sui target di Kokkos.

I punti 1, 2 e 4, se mancano, danno errori espliciti. **Il punto 3 no**: senza
quella chiamata ESPResSo compila, linka, importa e simula in silenzio con i
soli prior, come se il modello non ci fosse. Per questo, alla prima
simulazione, conviene verificare che `E_ML` non sia identicamente zero.

Tutte queste modifiche erano state fatte a mano nell'albero locale e non erano
automatizzate: un albero ricostruito da zero non le aveva.

---

## 6. Diario degli errori

Ogni riga è un errore realmente incontrato.

| errore | causa | rimedio |
|---|---|---|
| `Unable to locate a modulefile for 'apptainer'` | su Leonardo il runtime è `singularity`, di sistema | irrilevante: si usa l'ambiente nativo |
| `could not use fakeroot: no valid mapping entry` | niente subuid/subgid | ambiente nativo |
| `Invalid account or account/partition combination` | associazione DCGP su `lrd_all_serial`, oppure utenza non abilitata | nessun `--account` su `lrd_all_serial`; se persiste ovunque, è l'utenza |
| `Unsupported compiler GNU 8.5.0` | modulo `gcc` non caricato | `env_leonardo.sh` lo carica e fissa `CC`/`CXX` |
| `CXX compiler changed` | cache di CMake di un tentativo precedente | il bootstrap rigenera la build directory |
| `ESPResSo build directory not found` | plugin innestato prima di configurare | `configure` fa CMake → plugin → CMake |
| `Failed to connect to github.com port 443` | `FetchContent` su un nodo di calcolo | `configure` su `lrd_all_serial` |
| `fatal error: torch/torch.h` | include non propagati al modulo Cython (il core linka `"${TORCH_LIBRARIES}"`, una lista di percorsi e non un target) | `CPATH` in `env_leonardo.sh` |
| `Caffe2: CUDA cannot be found` | `TorchConfig` pretende il toolkit, anche senza GPU | modulo `cuda` in `env_leonardo.sh` |
| `undefined reference to log2@GLIBC_2.29` | zip LibTorch contro glibc più recente della 2.28 | torch dai wheel |
| `No rule to make target .../libkineto.a` | cache del trainer con un'altra LibTorch | il bootstrap rigenera `training/build` quando `MLCG_TORCH_ROOT` cambia |
| `painn.so: undefined symbol: global_painn_potential` | sorgente non in `target_sources()` | `install_painn_core_sources.py` |
| `espresso_core.so: undefined symbol: _ZTIN3c105ErrorE` | il core non linkava LibTorch | idem |
| `undefined symbol: ...get_labelEv` mentre la libreria definisce `get_label[abi:cxx11]()` | mismatch di ABI libstdc++: wheel torch con `_GLIBCXX_USE_CXX11_ABI=0` | torch ≥ 2.7 (qui 2.10.0+cu126), `cxx11 ABI True` |
| `undefined reference to cudaGetDriverEntryPointByVersion@libcudart.so.12` | modulo CUDA più vecchio della CUDA del wheel (API da 12.5) | `MODULE_CUDA=cuda/12.6`, allineato a `cu126` |
| lo stesso errore **nonostante** `cuda/12.6` nei moduli | `module load cuda/12.6` non sostituisce la `cuda/12.2` che `openmpi` carica come dipendenza | serve `module swap`, e CUDA va caricata **per ultima** |
| `-- Found CUDA: ... version 12.2` con `CUDA attiva: 12.6` nello stesso log | `CUDA_HOME` / `CUDA_TOOLKIT_ROOT_DIR` rimaste quelle del modulo sostituito | il toolkit si ricava **sempre** da `nvcc`, mai dalle variabili d'ambiente |
| `libboost_mpi.so...: cannot open shared object file` | `pypresso` senza ambiente | `source hpc/env_leonardo.sh` |
| `painn.so senza ordered-geometry` (falso allarme) | `strings` non attraversa le tabelle di stringhe di Cython | il controllo usa `grep -a` |
| `Job dependency problem` | `AFTER` su un job già uscito dalla coda | sottomettere senza dipendenza |

Un tratto comune vale la pena notarlo: **creando una libreria condivisa il
linker non pretende di risolvere tutti i simboli**. Molti di questi errori non
si vedono in fase di build e compaiono solo all'import, che è il motivo per cui
la verifica con `pypresso` è parte della procedura e non un optional.

---

## 7. L'ambiente

`hpc/env_leonardo.sh`, da caricare con `source`, imposta: i moduli (`gcc`,
`cuda/12.6`, `cmake`, `openmpi`, `boost`, `fftw`, `python`), `CC`/`CXX`/`FC`,
`LIBTORCH_ROOT` (il torch del venv, con fallback su `libtorch/`), `CPATH` e
`LIBRARY_PATH` per gli header e le librerie di LibTorch,
`CUDA_TOOLKIT_ROOT_DIR`, e attiva il virtualenv.

Due scelte di compilazione, entrambe reversibili:

- **ESPResSo senza CUDA** (`ESPRESSO_CUDA=ON` per riattivarla): la GPU serve a
  LibTorch, non al motore MD — inferenza e backward stanno nel plugin.
- **waLBerla spento** (`ESPRESSO_WALBERLA=ON`): è il lattice Boltzmann, non
  usato qui, ed è la parte più pesante dell'albero. Riattivandolo va rifatto il
  `configure`.

---

## 8. Dopo l'installazione

```bash
bash hpc/submit_leonardo.sh dataset \
     AA_TOPOLOGY=<topologia> AA_TRAJECTORY=<traiettoria> \
     [AA_FORCES_TRAJECTORY=<traiettoria delle forze>]
bash hpc/submit_leonardo.sh noisefloor
bash hpc/submit_leonardo.sh train
bash hpc/submit_leonardo.sh select
bash hpc/submit_leonardo.sh production
```

### Scegliere il sistema

Di default si lavora su TEL22. `SYSTEM=` sposta l'intera pipeline su un'altra
tutorial directory, da cui derivano anche i nomi dei file prodotti
(`<sistema>_dataset.bin`, `<sistema>_model.pt`, ...):

```bash
bash hpc/submit_leonardo.sh train SYSTEM=tel26
```

Il wrapper rifiuta un `SYSTEM` senza `tutorials/<nome>` ed elenca quelli che
ci sono. Gli stadi `select` e `analysis` si fermano con un errore esplicito
fuori dal TEL22: importano `_tel22_cv`, che codifica 22 nucleotidi e le loro
coordinate collettive, e vanno riscritti per un sistema di taglia diversa.

**A stadi e non tutto insieme, deliberatamente**: ogni passo produce il numero
che decide il successivo. Lo stadio `noisefloor`, subito dopo `dataset`, serve
a due cose: verificare che forze e configurazioni siano **allineate** — il
confronto col controllo shuffled è decisivo, e su una pipeline che appaia
`.xtc` e `.trr` per tempo è l'unico modo di accorgersi di uno sfasamento di un
frame — e vedere a quali distanze i prior lasciano segnale residuo.

L'R² che stampa **non è un tetto per la rete**, malgrado la dicitura che lo
script ha portato a lungo. Misura `E[F·u | r]`, cioè la sola parte radiale,
additiva a coppie e isotropa del segnale; PaiNN vede l'intorno completo e
arriva molto più in alto. Sul TEL26: 0,0050 dallo script, 0,087 di skill al
trainer alla seconda epoca (la *skill* del trainer è esattamente `100 × R²`).

**La selezione del checkpoint non si fa comunque sulla validation loss**, e non
per il rumore: nessuna metrica calcolata sull'ensemble di RIFERIMENTO può
vedere una deriva dell'ensemble del MODELLO. Su quattro modelli con g(r)
misurata le metriche da riferimento li hanno ordinati al contrario — la tabella
è nel commento sopra al calcolo della skill in `training/train_painn.cpp`. La
scelta si fa campionando la dinamica di ogni checkpoint, con lo sweep di
`select`, e il file salvato dall'early stopping non è il candidato migliore.

### Le forze nella traiettoria

```bash
grep -iE 'nstfout|nstxout|^dt|compressed-x-grps' <run>/MD.mdp
```

- `nstfout = 0`: nessun target. Si rigenera con `gmx mdrun -rerun`, che però
  richiede le coordinate di **tutto** il sistema: se la produzione ha salvato
  solo il soluto, serve una nuova produzione.
- `nstxout = 0` con `nstfout > 0`: il `.trr` contiene le sole forze, le
  posizioni stanno nell'`.xtc` (spesso ristretto ai `non-Water`). Si passano
  entrambi: `AA_TRAJECTORY` è l'`.xtc`, `AA_FORCES_TRAJECTORY` il `.trr`,
  `AA_FORCES_TOPOLOGY` il `.tpr` completo. La topologia ridotta per l'`.xtc` si
  ricava con `preprocessing/extract_solute_topology.py`.
- `nstfout` grande (10000 passi = 20 ps): un campione per frame, il target
  resta la forza istantanea. Per abbassare il pavimento di rumore servirebbe
  `nstfout` dell'ordine di 10 passi, da cui mediare su finestre di ~1 ps.

### Costi indicativi su A100

Dai tempi su M3 Max/MPS (training 64/2: 0,64 s/step, 128 s/epoca; CG MD: 82
ms/step) e da un fattore A100/MPS stimato 3-5×, **da verificare sul primo job**:
training C3 40 epoche sul dataset a 801 frame ~21 min (~0,09 nodo-ore); ciclo
completo ~30 min; regola di scala **~0,04 s per frame per epoca**. Oltre
~50 000 frame si sfora il limite di 24 h del Booster: il trainer ha `--resume`.
Il rischio di sforamento non è la GPU ma lo stadio `dataset`, che è CPU e
lineare nei frame: misurarlo su un campione e moltiplicare.

Per misurarlo ci sono `MAX_FRAMES` e `STRIDE`, che `02_build_dataset.sh` passa
a `build_cg_dataset.py`:

```bash
bash hpc/submit_leonardo.sh dataset SYSTEM=<sistema> MAX_FRAMES=200 AA_...
```

Il totale di quel job e' gia' un **limite superiore** utile: l'avvio -- gli
indici delle due traiettorie, che MDAnalysis non mette in cache se la
directory non e' scrivibile -- e' compreso nel tempo misurato, quindi il costo
per frame e' al piu' il totale diviso i frame, e l'estrapolazione sbaglia solo
per eccesso.  Misura reale sul TEL26: 200 frame in 3 min 23 s su DCGP, da cui
6 790 frame in meno di due ore contro le otto di coda.

Un dataset troncato serve alla misura, non al training: il lancio vero
riparte da zero e sovrascrive lo stesso nome, quindi il campione va copiato
altrove se lo si vuole tenere (e' utile per il `noisefloor` preliminare).

---

## 9. I file

| file | ruolo |
|---|---|
| `hpc/submit_leonardo.sh` | wrapper: l'unico posto con account, partizioni, risorse. `AFTER`, `QOS`, `ACCOUNT_*`, `PROJECT_ROOT` |
| `hpc/leonardo_submit.slurm` | corpo dei job, uno `case` per stadio; shell di login (`-l`) perché serve `module` |
| `hpc/env_leonardo.sh` | l'ambiente, da caricare con `source` |
| `hpc/setup_native.sh` | torch, venv, clone di ESPResSo — l'unico passo che scarica |
| `hpc/bootstrap_leonardo.sh` | configure, innesto, build (`STEP=configure\|build\|all`) |
| `simulation/espresso_plugin/install_painn_core_sources.py` | l'innesto nel core, idempotente |
| `simulation/espresso_plugin/myconfig.hpp` | le feature di ESPResSo da compilare: il default piu' `MORSE` |
| `preprocessing/extract_solute_topology.py` | topologia ridotta per un `.xtc` di soli `non-Water` |
| `preprocessing/build_g4_topology.py` | topologia CG di un G-quadruplex: mapping ereditato, registro delle tetradi dalla geometria |
| `tutorials/<sistema>/` | un sistema per directory; `SYSTEM=` sceglie quale |
| `hpc/painn_leonardo.def` | ricetta Apptainer, inutilizzabile qui, tenuta per sistemi con `--fakeroot` |

Monitoraggio: `squeue -u $USER`, poi
`tail -n 20 $(ls -t $PROJECT_ROOT/logs/slurm-*<stadio>-*.out | head -1)`.

I nomi dei job: `mlcg_<stadio>` per `setup`, `configure` e `build`, che
compilano il framework e non guardano il sistema; `<sistema>_<stadio>` per
tutti gli altri.

I log stanno **sempre** in `logs/` sotto la radice del progetto: il wrapper
passa `--output` assoluto, che vince sulla direttiva `#SBATCH` del file di job.
Quest'ultima e' relativa alla directory da cui si lancia `sbatch`, e prima i
log finivano sparsi a seconda di dove ci si trovava. Per cambiarlo: `LOGDIR=`.
Il tempo effettivo di un job finito: `sacct -X -o JobID,Elapsed,MaxRSS,State -j <jobid>`.
