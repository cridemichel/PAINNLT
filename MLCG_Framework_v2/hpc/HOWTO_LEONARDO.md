# Portare MLCG_Framework_v2 su Leonardo (CINECA)

Guida completa: dall'utenza nuova al framework compilato e funzionante.
Scritta ripercorrendo un'installazione reale, quindi ogni vincolo qui elencato
è stato incontrato davvero, non previsto in astratto.

**Il risultato**: `pypresso` con il plugin PaiNN, il trainer `train_painn` e
un ambiente Python con MDAnalysis, tutto nativo, senza container.

---

## 1. Il quadro: perché l'ambiente è fatto così

Quattro vincoli di Leonardo determinano l'intera procedura. Conviene averli
chiari prima di eseguire qualsiasi cosa, perché ogni scelta strana della
guida discende da uno di questi.

### 1.1 Niente container

Leonardo ha SingularityPRO 4.3 in `/usr/bin/singularity` (nessun modulo
`apptainer`), ma **non ha le mappe subuid/subgid**:

```
$ singularity build --fakeroot prova.sif docker://alpine:latest
FATAL: could not use fakeroot: no valid mapping entry found for cdemiche (26638)
```

Senza `--fakeroot` un file `.def` con `%post` non è costruibile sul cluster.
Resterebbe la via di costruire l'immagine altrove e trasferirla (la
conversione da `docker-archive` non richiede privilegi), ma l'ambiente nativo
è più semplice e più veloce, e usa MPI e Boost ottimizzati del cluster.

### 1.2 Il login node uccide i processi lunghi

`ulimit -t` vale **600 secondi di CPU**. Qualsiasi download corposo,
estrazione o compilazione va in un job, non lanciato a mano.

### 1.3 La rete c'è solo sui nodi di login

I nodi di calcolo (DCGP, Booster) non raggiungono internet. La partizione
`lrd_all_serial` gira **sui nodi di login** (`login08`, `login13`), quindi è
l'unico posto dove un job può scaricare qualcosa. Questo divide la procedura
in stadi "con rete" e stadi "con core":

| serve | stadio | dove |
|---|---|---|
| rete | `setup` (torch, pacchetti, clone ESPResSo) | `lrd_all_serial` |
| rete | `configure` (ESPResSo scarica heFFTe, Kokkos, Cabana) | `lrd_all_serial` |
| core | `build` (compilazione) | `dcgp_usr_prod`, 32 core |
| GPU | `train`, `select`, `production` | `boost_usr_prod`, 1 GPU |

Che anche la **configurazione** di ESPResSo richieda rete è la cosa meno
ovvia: ESPResSo 5 tira giù heFFTe, Kokkos e Cabana con `FetchContent`, gli
ultimi due incondizionatamente.

### 1.4 glibc 2.28, e l'ABI di libstdc++

Leonardo è RHEL 8. La distribuzione LibTorch ufficiale è costruita contro una
glibc più recente e il link fallisce così:

```
undefined reference to `log2@GLIBC_2.29'
```

Quei simboli versionati non esistono nella libm del sistema, e `-lm` non
cambia nulla. **Si usano i wheel di PyTorch**, che sono `manylinux_2_28` e
portano la stessa LibTorch C++ con i suoi `share/cmake` dentro
`site-packages/torch`.

Attenzione però a *quale* wheel: se è costruito con
`_GLIBCXX_USE_CXX11_ABI=0`, `TorchConfig.cmake` propaga quel flag via
`TORCH_CXX_FLAGS` a tutto ciò che linka Torch, e il core di ESPResSo si
ritrova con una `std::string` diversa da quella di Kokkos, Cabana e Boost.
Il sintomo è un `undefined symbol` su un nome **senza** il tag `[abi:cxx11]`
mentre la libreria lo definisce **con** quel tag. Il setup stampa
`cxx11 ABI` e deve dire `True`; altrimenti va scelta una versione di torch
più recente con `LIBTORCH_VERSION`.

---

## 2. Prerequisiti

### 2.1 L'utenza deve essere abilitata

Un'utenza appena creata può risultare incompleta: aree di lavoro non
accessibili e job rifiutati. Verifica:

```bash
id                                     # devono comparire i gruppi del progetto
ls -ld /leonardo_work/<PROGETTO>
sbatch --test-only -p lrd_all_serial -t 00:10:00 --wrap=hostname
```

Se `/leonardo_work/<PROGETTO>` dà *Permission denied* mentre le aree di altri
progetti si vedono, o se ogni `sbatch` risponde *Invalid account or
account/partition combination* benché `sacctmgr show assoc user=$USER`
elenchi associazioni valide, l'abilitazione non è completa: scrivi a
`superc@cineca.it`. **Prima però prova da un altro nodo di login**: in un caso
reale il blocco si presentava solo su `login01` e non su `login05`.

### 2.2 I due account

Un progetto ISCRA B ne ha due: `<PROGETTO>` sul Booster (ore GPU) e
`<PROGETTO>_0` su DCGP (ore CPU). Compilare e leggere traiettorie con
MDAnalysis non toccano la GPU: farli sul Booster brucia ore A100 per lavoro
multicore.

`lrd_all_serial` non accetta l'associazione DCGP (*Invalid account or
account/partition combination*), quindi gli stadi che ci girano non passano
`--account` e usano il default dell'utente.

### 2.3 Dove mettere le cose

`$HOME` ha quota piccola. Tutto sotto l'area di progetto:

```bash
BASE=/leonardo_work/<PROGETTO>/$USER
mkdir -p "$BASE"
```

Lo scratch (`$CINECA_SCRATCH`) viene ripulito periodicamente: non è il posto
per un ambiente che si riusa per mesi.

---

## 3. Installazione

### 3.1 Clonare il framework

```bash
cd "$BASE"
git clone https://github.com/cridemichel/PAINNLT.git
cd PAINNLT/MLCG_Framework_v2
```

Non c'è nulla da trasferire dal portatile: dataset, build e traiettorie sono
in `.gitignore`, e tutto il resto si scarica.

### 3.2 I tre stadi

```bash
bash hpc/submit_leonardo.sh setup        # ~5 min:  torch, pacchetti, ESPResSo
bash hpc/submit_leonardo.sh configure    # ~2 min:  CMake + innesto del plugin
bash hpc/submit_leonardo.sh build        # ~15 min: ESPResSo e trainer
```

**Vanno in sequenza**: ognuno ha bisogno del precedente. Per metterli in coda
tutti insieme si usa la dipendenza SLURM:

```bash
S=$(bash hpc/submit_leonardo.sh setup     | awk '/Submitted/{print $4}')
C=$(AFTER=$S bash hpc/submit_leonardo.sh configure | awk '/Submitted/{print $4}')
AFTER=$C bash hpc/submit_leonardo.sh build
```

`AFTER` aggiunge `--dependency=afterok`: se uno stadio fallisce i successivi
non partono e restano in coda come `DependencyNeverSatisfied`, da cancellare
con `scancel`.

Cosa fa ciascuno:

- **`setup`** (`hpc/setup_native.sh`): crea il virtualenv, installa numpy,
  scipy, matplotlib, MDAnalysis, h5py, Cython; installa `torch` dal wheel
  (~780 MB più le librerie CUDA); clona ESPResSo al commit fissato
  `84cc1d924`.
- **`configure`**: configura CMake per ESPResSo (che scarica heFFTe, Kokkos,
  Cabana), **poi** innesta il plugin PaiNN, **poi** riconfigura. L'ordine non
  è arbitrario: `install_switched_morse_nonbonded.py` attiva la feature Morse
  scrivendo nel file di configurazione dentro `espresso/build`, che quindi
  deve già esistere; e l'innesto aggiunge sorgenti che CMake deve raccogliere.
- **`build`**: compila ESPResSo (senza CUDA e senza waLBerla, vedi sotto) e i
  due eseguibili del trainer.

#### Cosa significa "innestare il plugin"

Copiare i file non basta, e nemmeno compilarli. `copy_plugin_files.sh` copia i
sorgenti e poi chiama tre installer idempotenti; quello del core
(`install_painn_core_sources.py`) fa tre cose distinte:

1. aggiunge `PaiNN_ML_Potential.cpp` a `target_sources()` — ESPResSo elenca i
   sorgenti esplicitamente, e un file copiato nella directory non viene
   compilato;
2. aggiunge `find_package(Torch)`, il link a `${TORCH_LIBRARIES}` e la define
   `ESPRESSO_PAINN` nel `CMakeLists.txt` del core;
3. inserisce in `forces.cpp`, dentro `System::calculate_forces()`, la chiamata

   ```cpp
   #ifdef ESPRESSO_PAINN
     if (global_painn_potential) {
       global_painn_potential->calculate_forces(*cell_structure, verlet_criterion);
     }
   #endif
   ```

I primi due, se mancano, danno errori espliciti (`undefined symbol`). **Il
terzo no**: senza quella chiamata ESPResSo compila, linka, importa e simula
senza dire nulla — con i soli prior, come se il modello non ci fosse. È
l'innesto che si nota di meno e conta di più, e per questo va verificato che
una simulazione produca una `E_ML` diversa da zero.

Tutte e tre queste modifiche erano state fatte a mano nell'albero locale e non
erano mai state automatizzate: un albero ricostruito da zero — un altro
cluster, un collega, o lo stesso repo fra sei mesi — non le avrebbe avute.

### 3.3 Verifica

Il log del `build` si chiude con:

```
[bootstrap] verifica
  ok        .../espresso/build/pypresso
  ok        .../espresso/build/src/python/espressomd/painn.so
  ok        .../training/build/train_painn
  ok        painn.so include il supporto ordered-geometry
[bootstrap] bootstrap completato
```

Prova diretta del plugin (l'ambiente va **sempre** caricato prima, altrimenti
`pypresso` non trova `libboost_mpi`):

```bash
source hpc/env_leonardo.sh
espresso/build/pypresso -c "
import espressomd.painn as p, inspect
print(sorted(inspect.signature(p.activate_painn_potential).parameters))"
```

---

## 4. L'ambiente

`hpc/env_leonardo.sh` si carica con `source` e definisce tutto. Serve nei job
(lo fanno da sé) e in ogni comando lanciato a mano.

Cosa imposta, e perché:

| cosa | perché |
|---|---|
| moduli `gcc`, `cuda`, `cmake`, `openmpi`, `boost`, `fftw`, `python` | il gcc di sistema è 8.5.0 e ESPResSo richiede ≥ 12.2; boost/fftw/openmpi sono compilati con gcc 12.2.0 |
| `CC`, `CXX`, `FC` | senza, CMake trova `/usr/bin/gcc` e si ferma con *Unsupported compiler GNU 8.5.0* |
| `LIBTORCH_ROOT` | preferisce il torch del venv, ricade su `libtorch/` se presente |
| `CPATH`, `LIBRARY_PATH` | il CMakeLists del core linka `"${TORCH_LIBRARIES}"`, una lista di percorsi e non un target: non propaga le include directory, e il modulo Cython fallisce con *fatal error: torch/torch.h* |
| `CUDA_TOOLKIT_ROOT_DIR` | `TorchConfig.cmake` include `Caffe2Config`, che per una LibTorch CUDA pretende il toolkit — **anche sui nodi DCGP, che GPU non ne hanno** |
| venv attivo | MDAnalysis e il Python di ESPResSo |

Scelte di compilazione, entrambe reversibili:

- **ESPResSo senza CUDA** (`ESPRESSO_CUDA=ON` per riattivarla): la GPU serve a
  LibTorch, non al motore MD — inferenza e backward stanno nel plugin.
- **waLBerla spento** (`ESPRESSO_WALBERLA=ON` per riattivarlo): è il lattice
  Boltzmann, che questo framework non usa, ed è la parte più pesante
  dell'albero sia da scaricare sia da compilare. Riattivandolo va rifatto il
  `configure`.

---

## 5. Diario degli errori, con i rimedi

Ogni riga è un errore realmente incontrato durante l'installazione.

| errore | causa | rimedio |
|---|---|---|
| `Unable to locate a modulefile for 'apptainer'` | su Leonardo il runtime è `singularity`, di sistema | gli script cercano il runtime invece di assumerlo — ma vedi la riga dopo |
| `could not use fakeroot: no valid mapping entry` | niente subuid/subgid | ambiente nativo, non container |
| `Invalid account or account/partition combination` | associazione DCGP su `lrd_all_serial`, oppure utenza non abilitata | non passare `--account` su `lrd_all_serial`; se persiste su tutte le partizioni, è l'utenza |
| `Unsupported compiler GNU 8.5.0` | modulo `gcc` non caricato | `env_leonardo.sh` lo carica e fissa `CC`/`CXX` |
| `CXX compiler changed` | cache di CMake di un tentativo precedente | il bootstrap rigenera la build directory da sé |
| `ESPResSo build directory not found` | plugin innestato prima di configurare | `configure` fa CMake → plugin → CMake |
| `Failed to connect to github.com port 443` | `FetchContent` su un nodo di calcolo | `configure` su `lrd_all_serial` |
| `fatal error: torch/torch.h` | include non propagati al modulo Cython | `CPATH` |
| `Caffe2: CUDA cannot be found` | `TorchConfig` vuole il toolkit | modulo `cuda` in `env_leonardo.sh` |
| `undefined reference to cudaGetDriverEntryPointByVersion@libcudart.so.12` | modulo CUDA più vecchio della CUDA del wheel torch: quella API esiste da 12.5, il default di Leonardo è 12.2 | `MODULE_CUDA=cuda/12.6`, allineato a `LIBTORCH_CUDA=cu126` |
| `undefined reference to log2@GLIBC_2.29` | zip LibTorch contro glibc più recente della 2.28 di RHEL 8 | torch dal wheel (`LIBTORCH_SOURCE=pip`) |
| `No rule to make target .../libkineto.a` | cache del trainer che punta a un'altra LibTorch | il bootstrap rigenera `training/build` quando `MLCG_TORCH_ROOT` cambia |
| `libboost_mpi.so.1.85.0: cannot open shared object file` | `pypresso` lanciato senza ambiente | `source hpc/env_leonardo.sh` |
| `painn.so: undefined symbol: global_painn_potential` | `PaiNN_ML_Potential.cpp` copiato nella directory ma non elencato in `target_sources()`, quindi non compilato nel core | `install_painn_core_sources.py`, chiamato da `copy_plugin_files.sh` |
| `espresso_core.so: undefined symbol: _ZTIN3c105ErrorE` | il core non linkava LibTorch: `find_package(Torch)` e `${TORCH_LIBRARIES}` erano modifiche a mano nell'albero del Mac | idem |
| `undefined symbol: ...SharedAllocationRecordCommon<HostSpace>::get_labelEv` mentre la libreria definisce `get_label[abi:cxx11]()` | mismatch di ABI di libstdc++: un wheel PyTorch con `_GLIBCXX_USE_CXX11_ABI=0` propaga quel flag via `TORCH_CXX_FLAGS` a tutto ciò che linka Torch, e il core finisce con una `std::string` diversa da quella di Kokkos, Cabana e Boost | usare un wheel con la ABI nuova (`torch._C._GLIBCXX_USE_CXX11_ABI` deve essere `True`); lo zip `cxx11-abi` lo sarebbe, ma non parte per via della glibc |
| `painn.so senza ordered-geometry` (falso allarme) | `strings` non attraversa le tabelle di stringhe di Cython | il controllo usa `grep -a` |

---

## 6. Dopo l'installazione: la pipeline

```bash
bash hpc/submit_leonardo.sh dataset \
     AA_TOPOLOGY=<topologia> AA_TRAJECTORY=<traiettoria> \
     [AA_FORCES_TRAJECTORY=<traiettoria delle forze>]
bash hpc/submit_leonardo.sh noisefloor
bash hpc/submit_leonardo.sh train
bash hpc/submit_leonardo.sh select
bash hpc/submit_leonardo.sh production
```

**A stadi e non tutto insieme, deliberatamente**: ogni passo produce il numero
che decide il successivo — quanti frame ha il dataset, quanto segnale c'è,
dove cade il picco della skill, quale checkpoint tiene la struttura. Una
pipeline che gira dritta fino in fondo restituisce un risultato che non si sa
leggere.

Lo stadio che conta più di tutti è **`noisefloor`, subito dopo `dataset`**.
Sul dataset TEL22 a 1001 frame il segnale di forza media era ~1% della
varianza del target: in quel regime la validation loss non ordina i modelli e
la selezione va fatta sulla struttura, con lo sweep di `select`. Con dati alla
scala di CGnet (Wang et al. 2019: 10⁶ frame su 11 μs per un modello a 5 bead)
la cross-validation sull'errore di forza torna a predire l'errore di energia
libera. Quale regime valga lo dice lo script 33 sul dataset nuovo: non si
assume.

### Le forze nella traiettoria

Il dataset è *force matching*. Prima di sottomettere `dataset`:

```bash
grep -iE 'nstfout|nstxout|^dt|compressed-x-grps' <run>/MD.mdp
```

- `nstfout = 0`: non ci sono forze, non c'è target. Si rigenerano con
  `gmx mdrun -rerun`, che però richiede le coordinate **di tutto il sistema**:
  se la produzione ha salvato solo il soluto, serve una nuova produzione.
- `nstxout = 0` con `nstfout > 0`: il `.trr` contiene le sole forze e le
  posizioni stanno nell'`.xtc`, spesso ristretto ai `non-Water`. Le due cose
  si leggono da file diversi: `AA_TRAJECTORY` è l'`.xtc`,
  `AA_FORCES_TRAJECTORY` il `.trr`, e `AA_FORCES_TOPOLOGY` il `.tpr`
  completo. La topologia ridotta per l'`.xtc` si ricava con
  `preprocessing/extract_solute_topology.py`.
- `nstfout` grande (es. 10000 passi = 20 ps): un solo campione per frame, il
  target resta la forza istantanea. Per abbassare il pavimento di rumore
  servirebbe `nstfout` dell'ordine di 10 passi, da cui mediare su finestre di
  ~1 ps.

---

## 7. Costi indicativi su A100

Dai tempi misurati su M3 Max/MPS (training 64/2: 0,64 s/step, 128 s/epoca;
CG MD: 82 ms/step) e da un fattore A100/MPS stimato 3-5× — il rapporto di
banda di memoria, perché il message passing è scatter/gather-bound —
**da verificare sul primo job reale**:

- training C3, 40 epoche, dataset a 801 frame: ~21 min (~0,09 nodo-ore)
- ciclo completo train + select + produzione 10 ps: ~30 min (~0,13 nodo-ore)
- regola di scala: **~0,04 s per frame per epoca**

Oltre ~50 000 frame si sfora il limite di 24 h del Booster: il trainer ha
`--resume` con validazione del manifest. `batch_size: 4` lascia l'A100 quasi
ferma e su dataset grandi alzarlo vale un altro 2-3×, ma cambia la traiettoria
di ottimizzazione. Il rischio di sforamento non è la GPU ma lo stadio
`dataset`, che è CPU e lineare nei frame: misuralo su 500 frame e moltiplica.

---

## 8. I file

| file | ruolo |
|---|---|
| `hpc/submit_leonardo.sh` | wrapper: l'unico posto con account, partizioni, risorse. `AFTER`, `QOS`, `ACCOUNT_*`, `PROJECT_ROOT` |
| `hpc/leonardo_submit.slurm` | corpo dei job, uno `case` per stadio; shell di login (`-l`) perché serve `module` |
| `hpc/env_leonardo.sh` | l'ambiente, da caricare con `source` |
| `hpc/setup_native.sh` | torch, venv, clone di ESPResSo (l'unico passo che scarica) |
| `hpc/bootstrap_leonardo.sh` | configure, innesto del plugin, build (`STEP=configure\|build\|all`) |
| `hpc/painn_leonardo.def` | ricetta Apptainer, inutilizzabile qui, tenuta per sistemi con `--fakeroot` |

Monitoraggio: `squeue -u $USER`, poi
`tail -n 20 $(ls -t slurm-mlcg_<stadio>-*.out | head -1)`.
