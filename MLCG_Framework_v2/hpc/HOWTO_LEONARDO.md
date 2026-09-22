# Deploy di MLCG_Framework_v2 su Leonardo (CINECA)

Il caso d'uso: le traiettorie all-atom lunghe **sono già su Leonardo**, e va
portato lì solo il framework. Niente da trasferire: il framework si clona, le
dipendenze si scaricano sul cluster.

## Niente container

Leonardo ha SingularityPRO 4.3, ma **non** le mappe subuid/subgid:

```
$ singularity build --fakeroot prova.sif docker://alpine:latest
FATAL: could not use fakeroot: no valid mapping entry found for cdemiche (26638)
```

Senza `--fakeroot` un `.def` con `%post` non è costruibile sul cluster, e con
esso cade l'idea di costruire l'immagine in loco. L'ambiente è quindi
**nativo**: i moduli del sistema per le dipendenze di ESPResSo, una
distribuzione LibTorch C++ scaricata, un virtualenv per il Python.

Conviene anche per un'altra ragione: il Python del framework **non ha bisogno
di torch**. Gli servono MDAnalysis, numpy e scipy; l'inferenza del modello sta
nel plugin C++ di ESPResSo e nel trainer. Quindi al posto di un'immagine da
10 GB basta uno zip da ~2,5 GB, e si usano MPI e Boost ottimizzati del cluster.

`painn_leonardo.def` resta nel repo per chi avesse `--fakeroot` altrove.

## I tre vincoli del sistema

1. **Sul nodo di login ogni processo muore dopo 10 minuti di CPU time**
   (`ulimit -t` → 600). Scaricare ed estrarre LibTorch li supera: è un job.
2. **I nodi di calcolo non raggiungono internet, i nodi di login sì.** La
   partizione `lrd_all_serial` gira *sui* nodi di login (`login08`, `login13`):
   è l'unico posto dove un job può scaricare. Da qui la separazione fra lo
   stadio `setup` (scarica, niente compilazione) e `bootstrap` (compila,
   niente rete), che può così usare i 32 core di DCGP.
3. **Il progetto ISCRA B ha due associazioni**: `IscrB_G4MES` sul Booster (ore
   GPU) e `IscrB_G4MES_0` su DCGP (ore CPU). Compilare e leggere traiettorie
   con MDAnalysis non toccano la GPU: farli su `boost_usr_prod` consuma ore
   A100 per lavoro multicore. Vanno su DCGP.

Nota: `lrd_all_serial` rifiuta l'associazione DCGP con *Invalid account or
account/partition combination*. Lo stadio `setup` non passa quindi alcun
`--account`, salvo che si definisca `ACCOUNT_SERIAL`.

## La procedura

```bash
# 0. l'area di lavoro (NON la home: quota piccola)
BASE=/leonardo_work/IscrB_G4MES/$USER
mkdir -p "$BASE" && cd "$BASE"

# 1. il framework
git clone https://github.com/cridemichel/PAINNLT.git
cd PAINNLT/MLCG_Framework_v2

# 2. LibTorch + venv + sorgente di ESPResSo   (lrd_all_serial, ha la rete)
bash hpc/submit_leonardo.sh setup

# 3. plugin + build di ESPResSo e del trainer  (DCGP, 32 core)
bash hpc/submit_leonardo.sh bootstrap

# 4. il dataset CG dalla traiettoria all-atom  (DCGP)
bash hpc/submit_leonardo.sh dataset \
     AA_TRAJECTORY=/percorso/traiettoria AA_TOPOLOGY=/percorso/topologia

# 5. il pavimento di rumore  (DCGP) — lo stadio che decide come si sceglie il modello
bash hpc/submit_leonardo.sh noisefloor

# 6. training, selezione del checkpoint, produzione  (Booster, 1 GPU)
bash hpc/submit_leonardo.sh train
bash hpc/submit_leonardo.sh select
bash hpc/submit_leonardo.sh production
```

`submit_leonardo.sh` è l'unico posto dove compaiono account, partizioni e
risorse. Per forzare una QOS: `QOS=boost_qos_dbg bash hpc/submit_leonardo.sh train`.

Monitoraggio: `squeue -u $USER`, poi `tail -f slurm-mlcg_<stadio>-<JOBID>.out`.

## L'ambiente nativo

`hpc/env_leonardo.sh` si carica con `source` e definisce tutto: moduli
(`cmake`, `openmpi`, `boost`, `fftw`, `python` — i nomi corti, cioè i default
del sistema), `LIBTORCH_ROOT`, `LD_LIBRARY_PATH` e il virtualenv. Ogni stadio
lo carica da sé, quindi vale anche per i comandi lanciati a mano.

`hpc/setup_native.sh` scarica LibTorch **cxx11-ABI, build cu121**. La scelta
della build CUDA non è libera: una `cu124` richiede driver NVIDIA ≥ 550 mentre
il runtime di sistema qui è CUDA 12.2. Per cambiarla: `LIBTORCH_CUDA=cu124` o
`LIBTORCH_URL=...`.

ESPResSo si compila **senza CUDA** (`ESPRESSO_CUDA=OFF`, il default): la GPU
serve a LibTorch, non al motore MD, perché l'inferenza del modello e la sua
backward stanno nel plugin. Per riattivarla, `ESPRESSO_CUDA=ON`.

ESPResSo è fissato al commit **`84cc1d924`** (`5.0.0-58-g84cc1d924`): il plugin
innesta file dentro `src/core/nonbonded_interactions` e `src/python/espressomd`
e tocca le liste di CMake, e una versione diversa può averli spostati.

## Le forze nella traiettoria

Il dataset è *force matching*: `build_cg_dataset.py` legge le forze frame per
frame. Prima di sottomettere `dataset`:

```bash
grep -iE 'nstfout|nstxout|^dt|compressed-x-grps' /percorso/MD.mdp
```

Se `nstfout = 0` non c'è target e le forze vanno rigenerate con
`gmx mdrun -rerun`, che però richiede le coordinate **di tutto il sistema**:
se la produzione ha salvato solo il soluto (`compressed-x-grps`), il rerun non
è possibile e serve una nuova produzione.

Attenzione anche al caso opposto: con `nstxout = 0` il `.trr` contiene le sole
forze e le coordinate stanno nell'`.xtc`. Le due cose vanno lette da file
diversi e appaiate per tempo.

## A stadi, e non tutto insieme

Deliberatamente. Ogni passo produce un numero che decide il successivo: quanti
frame ha il dataset, quanto segnale c'è, dove cade il picco della skill, quale
checkpoint tiene la struttura.

Lo stadio che conta più di tutti è **`noisefloor`, subito dopo `dataset`**.
Sul dataset a 1001 frame usato finora il segnale di forza media era ~1% della
varianza del target: in quel regime la validation loss non ordina i modelli e
la selezione va fatta sulla struttura, con lo sweep dello stadio `select`. Con
dati alla scala di CGnet (Wang et al. 2019: 10^6 frame su 11 μs per un modello
a 5 bead) la cross-validation sull'errore di forza torna a predire l'errore di
energia libera. Quale regime valga lo dice lo script 33 sul dataset nuovo.

## Varianti di training

`03_train_model.sh` legge `tel22_training_config.json` e basta. Per una
variante si copia il config scelto su quel nome, così il file che ha prodotto
il modello resta accanto al modello:

```bash
cd tutorials/tel22
cp diagnostics/configs/tel22_training_config_C3_spectral4_snapshots.json \
   tel22_training_config.json
bash ../../hpc/submit_leonardo.sh train
```

C3 (`spectral_projection_strength 4.0`) è la variante indicata dalla diagnosi:
il fallimento oltre il minimo della val loss era *forze non limitate*, e la
proiezione spettrale limita la costante di Lipschitz per layer.

## Note su CUDA

Il codice non va toccato. Il trainer preferisce già CUDA e cade su MPS solo su
Apple (`#ifdef __APPLE__`); il driver di simulazione accetta `--device cuda`,
che lo stadio `production` passa. `mps_empty_cache_every_batches` resta nelle
config ed è inerte su CUDA.
