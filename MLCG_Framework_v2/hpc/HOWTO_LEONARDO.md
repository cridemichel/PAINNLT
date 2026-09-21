# Deploy di MLCG_Framework_v2 su Leonardo (CINECA)

Il caso d'uso: le traiettorie all-atom lunghe **sono già su Leonardo**, e va
portato lì solo il framework. Niente da trasferire: il framework si clona, e
l'immagine del container si costruisce sul cluster.

## I quattro vincoli del sistema

Sono quelli che decidono la forma della procedura. Vale la pena averli in
mente prima di eseguire qualsiasi cosa.

1. **Sul nodo di login ogni processo muore dopo 10 minuti di CPU time**
   (`ulimit -t` → 600). Costruire il `.sif` dell'immagine PyTorch *devel*
   (~10 GB da scaricare, decomprimere e ricomprimere in squashfs) li supera.
   Quindi la costruzione è un job, non un comando interattivo.
2. **I nodi di calcolo non raggiungono internet, i nodi di login sì.** La
   partizione `lrd_all_serial` gira *sui* nodi di login (`login08`, `login13`),
   quindi è l'unico posto dove un job può scaricare l'immagine base da Docker
   Hub. Per lo stesso motivo il clone di ESPResSo lo fa il wrapper sul nodo di
   login, prima di sottomettere la compilazione: così quest'ultima può girare
   su DCGP, che ha molti più core ma nessuna rete.
3. **Il progetto ISCRA B ha due associazioni**: `IscrB_G4MES` sul Booster (le
   ore GPU) e `IscrB_G4MES_0` su DCGP (le ore CPU). Compilare ESPResSo e
   leggere una traiettoria con MDAnalysis non toccano la GPU: farli su
   `boost_usr_prod` consuma ore A100 per lavoro interamente seriale o
   multicore. Vanno su DCGP.
4. **`$HOME` ha una quota piccola.** Immagine, cache di Apptainer, build di
   ESPResSo e checkpoint stanno sotto l'area di progetto in `/leonardo_work`,
   non in home. Lo scratch è un'alternativa ma viene ripulito periodicamente:
   non è il posto dove tenere un'immagine e una build che si riusano per mesi.

## Prerequisito: l'area di progetto deve essere accessibile

```bash
id                                  # devono comparire i gruppi IscrB_G4MES*
ls -ld /leonardo_work/IscrB_G4MES
```

Se questo dà *Permission denied* mentre `id` mostra i gruppi giusti, non è un
problema di configurazione: è la mappa delle identità dei server Lustre non
ancora allineata, cosa che capita nelle prime ore dopo l'attivazione di
un'utenza. Si risolve da sola, o con una segnalazione a `superc@cineca.it`.
Fino ad allora non c'è dove mettere l'immagine, e non si parte.

## La procedura

```bash
# 0. l'area di lavoro
BASE=/leonardo_work/IscrB_G4MES/$USER
mkdir -p "$BASE" && cd "$BASE"

# 1. il framework
git clone https://github.com/cridemichel/PAINNLT.git
cd PAINNLT/MLCG_Framework_v2

# 2. l'immagine del container  (job su lrd_all_serial, ~1 ora)
bash hpc/submit_leonardo.sh image

# 3. ESPResSo + plugin + trainer  (clone qui, compilazione su DCGP, ~1 ora)
bash hpc/submit_leonardo.sh bootstrap

# 4. il dataset CG dalla traiettoria all-atom  (DCGP)
bash hpc/submit_leonardo.sh dataset \
     AA_TRAJECTORY=/percorso/md.trr AA_TOPOLOGY=/percorso/md.gro

# 5. il pavimento di rumore  (DCGP) — vedi sotto, è lo stadio che conta
bash hpc/submit_leonardo.sh noisefloor

# 6. training, selezione del checkpoint, produzione  (Booster, 1 GPU)
bash hpc/submit_leonardo.sh train
bash hpc/submit_leonardo.sh select
bash hpc/submit_leonardo.sh production
```

`submit_leonardo.sh` è l'unico posto dove compaiono account, partizioni e
risorse: assegna a ogni stadio quelle giuste e sottomette
`leonardo_submit.slurm`, che contiene solo il corpo dei job. Per forzare una
QOS diversa: `QOS=boost_qos_dbg bash hpc/submit_leonardo.sh train`.

Monitoraggio:

```bash
squeue -u $USER
tail -f slurm-mlcg_<stadio>-<JOBID>.out
```

## Le forze nella traiettoria

Il dataset è *force matching*: `build_cg_dataset.py` legge le forze frame per
frame. Una produzione GROMACS scritta con `nstfout=0` contiene solo coordinate
e non è utilizzabile. Da verificare **prima** di sottomettere lo stadio
`dataset`:

```bash
gmx check -f /percorso/md.trr      # deve elencare i frame di forza
grep -i nstfout /percorso/*.mdp
```

Se le forze non ci sono, si rigenerano con `gmx mdrun -rerun` sulla traiettoria
esistente: è un job GPU, e va messo in conto perché diventa il primo consumo di
ore del progetto.

## A stadi, e non tutto insieme

Deliberatamente. Ogni passo produce un numero che decide il successivo: quanti
frame ha il dataset, quanto segnale c'è, dove cade il picco della skill, quale
checkpoint tiene la struttura. Una pipeline che gira dritta fino in fondo
restituisce un risultato che non si sa leggere.

Lo stadio che conta più di tutti è **`noisefloor`, subito dopo `dataset`**.
Sul dataset a 1001 frame usato finora il segnale di forza media è ~1% della
varianza del target: in quel regime la validation loss non ordina i modelli — è
per il 98% rumore irriducibile — e la selezione va fatta sulla struttura, con
lo sweep dello stadio `select`. Con dati alla scala di CGnet (Wang et al. 2019:
10^6 frame su 11 μs per un modello a 5 bead) la cross-validation sull'errore di
forza torna a predire l'errore di energia libera, e la selezione diventa molto
più economica. Quale dei due regimi vale, lo dice lo script 33 sul dataset
nuovo — non si può assumere.

## Varianti di training

`03_train_model.sh` legge `tel22_training_config.json` e basta. Per allenare
una variante si copia il config scelto su quel nome, così il file che ha
prodotto il modello resta accanto al modello:

```bash
cd tutorials/tel22
cp diagnostics/configs/tel22_training_config_C3_spectral4_snapshots.json \
   tel22_training_config.json
bash ../../hpc/submit_leonardo.sh train
```

C3 (`spectral_projection_strength 4.0`) è la variante indicata dalla diagnosi
su TEL22: il fallimento osservato oltre il minimo della val loss è *forze non
limitate*, e la proiezione spettrale limita direttamente la costante di
Lipschitz per layer.

## Versione di ESPResSo e di PyTorch

ESPResSo è fissato al commit **`84cc1d924`** (`5.0.0-58-g84cc1d924`): il plugin
innesta file dentro `src/core/nonbonded_interactions` e `src/python/espressomd`
e tocca le liste di CMake, e una versione diversa può averli spostati o
cambiato le firme.

Il tag PyTorch è un argomento di `painn_leonardo.def`
(`TORCH_TAG=2.5.1-cuda12.4-cudnn9-devel`). I checkpoint sono archivi
TorchScript: per caricare su Leonardo un modello allenato sul Mac (torch 2.10)
serve un torch non più vecchio, e viceversa. Allenando da zero qui e analizzando
i `samples.npz` (che sono numpy) la versione è indifferente.

## Note su CUDA

Il codice non va toccato. Il trainer preferisce già CUDA e cade su MPS solo se
CUDA manca (`train_painn.cpp`); il driver di simulazione accetta
`--device cuda`, che lo stadio `production` passa. Il parametro
`mps_empty_cache_every_batches` resta nelle config ed è inerte su CUDA.
