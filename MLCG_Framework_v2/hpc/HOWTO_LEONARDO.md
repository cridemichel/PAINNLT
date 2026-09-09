# Deploy di MLCG_Framework_v2 su Leonardo (CINECA)

Il caso d'uso: le traiettorie all-atom lunghe **sono già su Leonardo**, e va
portato lì solo il framework. Questo cambia il piano rispetto alla versione
precedente di questa guida, che trasferiva un tar da 10 GB.

## Perché non costruire l'immagine con Docker sul Mac

Su Apple Silicon `docker build` produce un'immagine **arm64**, e Leonardo è
**x86_64**: quell'immagine non parte. Per ottenerne una giusta servirebbe
`docker build --platform linux/amd64`, che gira in emulazione QEMU — su
un'immagine PyTorch `devel` da ~10 GB sono decine di minuti, seguiti da un
`docker save` e da un trasferimento da 10 GB.

Apptainer su Leonardo scarica l'immagine base da Docker Hub da sé, nativamente.
Quindi si costruisce là, e sul Mac Docker non serve affatto.

## Cosa va trasferito

Niente, in pratica: si clona.

- **Il framework** è su GitHub. Un `git clone` porta esattamente i file
  versionati ed esclude dataset, build e traiettorie, che sono in `.gitignore`.
- **ESPResSo non è tracciato** nel repo: è un clone separato di
  `espressomd/espresso`. Il bootstrap lo clona al commit su cui il plugin è
  stato validato, **`84cc1d924`** (`5.0.0-58-g84cc1d924`). La versione va
  fissata: il plugin innesta file dentro `src/core/nonbonded_interactions` e
  `src/python/espressomd`, e una versione diversa può averli spostati o
  cambiato le firme.

## Attenzione alle quote

`$HOME` su Leonardo ha una quota piccola. L'immagine `.sif` pesa alcuni GB e la
build di ESPResSo altrettanto: mettere entrambe sotto `$CINECA_SCRATCH` o
`$WORK`. Gli script assumono `$CINECA_SCRATCH` e ricadono su `$HOME` solo se
non è definito.

## Passi

### 1. Verifica che si possa costruire

```bash
ssh <utente>@login.leonardo.cineca.it
module load apptainer
apptainer build --fakeroot /tmp/prova.sif docker://alpine:latest && echo OK
```

Se questo non funziona — nodo di login senza accesso a Docker Hub, o
`--fakeroot` non permesso — fermati qui e dimmelo: la strada diventa il
cross-build con Docker, che è più lenta ma percorribile.

### 2. Clona il framework

```bash
cd "$CINECA_SCRATCH"
git clone https://github.com/cridemichel/PAINNLT.git
```

### 3. Costruisci l'immagine

```bash
cd "$CINECA_SCRATCH"
apptainer build --fakeroot painn.sif PAINNLT/MLCG_Framework_v2/hpc/painn_leonardo.def
```

Il tag PyTorch è un argomento del file `.def`. Vale la pena scegliere una
versione **non più vecchia** di quella del Mac (attualmente 2.10) se vuoi che i
checkpoint TorchScript siano caricabili nei due sensi. Se alleni da zero su
Leonardo e analizzi i `samples.npz`, che sono numpy, la versione è indifferente.

### 4. Personalizza lo SLURM

In `leonardo_submit.slurm` c'è **una sola cosa obbligatoria**: sostituire
`--account=CAMBIA_QUESTO` con il tuo account CINECA. I percorsi si ricavano da
`$CINECA_SCRATCH` e si possono sovrascrivere con `PROJECT_ROOT` e `IMAGE`.

### 5. Esegui a stadi

```bash
cd "$CINECA_SCRATCH/PAINNLT"
sbatch --export=ALL,STAGE=bootstrap MLCG_Framework_v2/hpc/leonardo_submit.slurm
```

`bootstrap` clona ESPResSo, innesta il plugin e compila ESPResSo e il trainer.
Verifica alla fine che `pypresso`, `painn.so` e `train_painn` esistano, e
segnala se `painn.so` non include il supporto ordered-geometry.

Poi, uno stadio alla volta:

```bash
sbatch --export=ALL,STAGE=dataset,AA_TRAJECTORY=/percorso/md_whole.trr,AA_TOPOLOGY=/percorso/md.gro \
       MLCG_Framework_v2/hpc/leonardo_submit.slurm
sbatch --export=ALL,STAGE=noisefloor  MLCG_Framework_v2/hpc/leonardo_submit.slurm
sbatch --export=ALL,STAGE=train       MLCG_Framework_v2/hpc/leonardo_submit.slurm
sbatch --export=ALL,STAGE=production  MLCG_Framework_v2/hpc/leonardo_submit.slurm
sbatch --export=ALL,STAGE=select      MLCG_Framework_v2/hpc/leonardo_submit.slurm
```

**A stadi e non tutto insieme, deliberatamente.** Ogni passo produce un numero
che decide il successivo: quanti frame ha il dataset, quanto segnale c'è, dove
cade il picco, quale checkpoint tiene la struttura. Una pipeline che gira dritta
fino in fondo restituisce un risultato che non si sa leggere.

### 6. Lo stadio che conta più di tutti

**`noisefloor`, subito dopo `dataset`.** È il criterio guida che determina tutto
il resto, e va misurato prima di allenare.

Sul dataset a 1001 frame usato finora, il segnale di forza media è ~1% della
varianza del target. In quel regime la validation loss non ordina i modelli — è
per il 98% rumore irriducibile — e la selezione va fatta sulla struttura, con lo
sweep dello stadio `select`.

Con dati alla scala di CGnet la situazione cambia: Wang et al. 2019 usano
**1 000 000 di frame** su 11 μs per un modello a 5 bead, e verificano che la
cross-validation sull'errore di forza predice l'errore di energia libera — «i
minimi nella differenza di energia libera corrispondono ai minimi nelle curve di
cross-validation». Nello stesso articolo avvertono che il CG force matching
richiede molti più dati del force matching ordinario, perché la forza media va
appresa vicino a ogni configurazione.

Quindi: se sulle traiettorie lunghe il segnale sale, torni nel regime dove il
criterio della letteratura funziona, e la selezione diventa molto più economica.
Se resta all'1%, valgono le conclusioni raccolte sul dataset corto — inclusa
quella che la capacità ha un optimum e che la skill sulle forze può ordinare al
contrario.

## Note su CUDA

Il codice non va toccato. Il trainer preferisce già CUDA e cade su MPS solo se
CUDA manca (`train_painn.cpp`), e il driver di simulazione accetta
`--device cuda`, che lo stadio `production` passa. Il parametro
`mps_empty_cache_every_batches` resta nelle config ed è inerte su CUDA.

## Monitoraggio

```bash
squeue -u $USER
tail -f slurm-mlcg_tel22-<JOBID>.out
```
