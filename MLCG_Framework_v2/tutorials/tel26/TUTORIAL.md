# TEL26 — G-quadruplex ibrido (3+1), PDB 2JPZ

Secondo sistema del framework, dopo TEL22. Stessa pipeline, tre differenze che
cambiano i comandi: la piega, il formato delle traiettorie di partenza e il
fatto che la topologia CG non è scritta a mano ma **generata**.

| | TEL22 | TEL26 |
|---|---|---|
| PDB di riferimento | 143D | 2JPZ |
| piega | antiparallela (basket UDDU) | ibrida 3+1 |
| nucleotidi per copia | 22 | 26 |
| copie nella scatola | 10 | 10 |
| topologia CG | scritta a mano | generata da `build_g4_topology.py` |
| posizioni AA | `.trr` completo | `.xtc` ristretto ai non-Water |
| forze AA | stesso `.trr` | `.trr` separato (`nstxout=0`) |

Come TEL22, questa directory è un **esempio applicativo**: niente sotto
`preprocessing/`, `training/`, `simulation/` la importa. I file generati —
`tel26_solute.gro`, `tel26_topology.json`, `tel26_dataset.bin`, `cg_priors.json`,
`rigid_bodies_info.json`, `*.pt`, `samples.npz` — sono artefatti di runtime e
non stanno sotto controllo di versione. Sotto controllo di versione ci sono
solo gli script `01`–`05` e `tel26_training_config.json`.

---

## Le traiettorie all-atom

Su Leonardo, prodotte da Giulia Cerrato:

```
$WORK/GiuliaC/machine_learning/Tel26_Hybrid_2jpz_10monomeri/
    box12/    scatola da 12 nm
    box15/    scatola da 15 nm
```

Ciascuna produzione ha, per ogni parte:

- `prod-1.tpr` — il sistema completo, 168979 atomi;
- `prod-1.partNNNN.xtc` — le **posizioni**, 8680 atomi: i `compressed-x-grps`
  tengono solo i non-Water, cioè i 260 nucleotidi (10 × 26) più i 250 ioni K⁺;
- `prod-1.partNNNN.trr` — le **forze**, sul sistema completo.

La separazione posizioni/forze è la conseguenza di `nstxout = 0` con
`nstfout > 0` nel `.mdp`: è quello che rende necessario `--forces-trajectory`
in `build_cg_dataset.py`, che appaia i due file **sul tempo** e non
sull'indice di frame.

> Per le produzioni future conviene chiedere `nstxout = nstfout = 500`: con
> posizioni e forze nello stesso file lo stadio dataset torna a un solo input
> e sparisce la tolleranza temporale.

---

## Gli stadi

### 01 — topologia ridotta e topologia CG

```bash
AA_TPR=$WORK/GiuliaC/machine_learning/Tel26_Hybrid_2jpz_10monomeri/box12/prod-1.tpr \
AA_XTC=$WORK/GiuliaC/machine_learning/Tel26_Hybrid_2jpz_10monomeri/box12/prod-1.part0001.xtc \
    bash 01_prepare_topology.sh
```

Fa due cose.

**La topologia ridotta.** MDAnalysis non accetta `Universe(tpr, xtc)` quando i
due file hanno conteggi di atomi diversi — che è esattamente il caso qui.
`extract_solute_topology.py` estrae dal `.tpr` i soli atomi della selezione,
legge le coordinate del primo frame dell'`.xtc` **senza costruire l'Universe
combinato**, verifica che i conteggi coincidano e scrive `tel26_solute.gro`.
Il `.tpr` serve perché porta le masse del force field, e il mapping CG è un
centro di massa.

Se il conteggio non torna, la selezione non riproduce il gruppo usato in
`compressed-x-grps`: si aggiusta con `SELECTION=...`, non si forza il resto.

**La topologia CG.** `build_g4_topology.py` eredita dal template TEL22 il
mapping per residuo — DA e DT un sito, DG sei: è chimica del nucleotide, non
della piega — e rigenera backbone, angoli e contatti Morse per 26 residui.
L'unica parte che non si deduce dalla sequenza è il **registro delle tetradi**,
che lo script ricava dalla geometria: proietta i centri delle basi sull'asse
del quadruplex (SVD) e raggruppa le guanine **complanari**, non le più vicine —
le più vicine sono quelle impilate (~0.34 nm), che stanno in tetradi diverse.

### Leggere l'output dello stadio 01

Lo script stampa, e va guardato:

```
[INFO] terminali riconosciuti: DT3->DT, DT5->DT
[INFO] residui mappabili: 260  -> 10 copie da 26
[INFO] tetradi ricavate dalla geometria ...
         (3, 9, 15, 21)   fuori piano 0.05 nm, lati 0.61-1.08 nm
[INFO] tratti di guanine: [[4,5,6],[10,11,12],[16,17,18],[22,23,24]]
```

Tre controlli, in ordine di gravità:

1. **260 residui mappabili.** 240 significa che i terminali AMBER (`DA5`,
   `DT3`) non sono stati riconosciuti: sono 2 residui per copia × 10.
2. **Lati fra 0.6 e 1.1 nm, scarto dal piano sotto ~0.15 nm.** Lati di
   0.03–0.20 nm vogliono dire che la struttura non aveva coordinate.
   Guanine complanari stanno a ~0.6 nm, quelle impilate a ~0.34 nm: se i lati
   sono corti lo script ha trovato le pile invece dei piani.
3. **Un tratto controcorrente.** Lo script conta i versi dei quattro tratti di
   guanine: `min(su, giù)` vale 0 per una piega parallela, **1 per l'ibrida
   3+1**, 2 per l'antiparallela. Per il 2JPZ deve dare 1. Un valore diverso
   significa registro sbagliato, e il registro sbagliato produce un campo di
   forza che gira e vincola le basi sbagliate — il caso peggiore, perché non
   fallisce.

Se il registro non convince, si passa a mano:

```bash
TETRADS="3,9,15,21 4,10,16,22 5,11,17,23" bash 01_prepare_topology.sh
```

Non si usa `validate_antiparallel_topology.py`: assume la piega del 143D.

### 02 — dataset CG

```bash
AA_TOPOLOGY=$PWD/tel26_solute.gro \
AA_TRAJECTORY=.../prod-1.part0001.xtc \
AA_FORCES_TRAJECTORY=.../prod-1.part0001.trr \
AA_FORCES_TOPOLOGY=.../prod-1.tpr \
    bash 02_build_dataset.sh
```

Produce `tel26_dataset.bin`, `cg_priors.json`, `rigid_bodies_info.json`.
`AA_FORCES_TOPOLOGY` resta il `.tpr` **completo**: il `.trr` ha tutti gli
atomi, ed è da lì che la selezione estrae le forze del soluto.

I frame senza corrispondenza temporale nel file delle forze vengono saltati:
il conteggio finale stampato è quello dei frame effettivamente usati, non
quello dell'`.xtc`.

### Il pavimento di rumore — prima di allenare

```bash
D=$PWD/tel26_dataset.bin
python3 ../tel22/diagnostics/scripts/33_check_mean_force_signal.py "$D"
python3 ../tel22/diagnostics/scripts/32_measure_noise_floor_local.py "$D"
```

Questi due sono agnostici rispetto al sistema: prendono il dataset come
argomento e basta. È il criterio che decide tutto il resto. Con segnale
scarso la selezione del checkpoint va fatta sulla struttura; con segnale
abbondante la cross-validation sull'errore di forza torna valida.

### 03 — training

```bash
TRAINER=../../training/build/train_painn bash 03_train_model.sh
```

`tel26_training_config.json` parte da 128 canali nascosti e 2 layer, 40 epoche,
batch 4, `spectral_projection_strength: 4.0`, checkpoint ogni 5 epoche. Il
trainer legge quel nome e basta: per allenare una variante si copia il config
scelto su `tel26_training_config.json` prima di lanciare, così il file che ha
prodotto il modello resta accanto al modello.

### 04–05 — equilibrazione e produzione

```bash
PYRESSO=../../espresso/build/pypresso DEVICE=cuda bash 04_equilibrate.sh
PYRESSO=../../espresso/build/pypresso DEVICE=cuda CG_STEPS=20000 bash 05_run_espresso.sh
```

`05` scrive `samples.npz` — senza, la produzione gira ma non lascia niente da
analizzare.

---

## Su Leonardo

Il wrapper di sottomissione prende il sistema come variabile: la tutorial
directory e i nomi dei file prodotti derivano da `SYSTEM`.

```bash
bash hpc/submit_leonardo.sh dataset SYSTEM=tel26 \
     AA_TOPOLOGY=.../tel26_solute.gro \
     AA_TRAJECTORY=.../prod-1.part0001.xtc \
     AA_FORCES_TRAJECTORY=.../prod-1.part0001.trr

bash hpc/submit_leonardo.sh noisefloor SYSTEM=tel26
bash hpc/submit_leonardo.sh train      SYSTEM=tel26
bash hpc/submit_leonardo.sh production SYSTEM=tel26
```

Gli stadi `select` e `analysis` si fermano con un errore esplicito su TEL26:
usano `_tel22_cv`, che codifica i 22 nucleotidi e le loro coordinate
collettive. Vanno riscritti per 26 prima di poterli usare qui.

Lo stadio `01` non passa dallo scheduler: gira in pochi secondi su un nodo di
login, dentro l'ambiente del framework.

```bash
source hpc/env_leonardo.sh
cd MLCG_Framework_v2/tutorials/tel26
AA_TPR=... AA_XTC=... bash 01_prepare_topology.sh
```

Per l'installazione del framework su Leonardo, vedi
[`../../hpc/HOWTO_LEONARDO.md`](../../hpc/HOWTO_LEONARDO.md).

---

## Ordine dei controlli, in breve

| passo | cosa deve risultare | se non risulta |
|---|---|---|
| 01 | 8680 atomi = conteggio dell'`.xtc` | aggiusta `SELECTION` |
| 01 | 260 residui mappabili | terminali AMBER non riconosciuti |
| 01 | lati tetrade 0.6–1.1 nm | la struttura non ha coordinate |
| 01 | un tratto controcorrente | registro sbagliato, passa `TETRADS` |
| 02 | frame usati ≈ frame dell'`.xtc` | tolleranza temporale troppo stretta |
| pre-03 | segnale di forza sopra il pavimento | scegli il checkpoint sulla struttura |
