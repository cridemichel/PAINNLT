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
   sono corti lo script ha trovato le pile invece dei piani. Lati *lunghi*,
   oltre 1.5 nm, insieme a piani mal separati, sono invece la firma delle
   condizioni periodiche: un frame GROMACS e' ripiegato nella scatola e una
   molecola a cavallo di una faccia esce da un lato e rientra dall'altro.
   Lo script ricuce all'immagine minima — prima ogni base, poi la copia —
   quindi il sintomo puo' tornare solo se la struttura non porta la scatola,
   e in quel caso lo dice.
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

`MAX_FRAMES` e `STRIDE` servono a misurare il costo prima di impegnare un
nodo per ore. Misura reale: **200 frame in 3 min 23 s su DCGP**, avvio
compreso — e siccome l'avvio (gli indici delle due traiettorie, che
MDAnalysis non mette in cache perché la directory non è scrivibile) sta
dentro quei 203 s, il totale diviso i frame è un *limite superiore* al costo
per frame, e l'estrapolazione sbaglia solo per eccesso.

Verifica a posteriori: il limite dava ≤ 115 minuti per 6 790 frame, il lancio
completo ne ha impiegati **27 min 41 s** per 6 788 frame, cioè ~0,22 s a
frame. Quasi tutti i 203 secondi del campione erano avvio.

Il lancio completo sovrascrive lo stesso nome, quindi il campione va copiato
altrove se lo si vuole tenere — è utile per il pavimento di rumore
preliminare, che conviene guardare prima di impegnare le due ore.

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

#### Misura su TEL26 a confronto col TEL22

Lo script sottocampiona a ~300 frame **distribuiti su tutta la traiettoria**
(`step = T // NFRAMES`), non ne prende 300 di fila: le due colonne sono quindi
confrontabili, e una misura su un dataset troncato guarda solo l'inizio della
produzione.

| | TEL22 | TEL26 (6 788 frame) |
|---|---|---|
| RMS forza residua istantanea (kJ/mol/nm) | 887,1 | 721,2 |
| ampiezza del segnale di forza media | 51,1 | 50,9 |
| R² del canale radiale di coppia | 0,0033 | **0,0050** |
| \|z\| massimo dati reali | 11,0 | **21,9** |
| \|z\| massimo controllo shuffled | 2,4 | 2,1 |

La stessa misura sul campione da 200 frame dava R² 0,0064 e |z| 18,7: **la
stima su pochi frame è ottimistica**, perché l'ampiezza si prende come massimo
su quindici bin e un massimo su pochi campioni è distorto verso l'alto. Il
numero da usare è quello del dataset intero.

#### Quel R² è un pavimento, non un tetto

Lo script ha stampato a lungo la dicitura "tetto teorico R2", ed era
sbagliata. Misura `E[F·u | r]`: la frazione di varianza spiegabile da una
funzione **puramente radiale, additiva a coppie e isotropa** della distanza
intermolecolare. PaiNN non è vincolato a quella forma — vede l'intorno locale
completo, gli orientamenti dei corpi rigidi, l'identità dei siti, la geometria
a molti corpi. Per la legge della varianza totale, condizionare su più
informazione spiega almeno altrettanto: `E[F | configurazione]` non può fare
peggio di `E[F·u | r]`.

Misura reale sul TEL26: **0,0050 qui, 0,087 al trainer alla seconda epoca**.
La *skill* che il trainer stampa è esattamente `100 × R²` — `val_zero_f_norm`
e `val_loss_f_norm_avg` sono entrambe medie di quadrati, quindi il loro
rapporto è `MSE_modello / MSE_zero`.

Il rapporto fra i due numeri è informativo: dice quanto del segnale **non**
sta nel canale radiale di coppia, cioè quanto è struttura intramolecolare e
orientazionale che i prior non catturano. Ma come previsione della skill il
numero di questo script è fuori di un fattore venti.

Allora a cosa serve lo script: a verificare l'**allineamento** fra forze e
configurazioni — su una pipeline che appaia `.xtc` e `.trr` per tempo è
l'unico modo di accorgersi di uno sfasamento di un frame — e a vedere **dove**
il segnale residuo è significativo, cioè quali distanze i prior non
descrivono. Non serve a decidere se allenare.

Due verifiche di coerenza fra le due misure: il numero di coppie per bin
cresce di 1,54×, esattamente 309/200; e il |z| massimo passa da 18,7 a 21,9,
contro i 23 attesi da 18,7 × √(309/200). Il segnale cresce come √N, che è
quello che fa una media vera e non fa il rumore — mentre il controllo shuffled
resta inchiodato a 2.

Il controllo shuffled è la verifica che conta per questa pipeline: resta
piatto mentre i dati veri arrivano a |z| = 18,7. Se l'appaiamento per tempo
fra `.xtc` e `.trr` fosse sbagliato anche di un solo frame le due tabelle
sarebbero indistinguibili, quindi questo è il collaudo di
`--forces-trajectory`.

Il TEL26 ha circa il doppio del segnale del TEL22, che si è allenato e ha
superato la certificazione NVE: stesso regime, dalla parte buona. Ma il
99,5% della varianza delle forze **istantanee** resta rumore termico
dell'acqua integrata via. Questo dice che la MSE istantanea è una misura
rumorosa, non che il modello possa imparare poco: la parte sistematica che
PaiNN estrae vale l'8,7% della varianza già alla seconda epoca.

### La selezione del checkpoint

**La validation loss non ordina i checkpoint**, e il motivo non è il rumore.
Il commento in `training/train_painn.cpp`, sopra al calcolo della skill,
riporta una verifica su quattro modelli con g(r) misurata:

| modello | g(r) B3–B3 misurata | accordo dalle forze | dalla g(r) su riferimento |
|---|---|---|---|
| D=32 | 0,587 | 0,000 | 0,035 |
| D=64 | **0,746** (migliore) | 0,357 | 0,358 |
| D=128 | **0,286** (peggiore) | **0,480** (max) | **0,568** (max) |

Le metriche valutate sull'ensemble di riferimento **ordinano al contrario**:
premiano il modello che rompe di più il quadruplex. La ragione è strutturale e
non aggirabile — nessuna metrica calcolata sulle configurazioni di riferimento
può vedere una deriva dell'*ensemble del modello*. Per quella bisogna
campionare il modello, con uno sweep di MD brevi.

Conseguenza pratica: la riga `[Early Stopping] Miglioramento! Modello salvato.`
sceglie `tel26_model.pt` con quel criterio, quindi **quel file non è il modello
da usare**. I candidati sono i checkpoint ogni 5 epoche (`tel26_model.ep*.pt`),
e la scelta si fa dopo.

`stop_when_skill_negative` chiede due epoche negative consecutive. Con 6 788
frame la validazione al 20% ne conta ~1 360 contro i ~160 del TEL22, quindi la
stima della skill è ~3× meno rumorosa e il tripwire molto meno incline a
scattare per caso.

Dettaglio fisico: il TEL26 ha un nucleo repulsivo vero fra 0,32 e 0,45 nm
(z = −5,5 e −6,0) che nel TEL22 non compare. I prior Morse e WCA ereditati
dal template TEL22 assorbono meno bene il corto raggio su questa piega, ed è
da lì che viene il segnale in più da imparare.

### 03 — training

```bash
TRAINER=../../training/build/train_painn bash 03_train_model.sh
```

`tel26_training_config.json` parte da 128 canali nascosti e 2 layer, 40 epoche,
batch 4, `spectral_projection_strength: 4.0`, checkpoint ogni 5 epoche. Il
trainer legge quel nome e basta: per allenare una variante si copia il config
scelto su `tel26_training_config.json` prima di lanciare, così il file che ha
prodotto il modello resta accanto al modello.

### Il primo training, D=128: cosa ha insegnato

La prima config TEL26 era partita da quella del TEL22, con `hidden_channels:
128` e 40 epoche. Lo sweep sugli otto checkpoint periodici, `E_kin` allo stato
iniziale della produzione in unità di equipartizione (1419 kJ/mol):

| checkpoint | `E_kin` / equip. | `max_f` | esito |
|---|---|---|---|
| solo prior | 1,00 | 299 | ok |
| ep5 | 1,62 | 600 | completato, ma deriva |
| ep10 | 6,4 | 34 318 | abort |
| ep15 | 16,6 | 280 035 | abort |
| ep20–ep40 | 9–39 | 13 000–630 000 | abort |

**L'instabilità cresce con le epoche**, mentre la skill sulle forze resta
piatta all'8%: la metrica sul riferimento non vede nulla di tutto questo.
Anche ep5 non è sano — in 2 ps la coppia più vicina diventa B3–B3 a ~0,22 nm
con forze crescenti, la stessa firma dei checkpoint abortiti.

La larghezza 128 era proprio quella che sul TEL22 era risultata la **peggiore**
strutturalmente (commento sopra al calcolo della skill in
`training/train_painn.cpp`), ma la config canonica non lo rifletteva.

### Varianti di training: RUN

```bash
bash hpc/submit_leonardo.sh train SYSTEM=tel26 RUN=d64
```

legge `tel26_training_config.d64.json` e scrive `tel26_d64_model.pt` più i suoi
snapshot. La config canonica **non si sovrascrive**: la produzione valida
l'architettura del modello contro la config, e i modelli già allenati
diventerebbero inutilizzabili. `04` e `05` leggono la config giusta dal
manifest del modello (`config_path`), quindi per simulare una variante basta
passare `MODEL`. Il log del trainer, che ha nome fisso, viene copiato in
`<modello>.training_log.csv`.

La variante `d64` cambia tre cose rispetto alla D=128: `hidden_channels: 64`,
`epochs: 15`, `checkpoint_every_epochs: 1` — la finestra buona, se c'è, sta
nelle prime epoche.

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

### Lo sweep sui checkpoint

Basta passare `MODEL`: i nomi dello stato equilibrato e della traiettoria si
derivano da lì, quindi otto produzioni possono girare in parallelo nella stessa
directory senza pestarsi.

```bash
for EP in 5 10 15 20 25 30 35 40; do
  bash hpc/submit_leonardo.sh production SYSTEM=tel26 \
       MODEL=tel26_model.ep${EP}.pt CG_STEPS=2000
done
```

Ciascuna scrive `equilibrated_tel26_model.ep<N>.npz` e
`samples_tel26_model.ep<N>.npz`. Prima erano fissi su `equilibrated.npz`, e uno
sweep in parallelo faceva partire la produzione di un modello dallo stato
equilibrato di un altro — in silenzio, senza alcun errore.

Prima del dataset intero, il campione per misurare il costo — stesso comando
con `MAX_FRAMES=200` in più:

```bash
bash hpc/submit_leonardo.sh dataset SYSTEM=tel26 MAX_FRAMES=200 AA_...
sacct -X -o JobID,Elapsed,State -j <jobid>
```

Lo stadio `select` si ferma con un errore esplicito su TEL26: usa le coordinate
collettive di `_tel22_cv` — `Q` sul ciclo delle tetradi del 143D, l'RMSD — che
passano da `_hb_common` e richiedono il registro delle tetradi del sistema, non
solo il suo numero di residui.

La **g(r)** invece si può già fare: di specifico al TEL22 c'era solo `NUC`, il
numero di residui per copia, che serve a spezzare le molecole in copie. I
canali di tipo (S, B1–B5) valgono per qualunque sistema, perché sono chimica
del nucleotide e non della piega.

```bash
bash hpc/submit_leonardo.sh analysis SYSTEM=tel26 RUNS="priors=samples_priors.npz ml=samples.npz"
```

`NUC` lo ricava da `tel26_topology.json`, così non può divergere dalla
topologia usata. Con un `--nuc` sbagliato lo script si ferma invece di produrre
intra e inter mescolati, cioè numeri plausibili e falsi.

**Non lanciarlo sul nodo di login**: `load_reference` legge il dataset con un
ciclo Python per molecola — 1,76 milioni di iterazioni sui 6 788 frame del
TEL26 — e supera i 600 s di CPU del login. Lo `--stride` non aiuta, perché
agisce sull'accumulo delle distanze e non sul caricamento.

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
| pre-03 | controllo shuffled piatto, dati reali no | forze e posizioni disallineate |
| pre-03 | tetto R² e ampiezza del segnale | se basso, checkpoint scelto sulla struttura |
