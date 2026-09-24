# TEL26 — G-quadruplex ibrido (3+1), PDB 2JPZ

> **Obiettivo:** riprodurre la g(r) delle simulazioni atomistiche. Ogni
> scelta — prior, architettura, checkpoint, protocollo — si giudica su quanto
> la g(r) CG, intra e inter e per canale, si avvicina a quella all-atom
> mappata. Stabilità e skill sulle forze sono condizioni necessarie, non il
> bersaglio.

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

### Il termostato: gamma è un attrito, non un tasso

In ESPResSo il Langevin è `m v̇ = F − γ v + rumore`: `gamma` è un coefficiente
d'attrito, e il tempo di rilassamento della velocità è **m/γ**. Con masse di
250–330 amu nelle unità amu–nm–ps, il valore storico `γ = 1` dà **~300 ps**. Su
corse di pochi picosecondi il termostato non agisce e la dinamica è in pratica
microcanonica.

Sul TEL26 D=64 lo si vede dai numeri: in produzione `E_tot` resta quasi
costante mentre `E_ML` cala di 300–500 kJ/mol in 2 ps e `E_kin` sale. Accendendo
il potenziale ML, la configurazione rilassata sotto i soli prior non è un
minimo della nuova Hamiltoniana; il sistema ci scivola, e l'energia rilasciata
resta come calore — gli stati equilibrati uscivano a 1,7–2,2 volte
l'equipartizione, contro l'1,00 dell'equilibrazione classica. Per la stessa
ragione `EQ_DT=0.001` peggiorava le cose: a passi fissi dimezza la durata della
fase 3, l'unica con un termostato efficace (`γ = 50`, τ ≈ 6 ps).

Il corollario conta più del calore: **se il sistema sta ancora migrando verso il
minimo del modello, 2 ps di produzione fotografano un transiente**, non
l'ensemble del modello, e la sovrapposizione della P(r) misura in parte quanto
il sistema si è allontanato dalla struttura di partenza.

Le leve: `EQ_GAMMA` (attrito della fase 4), `EQ_ML_STEPS` (sua durata),
`GAMMA` (produzione), e `RUN_TAG` per non sovrascrivere i file di un'altra corsa
dello stesso modello. L'ensemble di equilibrio non dipende da `γ`, la cinetica
sì: per la validazione strutturale un attrito forte va bene.

```bash
bash hpc/submit_leonardo.sh production SYSTEM=tel26 MODEL=tel26_d64_model.ep1.pt \
     EQ_GAMMA=20 EQ_ML_STEPS=5000 GAMMA=20 CG_STEPS=20000 RUN_TAG=g20
```

### TF32: solo nel training

`"allow_tf32": true` nella config fa usare al trainer i tensor core per le
moltiplicazioni di matrici FP32: fattori arrotondati a 10 bit di mantissa,
accumulo in FP32, stesso intervallo di valori. I pesi salvati restano FP32.

Vale **solo per il training**. Il plugin ESPResSo fissa esplicitamente le
matmul in FP32 pieno, perché in MD le forze sono −∇E via autograd: con i
prodotti arrotondati anche nel passo all'indietro la forza non sarebbe più il
gradiente esatto dell'energia calcolata, e la conservazione dell'energia in
NVE ne risentirebbe. Nel training, invece, un errore relativo di ~10⁻³ è
sepolto dal rumore termico del target. Il plugin lo fissa anche contro la
variabile `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`, che altrimenti lo cambierebbe per
tutto il processo.

Il manifest registra sia `allow_tf32` (la richiesta) sia `tf32_active`
(l'effetto): su MPS e CPU il TF32 non esiste e la richiesta viene ignorata.

**Da sola non accelera molto.** Su 260 molecole con batch 4 il training è
limitato dall'overhead, non dal calcolo — D=64 e D=128 costano uguale. Il TF32
serve quando i kernel sono abbastanza grandi da far lavorare i tensor core,
cioè insieme a un batch più grande. Le config `bench_*` misurano le quattro
combinazioni (batch 4/32, con e senza TF32) su tre epoche:

```bash
for R in bench_b4_tf32 bench_b32 bench_b32_tf32; do
  bash hpc/submit_leonardo.sh train SYSTEM=tel26 RUN=$R
done
```

Il quarto punto, batch 4 in FP32, è il training `d64`. Si confronta il tempo
per epoca nel log. Un batch più grande a parità di learning rate cambia anche
la dinamica dell'ottimizzazione, quindi questi training servono al tempo, non
come modelli.

### 04–05 — equilibrazione e produzione

```bash
PYRESSO=../../espresso/build/pypresso DEVICE=cuda bash 04_equilibrate.sh
PYRESSO=../../espresso/build/pypresso DEVICE=cuda CG_STEPS=20000 bash 05_run_espresso.sh
```

`05` scrive `samples.npz` — senza, la produzione gira ma non lascia niente da
analizzare.

---

## Prior analitici alternativi — studio di fattibilità

Obiettivo: riprodurre la g(r) all-atom mappata con prior analitici non
convenzionali (Morse sito-sito, FENE, LJ attrattivo pair-specific) più il
residuo ML, tenendo per ora le tetradi formate.

### Perché i prior del template non bastano

I contatti delle tetradi ereditati dal TEL22 sono Morse **COM–COM** con
D = 50 kJ/mol e a = 0.3 nm⁻¹: la curvatura al minimo, 2Da² = 9 kJ/mol/nm²,
lascia fluttuare ogni contatto di ~0.5 nm. Nelle produzioni da 100 ps
(γ = 20) il solo-prior si srotola (massa a 1.5–2 nm nel canale intra B3–B3,
overlap 0.15); il D=64 ep1 resta compatto (overlap 0.49) ma perde la struttura
fine: il riferimento ha due picchi B3–B3 netti a 0.45 e 0.65 nm — lato e
diagonale della tetrade, rapporto √2 — e il modello tiene solo il primo. Il
residuo ML, su un segnale vicino al pavimento di rumore, non ricostruisce una
geometria che i prior non suggeriscono affatto.

### Gli insiemi di prior: `PRIOR_SET`

Il dataset **dipende** dai prior: le forze allenate sono il residuo
F_AA − F_prior. Ogni insieme ha quindi topologia, dataset, `cg_priors` e
modelli suoi (`_prior_set.sh`):

| `PRIOR_SET` | topologia | dataset | modelli |
|---|---|---|---|
| vuoto | `tel26_topology.json` | `tel26_dataset.bin` | `tel26_d64_model.pt` |
| `b3morse` | `tel26_topology.b3morse.json` | `tel26_b3morse_dataset.bin` | `tel26_b3morse_d64_model.pt` |

`02` e `03` lo leggono dall'ambiente. `04` e `05` lo ricavano dal manifest del
modello (`dataset_path`) e **si rifiutano** di mettere un residuo ML sopra
prior diversi da quelli su cui è stato allenato: senza questo controllo lo
sbaglio non darebbe alcun errore, solo una fisica sbagliata. Per i controlli
solo-prior (`CLASSICAL=1 DISABLE_ML=1`) il modello non entra nella dinamica e
`PRIOR_SET` va passato esplicitamente; i nomi degli output diventano
`equilibrated_priors_<set>.npz`, `samples_priors_<set>.npz`.

### Passo 1 — Morse sito-sito sulle B3

```bash
python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \
    --topology tel26_topology.json --out tel26_topology.b3morse.json
```

Ricava le tetradi dai grafi K4 dei Morse esistenti, sposta gli estremi sul sito
B3 (Hoogsteen) e stima dal riferimento, per ciascuna delle 18 classi
(tetrade × coppia, mediate sulle 10 copie):

- r0 = mediana della distanza B3–B3;
- a = √(kT / 2Dσ²), con σ = 1.4826·MAD (larghezza del nucleo; `--width std`
  per la deviazione standard, che le code lunghe gonfiano di 2–4 volte);
- r_cut = r0 + 7/a. D resta 50 kJ/mol: i Morse sono reversibili, una tetrade
  può ancora aprirsi.

Controllo da guardare: **diagonale / lato ≈ 1.41** per ogni tetrade. Sul
TEL22 dà 1.42, 1.42, 1.38, con r0 dei lati 0.42–0.58 nm e a fra 1.5 e 5.6
nm⁻¹ — curvature al minimo da 25 a 350 volte quella del template. Scrive anche
`tel26_topology.b3morse.report.json` con tutte le distribuzioni.

Poi lo stesso protocollo del modello canonico, così i numeri sono confrontabili:

```bash
bash hpc/submit_leonardo.sh dataset SYSTEM=tel26 PRIOR_SET=b3morse \
     AA_TOPOLOGY=... AA_TRAJECTORY=... AA_FORCES_TRAJECTORY=... AA_FORCES_TOPOLOGY=...
bash hpc/submit_leonardo.sh noisefloor SYSTEM=tel26 PRIOR_SET=b3morse
bash hpc/submit_leonardo.sh production SYSTEM=tel26 PRIOR_SET=b3morse \
     CLASSICAL=1 DISABLE_ML=1 GAMMA=20 EQ_GAMMA=20 CG_STEPS=100000 RUN_TAG=100ps
bash hpc/submit_leonardo.sh train SYSTEM=tel26 PRIOR_SET=b3morse RUN=d64
```

Il solo-prior viene **prima** del training: se i Morse sito-sito tengono le
tetradi e riproducono i due picchi B3–B3, il residuo ML ha un punto di
partenza sensato; se non li riproducono, il problema è nei prior e allenare
non serve. Il pavimento di rumore va rimisurato: con prior più vicini al
riferimento il residuo si restringe, e il rapporto segnale/rumore cambia.

### Passo 1b — impilamento fra tetradi

Risultato del passo 1 sul TEL26 (seconda metà di 100 ps, soli prior): B3–B3
intra 0,585 contro 0,131 dei prior canonici e 0,493 del canonico + ML; B5–B5
0,720; S–S 0,806; nessun eccesso di contatti fra copie. Le tetradi si formano,
ma **non restano impilate**: nel B5–B5 manca il picco a 0,42 nm delle guanine
sovrapposte, e il B3–B3 intra ha una coda fino a 2 nm dove il riferimento è
nullo oltre 1 nm. Fra due guanine consecutive dello stesso tratto non c'è
nulla che ne orienti le basi: il backbone agisce sui siti S, e la WCA fra
molecole legate è esclusa proprio sulla coppia legata.

```bash
python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \
    --topology tel26_topology.b3morse.json --keep-tetrads \
    --stacking CG_DG_B5 --out tel26_topology.b3stack.json
```

Aggiunge un Morse B5–B5 fra le guanine sovrapposte — stesso tratto, residui
consecutivi, tetradi adiacenti: 8 per copia — stimato come i contatti di
Hoogsteen. I nuovi Morse portano `"role": "stacking"`, così lo script continua
a riconoscere le tetradi dai soli contatti di Hoogsteen. Sul TEL22 alcune
coppie hanno r0 ≈ 0,9 nm invece di ~0,5: nell'antiparallela le guanine
alternano syn e anti e i siti B5 non sono sovrapposti. Il Morse resta un
vincolo reversibile alla geometria del riferimento, non un contatto fisico.

Poi lo stesso protocollo con `PRIOR_SET=b3stack`.

### Passo 2 — repulsione elettrostatica fra i backbone (Debye–Hückel)

Risultato del passo 1b (seconda metà di 100 ps): i **soli prior b3stack**
sono i migliori sui canali strutturali — B3–B3 0,722, B5–B5 0,842, S–S 0,838 —
meglio di ogni corsa con ML. Il ML sopra il b3morse porta il totale a 0,95 ma
**attacca le copie fra loro** (S–S inter 0,2–0,48 fra 0,7 e 1,1 nm, contro 0,04
del riferimento) e peggiora l'S–S intra. Nessuna corsa riproduce la struttura
dell'S–S intra a 1,2–2 nm (loop).

Ai prior manca l'elettrostatica: fra copie e fra siti S lontani agisce solo la
WCA. Il passo 2 aggiunge un Debye–Hückel fra i siti di backbone (sito unico di
DA/DT, sito S di DG), carica −1, λ_D dagli ioni del riferimento:

```bash
python3 add_debye_huckel.py --topology tel26_topology.b3stack.json \
    --out tel26_topology.b3dh.json --ions 250 --box 11.91
```

250 K⁺ in 11,91 nm → I = 0,123 M, λ_D = 0,87 nm a 300 K; U(1 nm) = 0,23 kT.

**Il DH agisce su tutte le coppie cariche, legate comprese.** ESPResSo non
applica le esclusioni di particella all'elettrostatica (il kernel Coulomb è
fuori dal controllo `do_nonbonded`), quindi il builder sottrae la stessa somma
senza esclusioni. Sui primi vicini la forza è ~4 kJ/mol/nm: sposta la
lunghezza del legame di ~10⁻³ nm. La coerenza si verifica con

```bash
pypresso ../../simulation/diagnose_debye_huckel.py --priors cg_priors.b3dh.json
```

che confronta forze ed energie ESPResSo con il kernel del dataset e controlla
che le esclusioni non spengano il DH. Sul TEL22 la sottrazione nel dataset
coincide con il DH ricalcolato indipendentemente entro 2·10⁻⁴ kJ/mol/nm
(precisione float32 del file).

Poi il solito protocollo con `PRIOR_SET=b3dh`.

### Passo 3 — LJ 12-6 al posto dei Morse

Stessi contatti (Hoogsteen B3–B3 e impilamento B5–B5), forma del modello
unfoldable:

```bash
python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \
    --topology tel26_topology.json --form lj --stacking CG_DG_B5 \
    --out tel26_topology.b3stack_lj.json
python3 add_debye_huckel.py --topology tel26_topology.b3stack_lj.json \
    --out tel26_topology.b3dh_lj.json --ions 250 --box 11.91
```

σ = r0/2^{1/6} (minimo in r0), ε = kT r0²/(72 σ_r²) (curvatura al minimo
72 ε/r0² = kT/σ_r²), r_cut = 3,5 σ con energia nulla al cutoff. Nel LJ
profondità e curvatura sono legate: fissata la larghezza osservata la
profondità segue dal riferimento invece di essere scelta (D = 50 kJ/mol dei
Morse). Il confronto `b3dh_lj` contro `b3dh` isola la forma del contatto.

Nel runtime i LJ usano gli stessi marker virtuali dei Morse
(`lennard_jones` fra tipi di marker); nel builder la sottrazione è nel ciclo
dei legami. Verifiche: sul TEL22 la differenza fra dataset con e senza i 260
contatti LJ coincide con il LJ ricalcolato entro 3·10⁻⁶ relativo (float32);
per il runtime

```bash
pypresso ../../simulation/diagnose_pair_specific_lj.py
```

confronta forza, coppia ed energia su due corpi rigidi con il kernel del
builder. Il muro r⁻¹² è molto più ripido del Morse: controllare nel report
del fit che `p01` delle distanze non cada troppo sotto σ.

### Struttura della copia e interazioni fra copie: due problemi separati

Nel riferimento le copie quasi non si toccano (g(r) inter < 0,15 fino a
2 nm): la struttura di un TEL26 e le interazioni fra TEL26 si studiano
separatamente. Il DH a carica piena svuota l'inter fino a 2 nm ma non cambia
l'intra (b3dh sovrapposto a b3stack in tutti i canali): è un parametro
dell'inter, e lo si riprende dopo. Per ora il bersaglio è la **P(r) intra per
canale**.

**Ciclo veloce per i soli prior.** Una corsa con i soli prior non ha bisogno
del dataset: i prior che cambiano sono i contatti pair-specific, e legami,
angoli e WCA restano quelli dell'insieme di partenza.

```bash
python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \
    --topology tel26_topology.json --stacking CG_DG_B5 --stacking-mode core \
    --out tel26_topology.b3core.json
python3 derive_prior_set.py --base b3stack --set b3core
```

`derive_prior_set.py` scrive `cg_priors.b3core.json` e
`rigid_bodies_info.b3core.json` sostituendo tutti i contatti; il dataset
`tel26_b3core_dataset.bin` non esiste, e 04/05 prendono la configurazione
iniziale da `tel26_dataset.bin` finché il ML non è attivo. 03 si rifiuta di
allenare senza il dataset proprio: quando un insieme merita il ML, si
costruisce con lo stadio dataset.

**`--stacking-mode`**: `tract` (default, 8 coppie per copia: guanine
sovrapposte dello stesso tratto), `layers` (48: tutte le coppie di tetradi
diverse), `core` (66: anche nel piano). `core` è una rete elastica sul nucleo
di guanine: vincola l'impilamento fra tratti diversi e l'orientazione delle
basi, cioè i picchi B5–B5 a 0,75–1,27 nm che con `tract` si fondono.
`--stack-D` ne regola la profondità indipendentemente dai contatti di
Hoogsteen.

### Contatti fisici, aggiustati per iterazione

I prior devono corrispondere a interazioni fisiche: i contatti di Hoogsteen
(B3–B3 nel piano) e l'impilamento fra guanine sovrapposte (B5–B5, modo
`tract`), cioè il b3stack. La rete `core` sul nucleo migliora la g(r) ma è
un vincolo elastico fra basi lontane (fino a 1,4 nm), non un'interazione: la
si tiene solo come confronto.

Il difetto del fit per singola distanza (ogni contatto stimato come se fosse
l'unica forza) si corregge per iterazione: una corsa con i soli prior, poi
per ogni classe r0 ← r0 + (mediana_rif − mediana_CG), a ← a·σ_CG/σ_rif (LJ:
σ e ε), tutte insieme.

```bash
python3 iterate_contacts.py --topology tel26_topology.b3stack.json \
    --run samples_priors_b3stack_100ps.npz --out tel26_topology.b3it1.json
python3 derive_prior_set.py --base b3stack --set b3it1
# corsa con i soli prior PRIOR_SET=b3it1, poi b3it1 -> b3it2, ...
```

Criterio di arresto: |Δmediana| < 0,01 nm e |σ_CG/σ_rif − 1| < 10 % per
tutte le classi. Lo spostamento di r0 è limitato a 0,05 nm per iterazione e il
fattore su a a [½, 2]: lontano dal riferimento la correzione non è lineare.

### Passi successivi

2. **FENE** sul backbone al posto dell'armonico (tipo già supportato:
   `k`, `r0`, `r_max`).
3. **LJ attrattivo pair-specific** per i legami a idrogeno, come nel modello
   unfoldable: oggi i LJ del runtime sono solo WCA per coppie di tipi, e un LJ
   fra siti specifici richiede di estendere il meccanismo dei marker usato dai
   Morse.

Ogni nuovo contatto di tetrade deve portare `exclude_wca: false`: per default
qualunque legame che non sia Morse esclude la WCA fra i due siti.

---

## Su Leonardo

Il wrapper di sottomissione prende il sistema come variabile: la tutorial
directory e i nomi dei file prodotti derivano da `SYSTEM`.

```bash
bash hpc/submit_leonardo.sh dataset SYSTEM=tel26 \
     AA_TOPOLOGY=.../tel26_solute.gro \
     AA_TRAJECTORY=.../prod-1.part0001.xtc \
     AA_FORCES_TRAJECTORY=.../prod-1.part0001.trr \
     AA_FORCES_TOPOLOGY=.../prod-1.tpr

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
