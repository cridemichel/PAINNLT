# Tutorial: Ethanol Coarse-Graining Pipeline

Benvenuto in questo tutorial! Qui imparerai come eseguire l'intera pipeline MLCG (Machine Learning Coarse-Graining) per l'Etanolo (Multi-Bead), partendo dai dati All-Atom fino ad arrivare alla simulazione in ESPResSo.

In questa cartella troverai uno script automatizzato `run_full_pipeline.sh` che esegue tutti i passaggi mostrati qui sotto in sequenza (addestrando la rete per sole 3 epoche per fare in fretta). Puoi eseguirlo direttamente, oppure seguire manualmente i passaggi.

---

## Passo 1: Preprocessing e Inversione di Boltzmann (`01_build_dataset.sh`)
L'Etanolo ha 3 siti CG (CH3, CH2, OH) e vogliamo calcolare i legami armonici tra di essi.
Puoi lanciare il primo script:

```bash
./01_build_dataset.sh
```
*Cosa succede:* Lo script usa `build_cg_dataset.py`, converte la traiettoria, calcola il prior armonico (salvato in `cg_priors.json`), sottrae analiticamente queste forze dalle forze All-Atom aggregate, e salva tutto nel file binario `my_ethanol_dataset.bin`. Inoltre, salva `rigid_bodies_info.json`.

---

## Passo 2: Addestramento del Modello (`02_train_model.sh`)
Una volta preparato il dataset e purificate le forze classiche, diamo il binario in pasto alla Rete Neurale.

```bash
./02_train_model.sh
```
*Cosa succede:* Il programma C++ instanzia il modello in LibTorch, addestra le previsioni sulle forze e sui torques dei residui, e compila il modello TorchScript esportandolo come `.pt`.

---

## Passo 3: Dinamica Molecolare in ESPResSo (`03_run_espresso.sh`)
Ora che abbiamo i Priors Classici (`cg_priors.json`) e il Potenziale Neurale (`my_ethanol_model.pt`), possiamo farli interagire nativamente nel motore di ESPResSo.

```bash
./03_run_espresso.sh
```
*Cosa succede:* Lo script Python carica il modello, carica i legami armonici in ESPResSo, e verifica che l'integrazione di Newton (con un integratore Simplettico NVE) conservi l'energia totale ($E = E_{kin} + E_{harmonic} + E_{ML}$). Lo script genera anche un plot per validare che l'errore dell'energia scali esattamente con $O(dt^2)$!
