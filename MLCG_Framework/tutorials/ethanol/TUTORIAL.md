# Tutorial: Ethanol Coarse-Graining Pipeline

Benvenuto in questo tutorial! Qui imparerai come eseguire l'intera pipeline MLCG (Machine Learning Coarse-Graining) per l'Etanolo (Multi-Bead), partendo dai dati All-Atom fino ad arrivare alla simulazione in ESPResSo.

In questa cartella troverai uno script automatizzato `run_full_pipeline.sh` che esegue tutti i passaggi mostrati qui sotto in sequenza (addestrando la rete per sole 3 epoche per fare in fretta). Puoi eseguirlo direttamente, oppure seguire manualmente i passaggi.

---

## Passo 1: Preprocessing e Inversione di Boltzmann
L'Etanolo ha 3 siti CG (CH3, CH2, OH) e vogliamo calcolare i legami armonici tra di essi.
Usa lo script `build_cg_dataset.py` fornendogli la traiettoria e il JSON di configurazione (che include il mapping e i bonds da dedurre statisticamente).

```bash
# Esegui dalla root del framework o usa percorsi relativi
python3 ../../preprocessing/build_cg_dataset.py \
    --traj ../../../GROMACS/ethanol.trr \
    --topol ../../../GROMACS/ethanol.gro \
    --config ethanol_topology.json \
    --output my_ethanol_dataset.bin
```
*Cosa succede:* Lo script converte la traiettoria, calcola il prior armonico (salvato in `cg_priors.json`), sottrae analiticamente queste forze dalle forze All-Atom aggregate, e salva tutto nel file binario `my_ethanol_dataset.bin`. Inoltre, salva `rigid_bodies_info.json`.

---

## Passo 2: Addestramento del Modello (C++)
Una volta preparato il dataset e purificate le forze classiche, diamo il binario in pasto alla Rete Neurale.
Assicurati di aver compilato il codice in `training/build/`.

```bash
# Modifica i parametri della rete (es. epochs=3 per un test rapido) in cg_model_config.json
../../training/build/train_painn \
    my_ethanol_dataset.bin \
    my_ethanol_model.pt \
    best_painn_etanolo_config.json
```
*Cosa succede:* Il programma C++ instanzia il modello in LibTorch, addestra le previsioni sulle forze e sui torques dei residui, e compila il modello TorchScript esportandolo come `.pt`.

---

## Passo 3: Dinamica Molecolare in ESPResSo
Ora che abbiamo i Priors Classici (`cg_priors.json`) e il Potenziale Neurale (`my_ethanol_model.pt`), possiamo farli interagire nativamente nel motore di ESPResSo.

```bash
# Usa il binario di ESPResSo (pypresso)
/path/to/espresso/build/pypresso verify_ethanol.py
```
*Cosa succede:* Lo script Python carica il modello, carica i legami armonici in ESPResSo, e verifica che l'integrazione di Newton (con un integratore Simplettico NVE) conservi l'energia totale ($E = E_{kin} + E_{harmonic} + E_{ML}$). Lo script genera anche un plot per validare che l'errore dell'energia scali esattamente con $O(dt^2)$!
