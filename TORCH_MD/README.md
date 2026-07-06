# TorchMD Coarse-Graining Pipeline

Questa directory contiene i template per eseguire l'intera pipeline di Coarse-Graining (Mapping, Training, Simulazione) utilizzando l'ecosistema **TorchMD** al posto dell'integrazione C++ ESPResSo.

## Prerequisiti

Essendo un progetto basato interamente su PyTorch, devi installare i pacchetti ufficiali. Puoi farlo nel tuo ambiente conda/venv:
```bash
pip install torch
pip install git+https://github.com/torchmd/torchmd-net.git
pip install git+https://github.com/torchmd/torchmd.git
pip install MDAnalysis numpy
```

## Flusso di Lavoro

### 1. Preparazione Dataset (`prepare_dataset.py`)
Come per l'altra pipeline, leggiamo la traiettoria All-Atom e calcoliamo i centri di massa. La differenza è che questo script sottrae i *Priors Classici* (WCA) e salva un file `.npz` nativamente digeribile da `torchmd-net` al posto di un file binario C++.
```bash
python prepare_dataset.py -c ../GROMACS/conf.gro -f ../GROMACS/traiettoria.trr -m ../python_scripts/cg_mapping.json -p ../GROMACS/priors.json -o dataset.npz
```

### 2. Addestramento Rete Neurale (`train.yaml`)
Invece di usare `cg_painn_train` scritto in C++, deleghiamo il training direttamente a `torchmd-net`. Nel file `train.yaml` è configurato un *Equivariant Transformer* (più moderno di PaiNN) che impara a minimizzare l'errore sulle forze "Delta".
```bash
python -m torchmdnet.scripts.train --conf train.yaml
```
I checkpoint verranno salvati nella cartella `checkpoints/`.

### 3. Simulazione 100% su GPU (`simulate.py`)
Lo script `simulate.py` è il cuore pulsante della dinamica molecolare su TorchMD.
Inizializza le posizioni (estraendole da `dataset.npz`), carica il modello addestrato (es. `epoch=100.ckpt`), aggiunge "al volo" il potenziale repulsivo WCA su GPU ed esegue l'integrazione (Dinamica di Langevin o Verlet) interamente in memoria video.

```bash
python simulate.py --model checkpoints/epoch=100.ckpt --dataset dataset.npz --steps 5000
```

> **Vantaggio Principale:** A differenza della pipeline ESPResSo, qui le posizioni e le forze non viaggiano mai dalla GPU (dove si fa AI) alla CPU (dove gira l'integratore C++). Tutto, dal calcolo della fisica classica al calcolo neurale, avviene in tensori paralleli su CUDA/MPS, massimizzando le performance per architetture AI.
