# TorchMD-Net Workflow - Guida Rapida (Apple Silicon MPS)

Questo documento riassume i passaggi fondamentali per eseguire il workflow di simulazione molecolare neurale basata su TorchMD-Net, specificamente adattato per il sistema Apple Silicon (M-series).

## 1. Preparazione del Dataset (Estrazione da GROMACS)
Per prima cosa, estraiamo i dati (posizioni e forze) da una traiettoria GROMACS e li salviamo nel formato NumPy atteso da TorchMD-Net. Le forze vengono divise per 1000 per stabilizzare il training.

```bash
# Esempio di come preparare il dataset (script Python)
python prepare_dataset.py
```
*Questo genererà i file: `dataset_pos.npy`, `dataset_z.npy`, `dataset_dy.npy`, e per retrocompatibilità `dataset.npz`.*

## 2. Addestramento del Modello (Training)
A causa del bug di PyTorch (MPS) con il calcolo dell'autograd sulle norme vettoriali che causa `NaN`, non usiamo `equivariant-transformer` (PaiNN), ma usiamo il modello `graph-network` (SchNet). La configurazione si trova in `train.yaml`.

**Comando di Training:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m torchmdnet.scripts.train --conf train.yaml
```
*I modelli addestrati (checkpoints) verranno salvati all'interno della cartella `logs/` (ad esempio `logs/epoch=9-val_loss=0.0491.ckpt`).*

## 3. Simulazione di Dinamica Molecolare Neurale
Il nostro script `simulate.py` non usa gli integratori standard di TorchMD che possono risultare instabili/incompatibili a causa di cambi di API, ma implementa un loop di Eulero robusto che combina le forze fisiche classiche (Prior WCA) e quelle predette dalla rete neurale.

**Comando di Simulazione:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python simulate.py --model logs/epoch=9-val_loss=0.0491.ckpt --dataset dataset.npz --steps 1000
```
*(Sostituisci il nome del `.ckpt` con l'ultimo modello generato dal training e `--steps` con il numero di step desiderati).*

*Al termine, lo script stamperà le energie ogni 10 step e salverà le posizioni nel file `trajectory.npy`.*

## 4. Calcolo e Confronto della RDF
Per calcolare la Funzione di Distribuzione Radiale (RDF) della traiettoria generata dal modello neurale e confrontarla automaticamente con la traiettoria di GROMACS (ground truth), usa lo script che abbiamo creato:

**Comando RDF:**
```bash
python compute_rdf.py
```
*Lo script caricherà `trajectory.npy` e `dataset.npz`, calcolerà le RDF, e genererà un'immagine PNG chiamata `plot_rdf.png` pronta per essere visualizzata.*

---
**Note Utili per il Futuro (CUDA):**
Se sposterai questo codice su una macchina Linux con GPU NVIDIA, ti basterà:
1. Rimuovere i prefissi `PYTORCH_ENABLE_MPS_FALLBACK=1`
2. Aprire `train.yaml` e rimettere `model: equivariant-transformer` (PaiNN) al posto di `graph-network`. La precisione e la RDF diventeranno perfette!
