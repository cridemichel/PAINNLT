# Tutorial: Water Coarse-Graining Pipeline

Benvenuto in questo tutorial! Qui imparerai come eseguire l'intera pipeline MLCG (Machine Learning Coarse-Graining) per l'Acqua, mappata a singola particella (Single-Bead COM mapping).

In questa cartella troverai lo script automatizzato `run_full_pipeline.sh` che esegue i passaggi qui sotto.

---

## Passo 1: Preprocessing 
Per l'acqua eseguiamo un mapping a singola bead, calcolando il Centro di Massa (COM) della molecola. Usa lo script `build_cg_dataset.py`:

```bash
python3 ../../preprocessing/build_cg_dataset.py \
    --traj traiettoria.trr \
    --topol conf.gro \
    --config water_topology.json \
    --output my_water_dataset.bin
```
*Cosa succede:* Lo script raggruppa O, H1, H2 in una singola particella nel Centro di Massa e aggrega le forze totali sulla molecola. Non calcolando legami interni, si limiterà ad estrarre le coordinate CG.

---

## Passo 2: Addestramento del Modello (C++)
Addestriamo la Rete Neurale a prevedere le interazioni inter-molecolari tra i centri di massa dell'acqua.

```bash
# Usa epoch=3 in cg_model_config.json per testare la pipeline in 5 secondi!
../../training/build/train_painn \
    my_water_dataset.bin \
    my_water_model.pt \
    best_cg_model_config.json
```

---

## Passo 3: Dinamica Molecolare in ESPResSo
Avvia la dinamica in ESPResSo. Poiché è a singola particella, non ci sono legami armonici o Virtual Sites complessi, ma l'interazione pura della rete neurale (più un eventuale WCA).

```bash
/path/to/espresso/build/pypresso verify_water.py
```
*Cosa succede:* Verifica la conservazione dell'energia nello stack ESPResSo + PyTorch, dimostrando lo scaling dell'errore di Newton $O(dt^2)$.
