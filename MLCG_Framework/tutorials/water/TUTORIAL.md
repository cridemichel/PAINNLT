# Tutorial: Water Coarse-Graining Pipeline

Benvenuto in questo tutorial! Qui imparerai come eseguire l'intera pipeline MLCG (Machine Learning Coarse-Graining) per l'Acqua, mappata a singola particella (Single-Bead COM mapping).

In questa cartella troverai lo script automatizzato `run_full_pipeline.sh` che esegue i passaggi qui sotto.

---

## Passo 1: Preprocessing (`01_build_dataset.sh`)
Per l'acqua eseguiamo un mapping a singola bead, calcolando il Centro di Massa (COM) della molecola. Usa lo script in dotazione:

```bash
./01_build_dataset.sh
```
*Cosa succede:* Lo script raggruppa O, H1, H2 in una singola particella nel Centro di Massa e aggrega le forze totali sulla molecola. Non calcolando legami interni, si limiterà ad estrarre le coordinate CG.

---

## Passo 2: Addestramento del Modello (`02_train_model.sh`)
Addestriamo la Rete Neurale a prevedere le interazioni inter-molecolari tra i centri di massa dell'acqua.

```bash
./02_train_model.sh
```

---

## Passo 3: Dinamica Molecolare in ESPResSo (`03_run_espresso.sh`)
Avvia la dinamica in ESPResSo. Poiché è a singola particella, non ci sono legami armonici o Virtual Sites complessi, ma l'interazione pura della rete neurale (più un eventuale WCA).

```bash
./03_run_espresso.sh
```
*Cosa succede:* Verifica la conservazione dell'energia nello stack ESPResSo + PyTorch, dimostrando lo scaling dell'errore di Newton $O(dt^2)$.
