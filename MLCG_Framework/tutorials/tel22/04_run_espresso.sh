#!/bin/bash
set -e

echo "======================================================"
echo " 04. ESPRESSO MD SIMULATION "
echo "======================================================"

if [ ! -f "tel22_model.pt" ]; then
    echo "Errore: Modello tel22_model.pt non trovato! Hai eseguito 03_train_model.sh?"
    exit 1
fi

echo "Avvio la simulazione ESPResSo usando il pypresso di sistema."
echo "Per personalizzare, apri e modifica lo script run_cg_md.py."

echo "Esecuzione in corso..."
../../../espresso/build/pypresso ../../simulation/run_cg_md.py \
    --model tel22_model.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --steps 50000 \
    --dt 0.001 \
    --kT 2.49
