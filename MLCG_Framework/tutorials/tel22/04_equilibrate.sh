#!/bin/bash
set -e

echo "======================================================"
echo " 04. ESPRESSO MD EQUILIBRATION "
echo "======================================================"

if [ ! -f "tel22_model.pt" ]; then
    echo "Errore: Modello tel22_model.pt non trovato! Hai eseguito 03_train_model.sh?"
    exit 1
fi

echo "Avvio l'equilibrazione con Langevin Dynamics e Force Capping..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
../../../espresso/build/pypresso ../../simulation/equilibrate.py \
    --model tel22_model.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --out_checkpoint equilibrated.npz \
    --kT 2.49

echo "[SUCCESS] Sistema equilibrato e salvato in equilibrated.npz"
