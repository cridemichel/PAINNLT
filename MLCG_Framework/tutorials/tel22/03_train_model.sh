#!/bin/bash
set -e

echo "======================================================"
echo " 03. TRAINING NEURAL NETWORK "
echo "======================================================"

if [ ! -f "tel22_dataset.bin" ]; then
    echo "Errore: tel22_dataset.bin non trovato!"
    exit 1
fi

echo "Avvio l'addestramento C++ (5 epoche)..."
../../training/build/train_painn \
    tel22_dataset.bin \
    tel22_model.pt \
    tel22_training_config.json

echo "Addestramento completato e modello salvato in tel22_model.pt!"
