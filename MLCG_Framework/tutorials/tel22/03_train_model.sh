#!/bin/bash
set -e

echo "======================================================"
echo " 03. TRAINING NEURAL NETWORK "
echo "======================================================"

if [ ! -f "tel22_dataset.bin" ]; then
    echo "Errore: tel22_dataset.bin non trovato!"
    exit 1
fi

# Generiamo un file di configurazione per il training (veloce per il tutorial)
cat << 'JSON' > tel22_training_config.json
{
    "num_species": 8,
    "hidden_channels": 64,
    "n_layers": 3,
    "num_rbf": 50,
    "cutoff": 1.0,
    "learning_rate": 0.0005,
    "epochs": 1000,
    "batch_size": 16
}
JSON

echo "Avvio l'addestramento C++ (5 epoche)..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
../../training/build/train_painn \
    tel22_dataset.bin \
    tel22_model.pt \
    tel22_training_config.json

echo "Addestramento completato e modello salvato in tel22_model.pt!"
