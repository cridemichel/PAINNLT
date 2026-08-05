#!/bin/bash
set -e

echo "======================================================"
echo " 02. TRAINING NEURAL NETWORK (ETHANOL) "
echo "======================================================"

if [ ! -f "my_ethanol_dataset.bin" ]; then
    echo "Errore: my_ethanol_dataset.bin non trovato!"
    exit 1
fi

cat << 'JSON' > fast_training_config.json
{
    "num_species": 3,
    "hidden_channels": 32,
    "n_layers": 2,
    "num_rbf": 30,
    "cutoff": 0.5,
    "learning_rate": 0.001,
    "epochs": 3,
    "batch_size": 16,
    "toxvaerd_alpha": 0.1,
    "torque_weight": 1.0,
    "weight_decay": 0.0,
    "lipschitz_lambda": 0.0
}
JSON

echo "Avvio l'addestramento C++ (3 epoche per test)..."
../../training/build/train_painn \
    my_ethanol_dataset.bin \
    my_ethanol_model.pt \
    fast_training_config.json

echo "Addestramento completato e modello salvato in my_ethanol_model.pt!"
