#!/bin/bash
set -e

echo "================================================="
echo " MLCG Framework - Water Tutorial Pipeline "
echo "================================================="

echo -e "\n[1/3] Preprocessing (Single-Bead COM Mapping)..."
python3 ../../preprocessing/build_cg_dataset.py \
    --traj traiettoria.trr \
    --topol conf.gro \
    --config water_topology.json \
    --output my_water_dataset.bin

echo -e "\n[2/3] Training the Neural Network (fast test: 3 epochs)..."
# Create a temporary config for fast training
cat << 'JSON' > fast_training_config.json
{
    "num_species": 1,
    "hidden_channels": 32,
    "n_layers": 2,
    "num_rbf": 30,
    "cutoff": 0.5,
    "learning_rate": 0.001,
    "epochs": 3,
    "batch_size": 16
}
JSON

# Run training
../../training/build/train_painn \
    my_water_dataset.bin \
    my_water_model.pt \
    fast_training_config.json

echo -e "\n[3/3] Running ESPResSo Energy Conservation Validation..."
echo "You must provide the path to your pypresso executable."
echo "For example: /path/to/espresso/build/pypresso verify_water.py"
echo "We cannot run it automatically without knowing your ESPResSo path!"
echo "Tutorial successfully reached the final step."
