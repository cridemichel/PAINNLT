#!/bin/bash
set -euo pipefail

echo "======================================================"
echo " 03. TRAINING NEURAL NETWORK "
echo "======================================================"

if [ ! -f "tel22_dataset.bin" ]; then
    echo "Errore: tel22_dataset.bin non trovato!"
    exit 1
fi

# Do not overwrite a tuned/ablated configuration on every launch.
# Create the historical TEL22 baseline only when no config exists yet.
if [ ! -f "tel22_training_config.json" ]; then
    cat << 'JSON' > tel22_training_config.json
{
    "num_species": 8,
    "hidden_channels": 128,
    "n_layers": 3,
    "num_rbf": 64,
    "cutoff": 1.6,

    "toxvaerd_alpha": 0.1,
    "learning_rate": 0.001,
    "epochs": 200,
    "batch_size": 4,
    "torque_weight": 0.5,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 30,
    "reduce_lr_patience": 8,
    "weight_decay": 0.0,
    "lipschitz_lambda": 0.0,
    "diagnostic_overfit_frames": 0
}
JSON
    echo "[INFO] Creato tel22_training_config.json con il profilo TEL22 di produzione validato dalle ablation."
else
    echo "[INFO] Uso tel22_training_config.json esistente senza riscriverlo."
fi

python3 - <<'PY'
import json
with open("tel22_training_config.json") as handle:
    c = json.load(handle)
print(
    "[INFO] Config: "
    f"hidden={c.get('hidden_channels')} | layers={c.get('n_layers')} | "
    f"rbf={c.get('num_rbf')} | cutoff={c.get('cutoff')} nm | "
    f"epochs={c.get('epochs')} | batch={c.get('batch_size')} | "
    f"torque_weight={c.get('torque_weight')} | "
    f"reduce_lr_patience={c.get('reduce_lr_patience')} | "
    f"early_stopping_patience={c.get('early_stopping_patience')}"
)
if int(c.get("diagnostic_overfit_frames", 0)) > 0:
    raise SystemExit(
        "[ERROR] tel22_training_config.json contiene diagnostic_overfit_frames > 0. "
        "Non usare una config tiny-set per il training di produzione."
    )
PY

echo "Avvio l'addestramento C++ (force+torque normalizzati)..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
../../training/build/train_painn \
    tel22_dataset.bin \
    tel22_model.pt \
    tel22_training_config.json

echo "Addestramento completato e modello salvato in tel22_model.pt!"
