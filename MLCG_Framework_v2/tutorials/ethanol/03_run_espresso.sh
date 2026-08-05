#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYPRESSO="${PYPRESSO:-../../espresso/build/pypresso}"
DEVICE="${DEVICE:-auto}"

for file in \
  my_ethanol_dataset.bin \
  my_ethanol_model.pt \
  my_ethanol_model.pt.manifest.json \
  fast_training_config.json \
  cg_priors.json \
  rigid_bodies_info.json; do
  if [[ ! -f "$file" ]]; then
    echo "[ERROR] Missing $file. Run 01_build_dataset.sh and 02_train_model.sh first." >&2
    exit 1
  fi
done

if [[ ! -x "$PYPRESSO" ]]; then
  echo "[ERROR] pypresso not found or not executable: $PYPRESSO" >&2
  echo "Set PYPRESSO=/absolute/path/to/pypresso and retry." >&2
  exit 1
fi

echo "======================================================"
echo " 03. ESPRESSO EQUILIBRATION AND NVE SCALING (ETHANOL) "
echo "======================================================"

"$PYPRESSO" ../../simulation/equilibrate.py \
  --model my_ethanol_model.pt \
  --config fast_training_config.json \
  --priors cg_priors.json \
  --rb_info rigid_bodies_info.json \
  --dataset my_ethanol_dataset.bin \
  --out_checkpoint equilibrated_ethanol.npz \
  --device "$DEVICE" \
  --dt 0.0005 \
  --steps_sd 2000 \
  --steps_md 2000 \
  --steps_ml_capped 2000 \
  --steps_ml_uncapped 2000

uv run python run_energy_scaling.py \
  --pypresso "$PYPRESSO" \
  --device "$DEVICE"
