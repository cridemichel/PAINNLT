#!/bin/bash
set -euo pipefail

# Final tiny-set confirmation for the architecture selected by 03c/03d:
#   hidden=128, layers=3, num_rbf=64, cutoff=1.6 nm, torque_weight=0.5
# The same 16 deterministic frames are used for train and validation.
# This is a representability/optimization test, NOT a generalization estimate.

case_tag="confirm_D_w050_200"
config="${case_tag}_run.json"
model="${case_tag}.pt"
log="${case_tag}.log"
csv="${case_tag}_run_training_log.csv"
summary="${case_tag}_summary.csv"

if [ ! -f "tel22_dataset.bin" ]; then
    echo "Errore: tel22_dataset.bin non trovato!"
    exit 1
fi

TRAINER="../../training/build/train_painn"
if [ ! -x "$TRAINER" ]; then
    echo "Errore: trainer non trovato/eseguibile: $TRAINER"
    echo "Ricompila prima con: cd ../../training/build && cmake .. && make -j"
    exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

cat > "$config" <<'JSON'
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
    "early_stopping_patience": 200,
    "reduce_lr_patience": 10,
    "diagnostic_overfit_frames": 16
}
JSON

rm -f "$model" "$model.manifest.json" cg_training_log.csv "$log" "$csv" "$summary"

echo "======================================================"
echo " 03e. FINAL TINY-SET CONFIRMATION: D_both + wT=0.5"
echo "======================================================"
echo " hidden=128 | layers=3 | rbf=64 | cutoff=1.6 nm"
echo " torque_weight=0.5 | epochs=200 | batch=4"
echo " 16 frame identici in train e validation"
echo " Early stopping effectively disabled for this 200-epoch diagnostic."
echo

"$TRAINER" tel22_dataset.bin "$model" "$config" | tee "$log"
mv cg_training_log.csv "$csv"

python3 ./summarize_training_grid.py \
    --prefix "$case_tag" \
    --output "$summary" \
    --epoch-metric balanced_ft

echo
if [ -f "torque_D_both_w050_training_log.csv" ]; then
    python3 - "$csv" "torque_D_both_w050_training_log.csv" <<'PY'
import csv
import sys


def best(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = min(rows, key=lambda r: float(r["Val_Loss_F_Norm"]) + float(r["Val_Loss_T_Norm"]))
    f = float(row["Val_Loss_F_Norm"])
    t = float(row["Val_Loss_T_Norm"])
    return int(row["Epoch"]), f, t, f + t, float(row["Val_MAE_F"]), float(row["Val_MAE_T"])

new = best(sys.argv[1])
old = best(sys.argv[2])
print("Confronto balanced F+T con il precedente scan wT=0.5:")
print(f"  precedente: epoch={old[0]:3d}  F={old[1]:.6f}  T={old[2]:.6f}  F+T={old[3]:.6f}  MAE_F={old[4]:.3f}  MAE_T={old[5]:.3f}")
print(f"  200 epoch : epoch={new[0]:3d}  F={new[1]:.6f}  T={new[2]:.6f}  F+T={new[3]:.6f}  MAE_F={new[4]:.3f}  MAE_T={new[5]:.3f}")
print(f"  delta score (new-old): {new[3] - old[3]:+.6f}")
PY
else
    echo "Nota: torque_D_both_w050_training_log.csv non trovato; salto il confronto automatico con il run da 100 epoche."
fi

echo
echo "Test completato. Risultati principali:"
echo "  $summary"
echo "  $csv"
echo "  $log"
echo "Non usare questo tiny-set score come misura di generalizzazione sul dataset completo."
