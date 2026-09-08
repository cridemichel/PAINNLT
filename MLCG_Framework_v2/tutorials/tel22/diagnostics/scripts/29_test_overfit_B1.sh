#!/usr/bin/env bash
# Test B1 — TEL22 capacity / overfit check (analogous to Ala2 test A1)
#
# PURPOSE
#   Verify that PaiNN with message passing CAN learn TEL22 residual forces
#   when given sufficient capacity and no regularization, trained on a
#   restricted subset of 200 frames from the 1001-frame dataset.
#
#   Expected outcomes:
#     Train MAE << Val MAE  → model has capacity; it needs more training data
#     Train MAE ≈ Val MAE   → irreducible CG noise floor dominates
#
# USAGE
#   Minimal (uses defaults):
#     bash 29_test_overfit_B1.sh
#
#   Custom run directory:
#     OVERFIT_B1_RUN_DIR=/path/to/dir bash 29_test_overfit_B1.sh
#
#   Reuse an already-built dataset (e.g. from variant_a_long_1001f_15ep):
#     OVERFIT_B1_REUSE_DATASET_DIR=/path/to/run bash 29_test_overfit_B1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"

CONFIG_SOURCE="${OVERFIT_B1_CONFIG:-${TEL22_DIR}/diagnostics/configs/tel22_training_config_overfit_B1.json}"
RUN_DIR="${OVERFIT_B1_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/overfit_B1_200frames}"

# Default: reuse the 1001-frame dataset already built by test 08
DEFAULT_REUSE="${TEL22_DIR}/diagnostics/smoke/variant_a_long_1001f_15ep"
REUSE_DIR="${OVERFIT_B1_REUSE_DATASET_DIR:-${DEFAULT_REUSE}}"

# ── Preflight ────────────────────────────────────────────────────────────────
if [[ ! -x "${TRAINER}" ]]; then
    printf '[ERROR] Trainer not found or not executable: %s\n' "${TRAINER}" >&2
    exit 2
fi

if [[ ! -f "${CONFIG_SOURCE}" ]]; then
    printf '[ERROR] Config not found: %s\n' "${CONFIG_SOURCE}" >&2
    exit 2
fi

if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Run directory is not empty: %s\n' "${RUN_DIR}" >&2
    printf '        Set OVERFIT_B1_RUN_DIR to a fresh path; evidence is never overwritten.\n' >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"

CONFIG_NAME="$(basename "${CONFIG_SOURCE}")"

# ── Dataset ──────────────────────────────────────────────────────────────────
if [[ -d "${REUSE_DIR}" ]]; then
    for artifact in tel22_dataset.bin cg_priors.json rigid_bodies_info.json; do
        if [[ ! -s "${REUSE_DIR}/${artifact}" ]]; then
            printf '[ERROR] Missing artifact in reuse directory: %s/%s\n' "${REUSE_DIR}" "${artifact}" >&2
            exit 2
        fi
    done
    cp "${REUSE_DIR}/tel22_dataset.bin"       "${RUN_DIR}/tel22_dataset.bin"
    cp "${REUSE_DIR}/cg_priors.json"          "${RUN_DIR}/cg_priors.json"
    cp "${REUSE_DIR}/rigid_bodies_info.json"  "${RUN_DIR}/rigid_bodies_info.json"
    printf '[INFO] Reused dataset and priors from: %s\n' "${REUSE_DIR}"
else
    printf '[ERROR] Reuse dataset directory not found: %s\n' "${REUSE_DIR}" >&2
    printf '        Run test 08 first (08_test_variant_a_pipeline_15ep.sh) or set OVERFIT_B1_REUSE_DATASET_DIR.\n' >&2
    exit 2
fi

cp "${CONFIG_SOURCE}" "${RUN_DIR}/${CONFIG_NAME}"

# patch cg_priors.json topology if needed (variant-a topology prep)
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_variant_a_topology.py" \
    --input "${RUN_DIR}/cg_priors.json"

# ── Training ─────────────────────────────────────────────────────────────────
printf '[INFO] Starting Test B1: overfit 200 frames, hidden=128, n_layers=4, 200 epochs\n'
cd "${RUN_DIR}"
"${TRAINER}" \
    tel22_dataset.bin \
    overfit_B1.pt \
    "${CONFIG_NAME}" \
    2>&1 | tee training_stdout.log

# ── Summary ──────────────────────────────────────────────────────────────────
"${PYTHON_BIN}" - << 'PYEOF'
import csv, sys, os

log_path = "cg_training_log.csv"
if not os.path.exists(log_path):
    print("[WARN] cg_training_log.csv not found — training may have failed.")
    sys.exit(0)

with open(log_path) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("[WARN] Training log is empty.")
    sys.exit(0)

# Best val (min Val_Loss_F_Norm)
best_val_row  = min(rows, key=lambda r: float(r["Val_Loss_F_Norm"]))
last_row      = rows[-1]

def skill(row, key="Val_Loss_F_Norm", zero_key="Val_Zero_F_Norm"):
    vl = float(row[key])
    vz = float(row[zero_key])
    return (1.0 - vl / vz) * 100.0 if vz > 0 else float("nan")

best_ep       = int(best_val_row["Epoch"])
best_train_mae = float(best_val_row["Train_MAE_F"])
best_val_mae   = float(best_val_row["Val_MAE_F"])
best_skill_f   = skill(best_val_row)
best_skill_t   = skill(best_val_row, "Val_Loss_T_Norm", "Val_Zero_T_Norm")
last_ep        = int(last_row["Epoch"])
last_train_mae = float(last_row["Train_MAE_F"])
last_val_mae   = float(last_row["Val_MAE_F"])
gap            = last_val_mae - last_train_mae

print()
print("=" * 60)
print("  Test B1 — TEL22 Overfit Check (200 frames, hidden=128, n_layers=4)")
print("=" * 60)
print(f"  Epochs completed:       {last_ep}")
print(f"  Best epoch (val F):     {best_ep}")
print(f"  Best train MAE F:       {best_train_mae:.1f} kJ/mol/nm")
print(f"  Best val   MAE F:       {best_val_mae:.1f} kJ/mol/nm")
print(f"  Val-train gap (last ep):{gap:+.1f} kJ/mol/nm")
print(f"  MSE skill forces:       {best_skill_f:.2f}%")
print(f"  MSE skill torques:      {best_skill_t:.2f}%")
print()

# Verdict
if best_skill_f > 50:
    verdict = "PASS — Model has clear capacity; gap indicates data-limited regime."
elif best_skill_f > 10 and gap > 30:
    verdict = "PASS — Model overfits; training is data-limited, not capacity-limited."
elif best_skill_f > 0 and gap > 10:
    verdict = "WEAK — Some learning observed; consider more epochs or larger model."
elif best_skill_f <= 0:
    verdict = "FAIL — Val skill negative; model unable to generalise even from 200 overfit frames."
else:
    verdict = "WARN — Skill near zero; noise floor may dominate even at this scale."

print(f"  Verdict: {verdict}")
print("=" * 60)
print()
print("  Interpretation guide:")
print("    Train≈Val, high skill → noise floor dominates (like Ala2 test A1)")
print("    Train<<Val, high skill → data-limited; add more training frames")
print("    Both near zero         → model capacity or cutoff problem")
print()
PYEOF

printf '[INFO] Test B1 complete. Results in: %s\n' "${RUN_DIR}"
