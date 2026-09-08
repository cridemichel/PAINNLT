#!/usr/bin/env bash
# Test A1 — Overfit diagnostico su 200 frame, senza regolarizzazione.
#
# Obiettivo: verificare che il modello PaiNN sia in grado di memorizzare
# un sottoinsieme piccolo di frame. Se il train MAE non scende sotto
# ~50 kJ/mol/nm entro 300 epoche c'è un problema nel modello o nei dati.
# Se scende, il modello funziona: il problema sul dataset completo è solo
# il segnale debole (10k frame al noise floor).
#
# Uso:
#   PYTHON_BIN="$(command -v python3)" \
#   TRAINER="$PWD/training/build/train_painn" \
#   bash tutorials/ala2_cgnet/diagnostics/scripts/07_test_overfit_A1.sh
#
# L'output va in diagnostics/smoke/overfit_A1_200frames/ (sovrascrivibile
# con ALA2_RUN_DIR).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALA2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${ALA2_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
DATA_DIR="${ALA2_DATA_DIR:-${ALA2_DIR}/data}"
CONFIG_SOURCE="${ALA2_DIR}/diagnostics/configs/ala2_training_config_overfit_A1.json"
RUN_DIR="${ALA2_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/overfit_A1_200frames}"

# Controlli preliminari
if [[ ! -x "${TRAINER}" ]]; then
    printf '[ERROR] Trainer non trovato o non eseguibile: %s\n' "${TRAINER}" >&2
    printf '        Compila prima con: cmake --build training/build\n' >&2
    exit 2
fi
if [[ ! -f "${CONFIG_SOURCE}" ]]; then
    printf '[ERROR] Config non trovato: %s\n' "${CONFIG_SOURCE}" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import numpy' >/dev/null 2>&1; then
    printf '[ERROR] %s non trova numpy. Attiva il virtualenv.\n' "${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] La directory non è vuota: %s\n' "${RUN_DIR}" >&2
    printf '        Scegli un altro ALA2_RUN_DIR o cancella la directory.\n' >&2
    exit 2
fi

mkdir -p "${DATA_DIR}" "${RUN_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

printf '[INFO] Test A1 — Overfit su 200 frame, nessuna regolarizzazione\n'
printf '[INFO] Run dir: %s\n' "${RUN_DIR}"

# Download dati se non presenti
"${PYTHON_BIN}" "${SCRIPT_DIR}/download_cgnet_ala2.py" \
    --output-dir "${DATA_DIR}"

# Copia config nella run dir (il trainer lo legge da lì)
cp "${CONFIG_SOURCE}" "${RUN_DIR}/ala2_training_config_50ep.json"

# Costruisce il dataset (harmonic prior, tail split standard)
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_ala2_dataset.py" \
    --coordinates "${DATA_DIR}/ala2_coordinates.npy" \
    --forces "${DATA_DIR}/ala2_forces.npy" \
    --output "${RUN_DIR}/ala2_dataset.bin" \
    --priors-output "${RUN_DIR}/ala2_priors.json" \
    --rb-info-output "${RUN_DIR}/ala2_rigid_bodies_info.json" \
    --reference-output "${RUN_DIR}/ala2_reference.npz" \
    --report "${RUN_DIR}/ala2_conversion_report.json" \
    --prior-mode harmonic \
    --validation-tail-frames 2000 \
    2>&1 | tee "${RUN_DIR}/conversion_stdout.log"

cd "${RUN_DIR}"

printf '\n[INFO] Avvio training (300 epoche, 200 frame, lr=0.001, no regolarizzazione)...\n'
"${TRAINER}" \
    ala2_dataset.bin \
    ala2_model.pt \
    ala2_training_config_50ep.json \
    2>&1 | tee training_stdout.log

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/training/create_model_manifest.py" \
    --model ala2_model.pt \
    --config ala2_training_config_50ep.json \
    --dataset ala2_dataset.bin

# Validazione formale (non produce un pass scientifico, solo verifica artefatti)
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_ala2_benchmark.py" \
    --run-dir "${RUN_DIR}" \
    --report "${RUN_DIR}/ala2_benchmark_report.json" || true

# Stampa un riassunto leggibile direttamente dal CSV
printf '\n[RISULTATI TEST A1] Ultime 10 epoche (train vs val MAE kJ/mol/nm):\n'
"${PYTHON_BIN}" - << 'PY'
import csv, sys
path = "cg_training_log.csv"
rows = list(csv.DictReader(open(path)))
print(f"  {'Epoch':>5}  {'Train_MAE_F':>14}  {'Val_MAE_F':>12}  {'Train_Loss_F_Norm':>20}  {'Val_Loss_F_Norm':>18}")
for r in rows[-10:]:
    print(f"  {r['Epoch']:>5}  {float(r['Train_MAE_F']):>14.2f}  {float(r['Val_MAE_F']):>12.2f}  {float(r['Train_Loss_F_Norm']):>20.6f}  {float(r['Val_Loss_F_Norm']):>18.6f}")
best_train = min(rows, key=lambda r: float(r['Train_MAE_F']))
print(f"\n  Miglior Train MAE: {float(best_train['Train_MAE_F']):.2f} kJ/mol/nm (epoca {best_train['Epoch']})")
zero = float(rows[0].get('Val_Zero_F_Norm', 'nan'))
best_val = min(float(r['Val_Loss_F_Norm']) for r in rows)
skill = (1.0 - best_val / zero) * 100 if zero > 0 else float('nan')
print(f"  MSE skill (val):   {skill:.2f}%")
print()
if float(best_train['Train_MAE_F']) < 50.0:
    print("  [PASS] Il modello sa memorizzare: train MAE < 50 kJ/mol/nm.")
    print("         Il problema sul dataset completo è il segnale debole, non il modello.")
else:
    print("  [WARN] Train MAE >= 50 kJ/mol/nm dopo 300 epoche.")
    print("         Possibile problema nel modello, nel formato dati o nel training loop.")
PY

printf '\n[DONE] Test A1 completato. Log completo: %s/training_stdout.log\n' "${RUN_DIR}"
