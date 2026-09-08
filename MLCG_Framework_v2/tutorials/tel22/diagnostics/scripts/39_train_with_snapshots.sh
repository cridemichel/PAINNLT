#!/usr/bin/env bash
# Training C1 — full Morse, 64/2, LR costante, con SNAPSHOT PERIODICI.
#
# PERCHE'
#   Il trainer salva il modello solo quando la val force loss migliora, e
#   quella metrica ha SNR ~1:300. Nel run antiparallel_long_40ep la val
#   migliorava fino all'epoca 6: i pesi delle epoche 7-40 sono stati
#   scartati. Il modello finora testato e' quindi quello dell'epoca 6.
#   Con checkpoint_every_epochs=5 si ottengono 8 snapshot valutabili con
#   le metriche strutturali 37/38.
#
# USO
#   bash 39_train_with_snapshots.sh          # ~86 min
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${ROOT}/training/build/train_painn}"
CONFIG="${C1_CONFIG:-${TEL22_DIR}/diagnostics/configs/tel22_training_config_C1_snapshots.json}"
RUN_DIR="${C1_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/C1_snapshots_64x2}"
SRC="${C1_SRC:-${TEL22_DIR}/diagnostics/smoke/antiparallel_long_40ep}"

[[ -x "${TRAINER}" ]] || { echo "[ERROR] trainer assente: ${TRAINER}" >&2; exit 2; }
"${TRAINER}" --help 2>&1 | head -0 || true
for f in tel22_dataset.bin cg_priors.json rigid_bodies_info.json; do
    [[ -s "${SRC}/${f}" ]] || { echo "[ERROR] assente: ${SRC}/${f}" >&2; exit 2; }
done
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "[ERROR] directory non vuota: ${RUN_DIR}" >&2
    echo "        mv '${RUN_DIR}' '${RUN_DIR}.STALE'  oppure  C1_RUN_DIR=... " >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"
cp "${SRC}/tel22_dataset.bin" "${SRC}/cg_priors.json" "${SRC}/rigid_bodies_info.json" "${RUN_DIR}/"
CONFIG_NAME="$(basename "${CONFIG}")"
cp "${CONFIG}" "${RUN_DIR}/${CONFIG_NAME}"

"${PYTHON_BIN}" - "${RUN_DIR}/cg_priors.json" << 'PYEOF'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
bt = {}
for b in p.get("bonds", []):
    t = str(b.get("type", "?")).lower(); bt[t] = bt.get(t, 0) + 1
print(f"[GUARD] harmonic={bt.get('harmonic',0)} morse={bt.get('morse',0)}")
if bt.get("morse", 0) == 0:
    sys.exit("[FAIL] prior senza Morse: non e' il setup C1.")
print("[GUARD] OK - full Morse.")
PYEOF

cd "${RUN_DIR}"
echo "[INFO] C1: 40 epoche, snapshot ogni 5 -> ~86 min"
PYTHONUNBUFFERED=1 "${TRAINER}" tel22_dataset.bin C1_model.pt "${CONFIG_NAME}" 2>&1 | tee training_stdout.log
echo
echo "[OK] snapshot prodotti:"
ls -1 C1_model.ep*.pt 2>/dev/null | sed 's/^/  /' || echo "  NESSUNO - il trainer e' stato ricompilato?"
