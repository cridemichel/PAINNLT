#!/usr/bin/env bash
# Test B2 — TEL22 full Morse, scheduler LR disattivato
#
# SCOPERTA CHE MOTIVA QUESTO TEST
#   Test B1 (128/4, tiny-set 50 frame) mostra un PLATEAU nelle prime ~13
#   epoche (val_F 0.99 -> 0.90) seguito da una rottura: ep15 0.83,
#   ep25 0.37, ep49 0.147 (skill 85%).
#   Nel baseline antiparallel_long_40ep il LR viene tagliato proprio a
#   partire dall'epoca 13: 0.001 -> 0.0005 (ep13) -> 0.00025 (ep19)
#   -> 0.000125 (ep25) -> ... -> 3.125e-05.
#   Lo scheduler ha interpretato il plateau di warm-up come convergenza e
#   ha congelato il modello nel plateau. Da qui il 12% di fit sul train.
#
# IPOTESI
#   Il fallimento dei run di produzione e' causato dallo scheduler LR
#   (reduce_lr_patience 4-6) piu' l'early stopping, non dalla capacita'.
#
# DISEGNO
#   Identico al baseline antiparallel_long_40ep (full Morse, hidden=64,
#   n_layers=2, batch 4, split reale 801/200, split_seed=42), con UNA
#   sola differenza sostanziale: LR costante a 0.001
#   (reduce_lr_patience=9999, epoch_lr_decay_factor=1.0), early stopping
#   disattivato, clipping off. 45 epoche, ~2.1 min/epoca -> ~96 min.
#
# REGOLA DI DECISIONE (confronto automatico epoca per epoca col baseline)
#   val_F scende sotto ~0.80 entro l'ep 45 -> era lo scheduler: risolto,
#       l'architettura 64/2 andava bene. Poi si scala la capacita'.
#   val_F resta ~0.98 e il train crolla -> data-limited, servono frame.
#   entrambi fermi ~0.98 -> allora conta la capacita': passare a 128/4.
#
# USO
#   bash 30_test_B2_morse_capacity.sh
#   B2_RUN_DIR=/path bash 30_test_B2_morse_capacity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
CONFIG_SOURCE="${B2_CONFIG:-${TEL22_DIR}/diagnostics/configs/tel22_training_config_B2_morse_noscheduler.json}"
RUN_DIR="${B2_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/B2_morse_noscheduler_64x2}"
REUSE_DIR="${B2_REUSE_DATASET_DIR:-${TEL22_DIR}/diagnostics/smoke/antiparallel_long_40ep}"
BASELINE_LOG="${B2_BASELINE_LOG:-${TEL22_DIR}/diagnostics/smoke/antiparallel_long_40ep/cg_training_log.csv}"

[[ -x "${TRAINER}" ]]      || { printf '[ERROR] Trainer non eseguibile: %s\n' "${TRAINER}" >&2; exit 2; }
[[ -f "${CONFIG_SOURCE}" ]]|| { printf '[ERROR] Config assente: %s\n' "${CONFIG_SOURCE}" >&2; exit 2; }

for artifact in tel22_dataset.bin cg_priors.json rigid_bodies_info.json; do
    [[ -s "${REUSE_DIR}/${artifact}" ]] || {
        printf '[ERROR] Artifact assente: %s/%s\n' "${REUSE_DIR}" "${artifact}" >&2; exit 2; }
done

if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Directory non vuota: %s\n' "${RUN_DIR}" >&2
    printf '        Usa un B2_RUN_DIR fresco; le evidenze non si sovrascrivono.\n' >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"

cp "${REUSE_DIR}/tel22_dataset.bin"      "${RUN_DIR}/tel22_dataset.bin"
cp "${REUSE_DIR}/cg_priors.json"         "${RUN_DIR}/cg_priors.json"
cp "${REUSE_DIR}/rigid_bodies_info.json" "${RUN_DIR}/rigid_bodies_info.json"
CONFIG_NAME="$(basename "${CONFIG_SOURCE}")"
cp "${CONFIG_SOURCE}" "${RUN_DIR}/${CONFIG_NAME}"

# GUARD: i Morse DEVONO essere presenti. Nessuna chiamata a
# prepare_variant_a_topology.py qui: quello li rimuoverebbe.
"${PYTHON_BIN}" - "${RUN_DIR}/cg_priors.json" << 'PYEOF'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
bt = {}
for b in p.get("bonds", []):
    t = str(b.get("type", "?")).lower()
    bt[t] = bt.get(t, 0) + 1
n_morse = bt.get("morse", 0)
n_harm  = bt.get("harmonic", 0)
print(f"[GUARD] cg_priors: harmonic={n_harm} morse={n_morse} "
      f"morse_type_pairs={len(p.get('morse_type_pairs', []))} "
      f"wca_pairs={len(p.get('wca_pairs', {}))} angles={len(p.get('angles', []))}")
if n_morse == 0:
    sys.exit("[FAIL] Nessun bond Morse nei prior: questo NON e' il setup B2.")
if n_harm == 0:
    sys.exit("[FAIL] Nessun bond harmonic nei prior.")
print("[GUARD] OK - prior full-Morse confermati.")
PYEOF

printf '[INFO] Test B2: full Morse, split reale 801/200, hidden=64, n_layers=2, LR costante, 45 epoche\n'
printf '[INFO] Stima: ~2.1 min/epoca -> ~96 min\n'
cd "${RUN_DIR}"
"${TRAINER}" tel22_dataset.bin B2_morse_noscheduler.pt "${CONFIG_NAME}" 2>&1 | tee training_stdout.log

BASELINE_LOG="${BASELINE_LOG}" "${PYTHON_BIN}" - << 'PYEOF'
import csv, os

def load(p):
    if not p or not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))

new = load("cg_training_log.csv")
base = load(os.environ.get("BASELINE_LOG", ""))
if not new:
    print("[WARN] cg_training_log.csv assente."); raise SystemExit(0)

def skill(r, k, z):
    v, zz = float(r[k]), float(r[z])
    return (1 - v / zz) * 100 if zz > 0 else float("nan")

print()
print("=" * 74)
print("  Test B2 — TEL22 full Morse, LR costante, validation VERA held-out")
print("=" * 74)
print(f"  {'ep':>3} | {'train_F':>8} {'val_F':>8} {'skillF%':>8} {'skillT%':>8} | {'base val_F':>10} {'delta':>8}")
print("  " + "-" * 70)
for i, r in enumerate(new):
    b = base[i]["Val_Loss_F_Norm"] if i < len(base) else None
    bs = f"{float(b):10.4f}" if b else " " * 10
    dl = f"{float(r['Val_Loss_F_Norm']) - float(b):+8.4f}" if b else " " * 8
    print(f"  {int(r['Epoch']):3d} | {float(r['Train_Loss_F_Norm']):8.4f} "
          f"{float(r['Val_Loss_F_Norm']):8.4f} "
          f"{skill(r,'Val_Loss_F_Norm','Val_Zero_F_Norm'):8.2f} "
          f"{skill(r,'Val_Loss_T_Norm','Val_Zero_T_Norm'):8.2f} | {bs} {dl}")

bestv = min(new, key=lambda r: float(r["Val_Loss_F_Norm"]))
lastt = float(new[-1]["Train_Loss_F_Norm"])
bskill = skill(bestv, "Val_Loss_F_Norm", "Val_Zero_F_Norm")
print()
print(f"  best val_F  : {float(bestv['Val_Loss_F_Norm']):.4f} (epoca {int(bestv['Epoch'])})  -> skill {bskill:.2f}%")
print(f"  train_F fine: {lastt:.4f}  -> train skill {(1-lastt)*100:.1f}%")
if base:
    bb = min(base[:len(new)], key=lambda r: float(r["Val_Loss_F_Norm"]))
    print(f"  baseline 64x2 stesso range: best val_F {float(bb['Val_Loss_F_Norm']):.4f} "
          f"-> skill {skill(bb,'Val_Loss_F_Norm','Val_Zero_F_Norm'):.2f}%")
print()
train_skill = (1 - lastt) * 100
if bskill > 10 and train_skill > 30:
    print("  VERDETTO: la capacita' era il vincolo. Generalizza -> scalare capacita'/epoche.")
elif train_skill > 30 and bskill <= 5:
    print("  VERDETTO: data-limited. Fitta il train ma non generalizza -> servono piu' frame.")
elif train_skill <= 20:
    print("  VERDETTO: underfit persistente anche a 128x4 -> indagare target/mapping/cutoff,")
    print("            non l'architettura.")
else:
    print("  VERDETTO: miglioramento parziale; valutare piu' epoche prima di conclusioni.")
print("=" * 74)
PYEOF

printf '[INFO] B2 completo in: %s\n' "${RUN_DIR}"
