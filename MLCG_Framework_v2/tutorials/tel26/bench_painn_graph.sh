#!/usr/bin/env bash
# Confronto dei due percorsi del grafo PaiNN nel plugin ESPResSo:
#   MLCG_PAINN_GRAPH=legacy  loop di coppie di ESPResSo + grafo sull'host (default)
#   MLCG_PAINN_GRAPH=device  ricerca delle coppie e forze sui nodi sul device del modello
#
# Tre prove, stesso modello e stesso stato iniziale:
#   1. parita': forze e coppie dei corpi e E_ML sulla configurazione iniziale
#      (--dump_initial_forces); le differenze sono solo del residuo ML, i prior
#      sono identici;
#   2. profilo per fasi (dt 4 fs, gamma 2), con le fasi CUDA sincronizzate;
#   3. NVE breve (dt 2 fs): deriva di E_tot dei due percorsi a confronto.
#
# Uso su un nodo GPU (Booster), dopo aver ricompilato il plugin:
#   sbatch -A IscrB_G4MES -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 \
#       --time=00:45:00 -J tel26_graph --wrap "bash <percorso assoluto>/bench_painn_graph.sh"
#
# Variabili: MODEL, CHECKPOINT (default re1 it30 / lp2 100 ps), CONFIG (dal manifest), DEVICE (cuda),
#            BENCH_STEPS (1000), NVE_STEPS (2000), OUT (bench_graph/ qui accanto),
#            SKIP_NVE=1 per saltare la prova 3.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh" ]]; then
    # shellcheck disable=SC1091
    source "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh"
fi

export MODEL="${MODEL:-tel26_lp2_re1_it30.pt}"
export CHECKPOINT="${CHECKPOINT:-equilibrated_priors_lp2_100ps.npz}"
# 05_run_espresso.sh legge la config dal manifest del modello con un percorso
# relativo alla cartella corrente: si lavora quindi qui accanto (sbatch --wrap
# parte dalla cartella di sottomissione).
cd "${SCRIPT_DIR}"
export DEVICE="${DEVICE:-cuda}"
BENCH_STEPS="${BENCH_STEPS:-1000}"
NVE_STEPS="${NVE_STEPS:-2000}"
OUT="${OUT:-${SCRIPT_DIR}/bench_graph}"
mkdir -p "${OUT}"
RUN05="${SCRIPT_DIR}/05_run_espresso.sh"
# Energia completa a ogni registrazione: la prova NVE ha bisogno di E_tot.
export ENERGY_INTERVAL=1
# Lo stato iniziale viene da una corsa con i soli prior (altro modello, altra
# config): la provenienza del checkpoint non coincide per costruzione, come
# nelle validazioni e nella catena RE.  Si parte quindi con
# --allow_checkpoint_mismatch; legacy e device partono dallo stesso stato.
COMMON_ARGS="${COMMON_ARGS:---allow_checkpoint_mismatch}"
echo "[bench] modello ${MODEL}, config ${CONFIG:-dal manifest}, stato ${CHECKPOINT}, device ${DEVICE}, uscite in ${OUT}"

# Modi confrontati (MODES, default "legacy device graph"):
#   legacy  loop di coppie di ESPResSo + grafo sull'host
#   device  ricerca delle coppie e forze sui nodi sul device
#   graph   come device, con lista delle coppie a capacita' fissa e forward +
#           autograd catturati in un CUDA graph (MLCG_PAINN_CUDA_GRAPH=1)
MODES="${MODES:-legacy device graph}"
run_mode() {   # modo  log  [variabili...]
    local mode="$1" log="$2"; shift 2
    local graph="${mode}" static=0
    if [[ "${mode}" == "graph" ]]; then graph=device; static=1; fi
    env MLCG_PAINN_GRAPH="${graph}" MLCG_PAINN_CUDA_GRAPH="${static}" "$@" \
        bash "${RUN05}" > "${log}" 2>&1 || echo "[bench] FALLITO: vedi ${log}"
}

# ── 1. parita' ──────────────────────────────────────────────────────────────
for mode in ${MODES}; do
    echo "[bench] parita': ${mode}"
    run_mode "${mode}" "${OUT}/log_dump_${mode}.txt" CG_STEPS=1 \
        SAMPLE_NPZ="${OUT}/dump_${mode}.samples.npz" \
        MD_EXTRA_ARGS="${COMMON_ARGS} --dump_initial_forces ${OUT}/forces_${mode}.npz"
done
python3 - "${OUT}" ${MODES} <<'EOF' | tee "${OUT}/parity.txt"
import os, sys
import numpy as np
out, modes = sys.argv[1], sys.argv[2:]
ref = np.load(os.path.join(out, "forces_legacy.npz"))
ok = True
for mode in modes[1:]:
    path = os.path.join(out, f"forces_{mode}.npz")
    if not os.path.exists(path):
        print(f"{mode}: forze mancanti"); ok = False; continue
    b = np.load(path)
    print(f"legacy contro {mode}:")
    for key in ("force", "torque_lab"):
        scale = np.abs(ref[key]).max()
        diff = np.abs(ref[key] - b[key]).max()
        rel = diff / scale if scale > 0 else diff
        ok &= rel < 1e-4
        print(f"  {key:10s} max|legacy| {scale:.6g}  max|diff| {diff:.3e}  relativo {rel:.3e}")
    ea, eb = float(ref["e_ml"]), float(b["e_ml"])
    print(f"  E_ML       legacy {ea:.9f}  {mode} {eb:.9f}  diff {eb - ea:.3e}")
    ok &= abs(eb - ea) <= 1e-4 * max(1.0, abs(ea))
print("PARITA':", "OK" if ok else "FALLITA")
EOF

# ── 2. profilo per fasi ─────────────────────────────────────────────────────
for mode in ${MODES}; do
    echo "[bench] profilo: ${mode}"
    run_mode "${mode}" "${OUT}/log_prof_${mode}.txt" CG_STEPS="${BENCH_STEPS}" CG_DT=0.004 GAMMA=2 \
        LOG_INTERVAL="$((BENCH_STEPS / 2))" \
        SAMPLE_NPZ="${OUT}/prof_${mode}.samples.npz" \
        MD_EXTRA_ARGS="${COMMON_ARGS} --painn_profile_report ${OUT}/prof_${mode}.json --painn_profile_warmup_calls 50 --no_vtf --energy_file ${OUT}/prof_${mode}.energy.csv"
done
python3 - "${OUT}" ${MODES} <<'EOF' | tee "${OUT}/profile.txt"
import json, os, sys
out, modes = sys.argv[1], [m for m in sys.argv[2:] if os.path.exists(os.path.join(sys.argv[1], f"prof_{m}.json"))]
rows = ("node_index", "neighbor_traversal", "edge_pack", "tensor_inputs", "forward",
        "energy_scalar", "autograd", "force_to_cpu", "force_scatter", "total")
data = [json.load(open(os.path.join(out, f"prof_{m}.json"))) for m in modes]
print(f"{'fase (ms/chiamata)':22s}" + "".join(f"{m:>10s}" for m in modes))
for r in rows:
    print(f"{r:22s}" + "".join(f"{d['timings_ms'].get(r + '_mean', 0.0):10.3f}" for d in data))
w = [d["integration"]["wall_ms_per_step"] for d in data]
print(f"{'passo intero (ms)':22s}" + "".join(f"{x:10.3f}" for x in w))
print(f"{'archi diretti':22s}" + "".join(f"{d['graph']['directed_edges_mean']:10.1f}" for d in data))
print(f"{'catture CUDA graph':22s}" + "".join(f"{d.get('static_graph_captures', 0):10d}" for d in data))
print(f"{'ns/giorno (dt 4 fs)':22s}" + "".join(f"{86400e3 / x * 4e-6:10.2f}" for x in w))
EOF

# ── 3. NVE ──────────────────────────────────────────────────────────────────
if [[ -z "${SKIP_NVE:-}" ]]; then
    for mode in ${MODES}; do
        echo "[bench] NVE: ${mode}"
        run_mode "${mode}" "${OUT}/log_nve_${mode}.txt" CG_STEPS="${NVE_STEPS}" CG_DT=0.002 LOG_INTERVAL=50 \
            SAMPLE_NPZ="${OUT}/nve_${mode}.samples.npz" \
            MD_EXTRA_ARGS="${COMMON_ARGS} --nve --allow_nonconservative_tables --no_vtf --energy_file ${OUT}/nve_${mode}.energy.csv"
    done
    python3 - "${OUT}" ${MODES} <<'EOF' | tee "${OUT}/nve.txt"
import os, sys
import numpy as np
out = sys.argv[1]
for mode in sys.argv[2:]:
    path = os.path.join(out, f"nve_{mode}.energy.csv")
    if not os.path.exists(path):
        print(f"{mode:7s} mancante"); continue
    d = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1, 2, 3))
    t, etot, ekin = d[:, 0], d[:, 1], d[:, 2]
    slope = np.polyfit(t, etot, 1)[0]
    print(f"{mode:7s} E_tot media {etot.mean():.4f}  std {etot.std():.4f}  "
          f"deriva {slope:.4f} kJ/mol/ps  E_kin media {ekin.mean():.1f}")
EOF
fi
echo "[bench] fatto: ${OUT}/parity.txt, profile.txt, nve.txt"
