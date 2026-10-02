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
# Variabili: MODEL, CHECKPOINT (default re1 it30 / lp2 100 ps), DEVICE (cuda),
#            BENCH_STEPS (1000), NVE_STEPS (2000), OUT (bench_graph/ qui accanto),
#            SKIP_NVE=1 per saltare la prova 3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh" ]]; then
    # shellcheck disable=SC1091
    source "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh"
fi

export MODEL="${MODEL:-tel26_lp2_re1_it30.pt}"
export CHECKPOINT="${CHECKPOINT:-equilibrated_priors_lp2_100ps.npz}"
export DEVICE="${DEVICE:-cuda}"
BENCH_STEPS="${BENCH_STEPS:-1000}"
NVE_STEPS="${NVE_STEPS:-2000}"
OUT="${OUT:-${SCRIPT_DIR}/bench_graph}"
mkdir -p "${OUT}"
RUN05="${SCRIPT_DIR}/05_run_espresso.sh"
echo "[bench] modello ${MODEL}, stato ${CHECKPOINT}, device ${DEVICE}, uscite in ${OUT}"

# ── 1. parita' ──────────────────────────────────────────────────────────────
for mode in legacy device; do
    echo "[bench] parita': ${mode}"
    MLCG_PAINN_GRAPH="${mode}" CG_STEPS=1 \
        SAMPLE_NPZ="${OUT}/dump_${mode}.samples.npz" \
        MD_EXTRA_ARGS="--dump_initial_forces ${OUT}/forces_${mode}.npz" \
        bash "${RUN05}" > "${OUT}/log_dump_${mode}.txt" 2>&1
done
python3 - "${OUT}/forces_legacy.npz" "${OUT}/forces_device.npz" <<'EOF' | tee "${OUT}/parity.txt"
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
print(f"percorsi: {a['painn_graph']} contro {b['painn_graph']}")
ok = True
for key in ("force", "torque_lab"):
    fa, fb = a[key], b[key]
    scale = np.abs(fa).max()
    diff = np.abs(fa - fb).max()
    rel = diff / scale if scale > 0 else diff
    ok &= rel < 1e-4
    print(f"{key:10s} max|legacy| {scale:.6g}  max|diff| {diff:.3e}  relativo {rel:.3e}")
ea, eb = float(a["e_ml"]), float(b["e_ml"])
print(f"E_ML       legacy {ea:.9f}  device {eb:.9f}  diff {eb - ea:.3e}")
ok &= abs(eb - ea) <= 1e-4 * max(1.0, abs(ea))
print("PARITA':", "OK" if ok else "FALLITA")
EOF

# ── 2. profilo per fasi ─────────────────────────────────────────────────────
for mode in legacy device; do
    echo "[bench] profilo: ${mode}"
    MLCG_PAINN_GRAPH="${mode}" CG_STEPS="${BENCH_STEPS}" CG_DT=0.004 GAMMA=2 \
        LOG_INTERVAL="$((BENCH_STEPS / 2))" \
        SAMPLE_NPZ="${OUT}/prof_${mode}.samples.npz" \
        MD_EXTRA_ARGS="--painn_profile_report ${OUT}/prof_${mode}.json --painn_profile_warmup_calls 50 --no_vtf --energy_file ${OUT}/prof_${mode}.energy.csv" \
        bash "${RUN05}" > "${OUT}/log_prof_${mode}.txt" 2>&1
done
python3 - "${OUT}/prof_legacy.json" "${OUT}/prof_device.json" <<'EOF' | tee "${OUT}/profile.txt"
import json, sys
rows = ("node_index", "neighbor_traversal", "edge_pack", "tensor_inputs", "forward",
        "energy_scalar", "autograd", "force_to_cpu", "force_scatter", "total")
data = [json.load(open(p)) for p in sys.argv[1:]]
print(f"{'fase (ms/chiamata)':22s} {'legacy':>10s} {'device':>10s}")
for r in rows:
    vals = [d["timings_ms"].get(f"{r}_mean", 0.0) for d in data]
    print(f"{r:22s} {vals[0]:10.3f} {vals[1]:10.3f}")
w = [d["integration"]["wall_ms_per_step"] for d in data]
print(f"{'passo intero (ms)':22s} {w[0]:10.3f} {w[1]:10.3f}")
e = [d["graph"]["directed_edges_mean"] for d in data]
print(f"{'archi diretti':22s} {e[0]:10.1f} {e[1]:10.1f}")
print(f"ns/giorno a dt 4 fs (solo integrazione): legacy {86400e3 / w[0] * 4e-6:.2f}, "
      f"device {86400e3 / w[1] * 4e-6:.2f}  (guadagno {w[0] / w[1]:.2f}x)")
EOF

# ── 3. NVE ──────────────────────────────────────────────────────────────────
if [[ -z "${SKIP_NVE:-}" ]]; then
    for mode in legacy device; do
        echo "[bench] NVE: ${mode}"
        MLCG_PAINN_GRAPH="${mode}" CG_STEPS="${NVE_STEPS}" CG_DT=0.002 LOG_INTERVAL=50 \
            SAMPLE_NPZ="${OUT}/nve_${mode}.samples.npz" \
            MD_EXTRA_ARGS="--nve --allow_nonconservative_tables --no_vtf --energy_file ${OUT}/nve_${mode}.energy.csv" \
            bash "${RUN05}" > "${OUT}/log_nve_${mode}.txt" 2>&1
    done
    python3 - "${OUT}/nve_legacy.energy.csv" "${OUT}/nve_device.energy.csv" <<'EOF' | tee "${OUT}/nve.txt"
import sys
import numpy as np
for path, mode in zip(sys.argv[1:], ("legacy", "device")):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1, 2, 3))
    t, etot, ekin = d[:, 0], d[:, 1], d[:, 2]
    slope = np.polyfit(t, etot, 1)[0]
    print(f"{mode:7s} E_tot media {etot.mean():.4f}  std {etot.std():.4f}  "
          f"deriva {slope:.4f} kJ/mol/ps  E_kin media {ekin.mean():.1f}")
EOF
fi
echo "[bench] fatto: ${OUT}/parity.txt, profile.txt, nve.txt"
