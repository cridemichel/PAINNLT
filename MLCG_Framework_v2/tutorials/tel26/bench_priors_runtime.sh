#!/usr/bin/env bash
# Quanto costano i prior a dt 4 fs, e quanto si guadagna (opzione C)?
#
# Dopo l'opzione A il passo vale ~20 ms, di cui ~6,7 di PaiNN: il resto sono
# prior, integratore e liste di vicini.  Qui si misura, con il grafo PaiNN sul
# device e il profilo per fasi:
#   - ricerca dei vicini: link-cell (default di 05) contro verlet;
#   - contatti Morse pair-specific: marker sul lato N^2 della decomposizione
#     ibrida (produzione) contro legami Morse analitici sugli estremi fisici
#     (--pair_specific_morse_runtime bonded-analytic, gia' presente come
#     controllo diagnostico: stessi D/a/r0/r_cut ma taglio netto a r_cut invece
#     della coda raccordata C2).
# Piu' un confronto delle forze iniziali marker contro legami, per sapere
# quanto differisce la fisica prima di renderlo esatto.
#
# Uso su un nodo GPU (Booster):
#   sbatch -A IscrB_G4MES -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 \
#       --time=00:45:00 -J tel26_priors --wrap "bash <percorso assoluto>/bench_priors_runtime.sh"
# Variabili: MODEL, CHECKPOINT, DEVICE, BENCH_STEPS (1000), OUT (bench_priors/).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh" ]]; then
    # shellcheck disable=SC1091
    source "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh"
fi
cd "${SCRIPT_DIR}"

export MODEL="${MODEL:-tel26_lp2_re1_it30.pt}"
export CHECKPOINT="${CHECKPOINT:-equilibrated_priors_lp2_100ps.npz}"
export DEVICE="${DEVICE:-cuda}"
export MLCG_PAINN_GRAPH=device
BENCH_STEPS="${BENCH_STEPS:-1000}"
OUT="${OUT:-${SCRIPT_DIR}/bench_priors}"
mkdir -p "${OUT}"
RUN05="${SCRIPT_DIR}/05_run_espresso.sh"
COMMON_ARGS="--allow_checkpoint_mismatch"
echo "[bench] modello ${MODEL}, stato ${CHECKPOINT}, device ${DEVICE}, uscite in ${OUT}"

run_case() {   # nome  neighbor_search  morse_runtime
    local name="$1" search="$2" runtime="$3"
    echo "[bench] profilo: ${name}"
    NEIGHBOR_SEARCH="${search}" CG_STEPS="${BENCH_STEPS}" CG_DT=0.004 GAMMA=2 \
        LOG_INTERVAL="$((BENCH_STEPS / 2))" \
        SAMPLE_NPZ="${OUT}/prof_${name}.samples.npz" \
        MD_EXTRA_ARGS="${COMMON_ARGS} --pair_specific_morse_runtime ${runtime} --painn_profile_report ${OUT}/prof_${name}.json --painn_profile_warmup_calls 50 --no_vtf --energy_file ${OUT}/prof_${name}.energy.csv" \
        bash "${RUN05}" > "${OUT}/log_prof_${name}.txt" 2>&1 \
        || echo "[bench] ${name} FALLITO: vedi ${OUT}/log_prof_${name}.txt"
}

run_case lc_marker link-cell marker-nonbonded
run_case vl_marker verlet    marker-nonbonded
run_case lc_bonded link-cell bonded-analytic
run_case vl_bonded verlet    bonded-analytic

for runtime in marker-nonbonded bonded-analytic; do
    echo "[bench] forze iniziali: ${runtime}"
    CG_STEPS=1 SAMPLE_NPZ="${OUT}/dump_${runtime}.samples.npz" \
        MD_EXTRA_ARGS="${COMMON_ARGS} --pair_specific_morse_runtime ${runtime} --dump_initial_forces ${OUT}/forces_${runtime}.npz" \
        bash "${RUN05}" > "${OUT}/log_dump_${runtime}.txt" 2>&1 \
        || echo "[bench] dump ${runtime} FALLITO: vedi ${OUT}/log_dump_${runtime}.txt"
done

python3 - "${OUT}" <<'EOF' | tee "${OUT}/summary.txt"
import json, os, sys
import numpy as np
out = sys.argv[1]
print(f"{'caso':12s} {'passo (ms)':>11s} {'PaiNN (ms)':>11s} {'resto (ms)':>11s} {'ns/giorno':>10s}")
for name in ("lc_marker", "vl_marker", "lc_bonded", "vl_bonded"):
    path = os.path.join(out, f"prof_{name}.json")
    if not os.path.exists(path):
        print(f"{name:12s} {'mancante':>11s}")
        continue
    d = json.load(open(path))
    w = d["integration"]["wall_ms_per_step"]
    p = d["timings_ms"]["total_mean"]
    print(f"{name:12s} {w:11.2f} {p:11.2f} {w - p:11.2f} {86400e3 / w * 4e-6:10.2f}")
fa, fb = (os.path.join(out, f"forces_{r}.npz") for r in ("marker-nonbonded", "bonded-analytic"))
if os.path.exists(fa) and os.path.exists(fb):
    a, b = np.load(fa), np.load(fb)
    for key in ("force", "torque_lab"):
        scale = np.abs(a[key]).max()
        diff = np.abs(a[key] - b[key]).max()
        print(f"marker contro legami, {key:10s}: max|diff| {diff:.3e}  relativo {diff / scale:.3e}")
EOF
grep -h "Log vettoriale, tempi" "${OUT}"/log_prof_lc_marker.txt 2>/dev/null
echo "[bench] fatto: ${OUT}/summary.txt"
