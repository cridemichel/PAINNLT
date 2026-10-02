#!/usr/bin/env bash
# Quanta capacita' libera ha la A100?  Piu' corse indipendenti sulla stessa GPU.
#
# Il sistema TEL26 (860 siti, ~48 000 archi) e' piccolo per una A100: forward e
# autograd PaiNN (~6 ms) sono dominati dalla latenza dei lanci dei kernel.  Se la
# GPU resta in parte ferma, piu' corse in parallelo sulla stessa GPU aumentano i
# ns/giorno totali senza toccare il modello: la fisica e la precisione di ogni
# corsa restano identiche (stesso codice, FP32 pieno, TF32 disattivato).
#
# Tutto dentro UNA GPU e gli 8 core che le spettano sul nodo (32 core / 4 GPU):
# e' l'unita' di costo dell'allocazione.  Prove:
#   1. scala con i thread di una corsa sola: 1, 2, 4, 8 core (la parte ESPResSo
#      usa davvero i core?);
#   2. N = 2 e 4 corse in parallelo, 8/N core ciascuna, senza MPS;
#   3. le stesse con CUDA MPS (kernel di processi diversi in contemporanea),
#      se nvidia-cuda-mps-control e' disponibile nel job.
# Ogni corsa: grafo PaiNN sul device, Verlet, dt 4 fs, gamma 2, profilo per fasi.
#
# Uso (Booster):
#   sbatch -A IscrB_G4MES -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 \
#       --time=01:00:00 -J tel26_conc --wrap "bash <percorso assoluto>/bench_concurrency.sh"
# Variabili: MODEL, CHECKPOINT, BENCH_STEPS (5000: abbastanza lunghe da
# sovrapporsi nonostante avvii sfalsati di qualche secondo), OUT (bench_conc/).
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
export DEVICE=cuda MLCG_PAINN_GRAPH=device NEIGHBOR_SEARCH=verlet CG_DT=0.004 GAMMA=2
export ENERGY_INTERVAL=10
BENCH_STEPS="${BENCH_STEPS:-5000}"
export CG_STEPS="${BENCH_STEPS}" LOG_INTERVAL="$((BENCH_STEPS / 2))"
OUT="${OUT:-${SCRIPT_DIR}/bench_conc}"
mkdir -p "${OUT}"
RUN05="${SCRIPT_DIR}/05_run_espresso.sh"

# core assegnati al job, in ordine
mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
echo "[bench] core del job: ${CPUS[*]}  (${#CPUS[@]})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

one_run() {   # nome  primo_core  n_core
    local name="$1" first="$2" ncores="$3"
    local list
    list="$(IFS=,; echo "${CPUS[*]:${first}:${ncores}}")"
    OMP_NUM_THREADS="${ncores}" taskset -c "${list}" env \
        SAMPLE_NPZ="${OUT}/${name}.samples.npz" \
        MD_EXTRA_ARGS="--allow_checkpoint_mismatch --no_vtf --painn_profile_report ${OUT}/${name}.json --painn_profile_warmup_calls 50 --energy_file ${OUT}/${name}.energy.csv" \
        bash "${RUN05}" > "${OUT}/${name}.log" 2>&1 \
        || echo "[bench] ${name} FALLITO: vedi ${OUT}/${name}.log"
}

group() {   # prefisso  N   (N corse in parallelo, 8/N core ciascuna)
    local prefix="$1" n="$2" per=$(( ${#CPUS[@]} / $2 )) i
    echo "[bench] ${prefix}: ${n} corse x ${per} core"
    for ((i = 0; i < n; i++)); do
        one_run "${prefix}_r${i}" "$((i * per))" "${per}" &
    done
    wait
}

# 1. scala con i thread, una corsa sola
for t in 1 2 4 8; do
    [[ ${t} -le ${#CPUS[@]} ]] || continue
    echo "[bench] una corsa, ${t} core"
    one_run "single_t${t}" 0 "${t}"
done

# 2. corse in parallelo senza MPS
group nomps_n2 2
group nomps_n4 4

# 3. con CUDA MPS
if command -v nvidia-cuda-mps-control > /dev/null 2>&1; then
    export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_pipe_${SLURM_JOB_ID:-$$}"
    export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_log_${SLURM_JOB_ID:-$$}"
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    if nvidia-cuda-mps-control -d; then
        echo "[bench] MPS attivo"
        group mps_n1 1
        group mps_n2 2
        group mps_n4 4
        echo quit | nvidia-cuda-mps-control
    else
        echo "[bench] MPS non avviabile in questo job: salto la prova 3"
    fi
else
    echo "[bench] nvidia-cuda-mps-control assente: salto la prova 3"
fi

python3 - "${OUT}" <<'EOF' | tee "${OUT}/summary.txt"
import glob, json, os, re, sys
out = sys.argv[1]
rows = {}
for path in sorted(glob.glob(os.path.join(out, "*.json"))):
    name = os.path.basename(path)[:-5]
    m = re.match(r"(.+?)(?:_r\d+)?$", name)
    d = json.load(open(path))
    rows.setdefault(m.group(1), []).append(
        (d["integration"]["wall_ms_per_step"], d["timings_ms"]["total_mean"]))
ref = None
if "single_t8" in rows:
    ref = sum(86400e3 / v[0] * 4e-6 for v in rows["single_t8"])
print(f"{'prova':12s} {'corse':>5s} {'ms/passo (media)':>17s} {'PaiNN ms':>9s} "
      f"{'ns/g per corsa':>15s} {'ns/g totali':>12s} {'vs 1 corsa':>11s}")
order = ["single_t1", "single_t2", "single_t4", "single_t8",
         "nomps_n2", "nomps_n4", "mps_n1", "mps_n2", "mps_n4"]
for key in order + sorted(set(rows) - set(order)):
    if key not in rows:
        continue
    vals = rows[key]
    n = len(vals)
    w = sum(v[0] for v in vals) / n
    p = sum(v[1] for v in vals) / n
    per = 86400e3 / w * 4e-6
    tot = sum(86400e3 / v[0] * 4e-6 for v in vals)
    rel = f"{tot / ref:10.2f}x" if ref else ""
    print(f"{key:12s} {n:5d} {w:17.2f} {p:9.2f} {per:15.2f} {tot:12.2f} {rel:>11s}")
print("\nns/g totali = somma sulle corse in parallelo (dt 4 fs, sola integrazione).")
EOF
echo "[bench] fatto: ${OUT}/summary.txt"
