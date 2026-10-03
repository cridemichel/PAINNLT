#!/usr/bin/env bash
# Verifica del profilo MD "fast" della catena RE contro "legacy".
#
# Stesso modello, stesso stato iniziale, stesso gamma (20) e stessi ps di MD,
# due profili (06_relent.sh, MD_PROFILE):
#   legacy  dt 1 fs, link-cell, PaiNN legacy, energia a ogni registrazione
#   fast    dt 4 fs, Verlet, PaiNN device, energia ogni 10 registrazioni
# Le due MD girano in parallelo sulla stessa GPU; poi lo script 46 confronta la
# struttura campionata dopo il rilassamento (finestra SKIP_PS..MD_PS).  Se
# coincidono entro il rumore, i campioni che train_relent riceve sono
# equivalenti e il profilo fast si puo' usare per le nuove catene.
#
# Uso (Booster, 1 GPU + 8 core):
#   sbatch -A IscrB_G4MES -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 \
#       --time=02:00:00 -J tel26_remd --wrap "bash <percorso assoluto>/compare_relent_md.sh"
# Variabili: MODEL (tel26_lp2_re1_it30.pt), CHECKPOINT (equilibrated_priors_lp2_100ps.npz),
#            MD_PS (100), SKIP_PS (20), SAMPLE_PS (0.1), OUT (cmp_relent_md/).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
[[ -f "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh" ]] && source "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh"
cd "${SCRIPT_DIR}"

export MODEL="${MODEL:-tel26_lp2_re1_it30.pt}"
export CHECKPOINT="${CHECKPOINT:-equilibrated_priors_lp2_100ps.npz}"
export DEVICE="${DEVICE:-cuda}" GAMMA=20
MD_PS="${MD_PS:-100}"; SKIP_PS="${SKIP_PS:-20}"; SAMPLE_PS="${SAMPLE_PS:-0.1}"
OUT="${OUT:-${SCRIPT_DIR}/cmp_relent_md}"
mkdir -p "${OUT}"
mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
HALF=$(( ${#CPUS[@]} / 2 )); (( HALF >= 1 )) || HALF=1
steps() { python3 -c "import sys; print(int(round(float(sys.argv[1]) / float(sys.argv[2]))))" "$1" "$2"; }

run_profile() {   # nome dt vicini grafo energia primo_core
    local name="$1" dt="$2" search="$3" graph="$4" energy="$5" first="$6"
    local cores; cores="$(IFS=,; echo "${CPUS[*]:${first}:${HALF}}")"
    echo "[cmp] ${name}: dt ${dt}, ${search}, PaiNN ${graph}, core ${cores}"
    OMP_NUM_THREADS="${HALF}" taskset -c "${cores}" env \
        CG_DT="${dt}" NEIGHBOR_SEARCH="${search}" MLCG_PAINN_GRAPH="${graph}" ENERGY_INTERVAL="${energy}" \
        CG_STEPS="$(steps "${MD_PS}" "${dt}")" LOG_INTERVAL="$(steps "${SAMPLE_PS}" "${dt}")" \
        SAMPLE_NPZ="${OUT}/${name}.samples.npz" \
        MD_EXTRA_ARGS="--allow_checkpoint_mismatch --no_vtf --energy_file ${OUT}/${name}.energy.csv" \
        bash "${SCRIPT_DIR}/05_run_espresso.sh" > "${OUT}/${name}.log" 2>&1 \
        || echo "[cmp] ${name} FALLITO: vedi ${OUT}/${name}.log"
}
run_profile legacy 0.001 link-cell legacy 1 0 &
run_profile fast   0.004 verlet   device 10 "${HALF}" &
wait
grep -h "\[TIMING\]" "${OUT}"/legacy.log "${OUT}"/fast.log
python3 "${SCRIPT_DIR}/46_tel26_structure.py" tel26_lp1_dataset.bin \
    half2=@ref:0.5:1 \
    legacy="${OUT}/legacy.samples.npz#${SKIP_PS}:${MD_PS}" \
    fast="${OUT}/fast.samples.npz#${SKIP_PS}:${MD_PS}" \
    --ref-range 0:0.5 | tee "${OUT}/struct.txt"
python3 - "${OUT}" "${SKIP_PS}" <<'PY'
import sys
import numpy as np
out, skip = sys.argv[1], float(sys.argv[2])
for name in ("legacy", "fast"):
    d = np.genfromtxt(f"{out}/{name}.energy.csv", delimiter=",", skip_header=1, usecols=(1, 3))
    sel = d[:, 0] >= skip
    print(f"{name:7s} E_kin media {d[sel, 1].mean():.1f}  std {d[sel, 1].std():.1f}  (dopo {skip:g} ps)")
PY
