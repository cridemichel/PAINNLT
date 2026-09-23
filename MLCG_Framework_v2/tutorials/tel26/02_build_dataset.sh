#!/usr/bin/env bash
# Costruisce il dataset CG del TEL26 ibrido (2JPZ) dalla traiettoria all-atom.
#
# DIVERSO DAL TEL22 IN DUE PUNTI
#   1. Posizioni e forze stanno in file diversi: la produzione ha nstxout=0 e
#      nstfout>0, quindi il .trr contiene le sole forze e le posizioni sono
#      nell'.xtc, ristretto ai non-Water da compressed-x-grps.  Si passano
#      AA_TRAJECTORY (xtc), AA_FORCES_TRAJECTORY (trr) e AA_FORCES_TOPOLOGY
#      (il .tpr completo, che ha tutti gli atomi).
#   2. Niente validate_antiparallel_topology.py: quel validatore assume la
#      piega antiparallela del 143D, mentre il 2JPZ e' ibrido (3+1).  Il
#      registro delle tetradi e' gia' verificato da build_g4_topology.py, che
#      controlla planarita', provenienza dai quattro tratti e versi dei
#      filamenti.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILDER="${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py"

cd "${SCRIPT_DIR}"

# PRIOR_SET: quale topologia (e quindi quali prior) sottrarre; vedi _prior_set.sh.
# shellcheck source=_prior_set.sh
source "${SCRIPT_DIR}/_prior_set.sh"
prior_set_files "${PRIOR_SET:-}"
echo "[INFO] prior: ${PRIOR_SET:-canonico} (${TOPOLOGY_JSON})"

: "${AA_TOPOLOGY:?indica AA_TOPOLOGY (la topologia ridotta ai non-Water, vedi extract_solute_topology.py)}"
: "${AA_TRAJECTORY:?indica AA_TRAJECTORY, la traiettoria compressa con le posizioni}"

for path in "${AA_TOPOLOGY}" "${AA_TRAJECTORY}" "${TOPOLOGY_JSON}"; do
    [ -f "${path}" ] || { echo "[ERROR] manca: ${path}" >&2; exit 1; }
done

echo "[INFO] Topologia:   ${AA_TOPOLOGY}"
echo "[INFO] Traiettoria: ${AA_TRAJECTORY}"

forces_args=()
if [ -n "${AA_FORCES_TRAJECTORY:-}" ]; then
    # Il .trr ha TUTTI gli atomi: senza il .tpr completo il builder userebbe
    # AA_TOPOLOGY, la topologia ridotta, e MDAnalysis rifiuterebbe la coppia
    # solo dopo aver indicizzato 14 GB di .trr.  Meglio fermarsi subito.
    : "${AA_FORCES_TOPOLOGY:?con AA_FORCES_TRAJECTORY serve AA_FORCES_TOPOLOGY, il .tpr completo}"
    [ -f "${AA_FORCES_TOPOLOGY}" ] || { echo "[ERROR] manca: ${AA_FORCES_TOPOLOGY}" >&2; exit 1; }
    forces_args+=(--forces-trajectory "${AA_FORCES_TRAJECTORY}")
    [ -n "${AA_FORCES_TOPOLOGY:-}" ] && forces_args+=(--forces-topology "${AA_FORCES_TOPOLOGY}")
    [ -n "${AA_FORCES_SELECTION:-}" ] && forces_args+=(--forces-selection "${AA_FORCES_SELECTION}")
    echo "[INFO] Forze da:    ${AA_FORCES_TRAJECTORY}"
fi

# Conteggio dei frame prima di costruire: un dataset corto va visto subito.
n_frames="$("${PYTHON_BIN}" - "${AA_TOPOLOGY}" "${AA_TRAJECTORY}" <<'PYEOF' 2>/dev/null || true
import sys, warnings
warnings.filterwarnings("ignore")
try:
    import MDAnalysis as mda
    u = mda.Universe(sys.argv[1], sys.argv[2])
    print(f"{len(u.trajectory)} {u.trajectory.dt:.3f}")
except Exception:
    pass
PYEOF
)"
if [ -n "${n_frames}" ]; then
    set -- ${n_frames}
    echo "[INFO] Frame: $1 | dt: $2 ps"
fi

# MAX_FRAMES e STRIDE servono a misurare il costo prima di impegnare otto ore
# di nodo: la lettura e' lineare nei frame, quindi il tempo su 500 si
# estrapola.  Un dataset troncato NON si usa per allenare.
limit_args=()
[ -n "${MAX_FRAMES:-}" ] && limit_args+=(--max-frames "${MAX_FRAMES}")
[ -n "${STRIDE:-}" ]     && limit_args+=(--stride "${STRIDE}")

"${PYTHON_BIN}" "${BUILDER}" \
    --topology "${AA_TOPOLOGY}" \
    --trajectory "${AA_TRAJECTORY}" \
    ${forces_args[@]+"${forces_args[@]}"} \
    ${limit_args[@]+"${limit_args[@]}"} \
    --config "${TOPOLOGY_JSON}" \
    --output "${DATASET_BIN}" \
    --priors-output "${PRIORS_JSON}" \
    --rb-info-output "${RB_INFO_JSON}"

built="$("${PYTHON_BIN}" -c "
import struct
print(struct.unpack('i', open('${DATASET_BIN}','rb').read(4))[0])" 2>/dev/null || echo "?")"
echo "[DONE] ${DATASET_BIN} (${built} frame), ${PRIORS_JSON}, ${RB_INFO_JSON}"
echo
echo "[NOTA] Prossimo passo: il pavimento di rumore, PRIMA di allenare."
echo "       python3 ../tel22/diagnostics/scripts/33_check_mean_force_signal.py ${DATASET_BIN}"
