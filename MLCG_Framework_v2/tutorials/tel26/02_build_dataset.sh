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

: "${AA_TOPOLOGY:?indica AA_TOPOLOGY (la topologia ridotta ai non-Water, vedi extract_solute_topology.py)}"
: "${AA_TRAJECTORY:?indica AA_TRAJECTORY (l'.xtc con le posizioni)}"

for path in "${AA_TOPOLOGY}" "${AA_TRAJECTORY}" tel26_topology.json; do
    [ -f "${path}" ] || { echo "[ERROR] manca: ${path}" >&2; exit 1; }
done

echo "[INFO] Topologia:   ${AA_TOPOLOGY}"
echo "[INFO] Traiettoria: ${AA_TRAJECTORY}"

forces_args=()
if [ -n "${AA_FORCES_TRAJECTORY:-}" ]; then
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

"${PYTHON_BIN}" "${BUILDER}" \
    --topology "${AA_TOPOLOGY}" \
    --trajectory "${AA_TRAJECTORY}" \
    ${forces_args[@]+"${forces_args[@]}"} \
    --config tel26_topology.json \
    --output tel26_dataset.bin \
    --priors-output cg_priors.json \
    --rb-info-output rigid_bodies_info.json

built="$("${PYTHON_BIN}" -c "
import struct
print(struct.unpack('i', open('tel26_dataset.bin','rb').read(4))[0])" 2>/dev/null || echo "?")"
echo "[DONE] tel26_dataset.bin (${built} frame), cg_priors.json, rigid_bodies_info.json"
echo
echo "[NOTA] Prossimo passo: il pavimento di rumore, PRIMA di allenare."
echo "       python3 ../tel22/diagnostics/scripts/33_check_mean_force_signal.py tel26_dataset.bin"
