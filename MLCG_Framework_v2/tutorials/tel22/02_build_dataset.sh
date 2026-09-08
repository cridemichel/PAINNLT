#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILDER="${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py"
TOPOLOGY_VALIDATOR="${SCRIPT_DIR}/diagnostics/scripts/validate_antiparallel_topology.py"

cd "${SCRIPT_DIR}"

# Scelta della traiettoria all-atom.
#
# Il default era md_whole.trr nella radice del tutorial, che e' la produzione
# CORTA: 51 frame a dt 1 ps, cioe' 51 ps.  Con lo split di validazione al 20%
# restano 41 frame di training, che non bastano per addestrare nulla - ed era
# invisibile, perche' niente stampava il conteggio.  La produzione lunga sta in
# long_run/ (1001 frame a dt 10 ps, 10 ns) e viene preferita quando esiste.
#
# La topologia viene presa dalla stessa directory della traiettoria se c'e',
# per non appaiare per sbaglio un .gro di una produzione con la traiettoria di
# un'altra.
if [ -z "${AA_TRAJECTORY:-}" ]; then
    if [ -f long_run/md_whole.trr ]; then
        AA_TRAJECTORY=long_run/md_whole.trr
    else
        AA_TRAJECTORY=md_whole.trr
    fi
fi
if [ -z "${AA_TOPOLOGY:-}" ]; then
    traj_dir="$(dirname "${AA_TRAJECTORY}")"
    if [ -f "${traj_dir}/md.gro" ]; then
        AA_TOPOLOGY="${traj_dir}/md.gro"
    else
        AA_TOPOLOGY=md.gro
    fi
fi

for path in "${AA_TOPOLOGY}" "${AA_TRAJECTORY}" tel22_topology.json; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

# Conteggio dei frame PRIMA di costruire: un dataset troppo corto va visto
# subito, non scoperto a training finito.  Best effort - se MDAnalysis non e'
# disponibile si procede senza il controllo.
echo "[INFO] Topologia:   ${AA_TOPOLOGY}"
echo "[INFO] Traiettoria: ${AA_TRAJECTORY}"
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
    nf="$1"; dt="$2"
    total="$("${PYTHON_BIN}" -c "print(f'{$nf * $dt:.1f}')" 2>/dev/null || echo "?")"
    echo "[INFO] Frame: ${nf} | dt: ${dt} ps | durata: ${total} ps"
    MIN_FRAMES="${MIN_FRAMES:-200}"
    if [ "${nf}" -lt "${MIN_FRAMES}" ]; then
        echo "[ATTENZIONE] Solo ${nf} frame: con validation_fraction 0.2 restano" >&2
        echo "             $(( nf * 4 / 5 )) frame di training. Il segnale apprendibile" >&2
        echo "             del force matching e' dell'ordine dell'1% della varianza" >&2
        echo "             del target, quindi un dataset cosi' corto non addestra." >&2
        if [ -f long_run/md_whole.trr ] && [ "${AA_TRAJECTORY}" != "long_run/md_whole.trr" ]; then
            echo "             Esiste long_run/md_whole.trr: usa" >&2
            echo "               AA_TRAJECTORY=long_run/md_whole.trr bash 02_build_dataset.sh" >&2
        fi
        if [ "${MIN_FRAMES_STRICT:-1}" = "1" ]; then
            echo "[ERROR] Interrompo. Per costruirlo comunque (smoke test):" >&2
            echo "        MIN_FRAMES_STRICT=0 bash 02_build_dataset.sh" >&2
            exit 2
        fi
    fi
else
    echo "[INFO] Conteggio frame non disponibile (MDAnalysis assente): proseguo."
fi

validator_args=(
    --topology tel22_topology.json
    --r0-mode auto
    --require-reference-metadata
)
if [[ -f 143D.pdb ]]; then
    validator_args+=(--pdb 143D.pdb)
fi
"${PYTHON_BIN}" "${TOPOLOGY_VALIDATOR}" "${validator_args[@]}"

"${PYTHON_BIN}" "${BUILDER}" \
    --topology "${AA_TOPOLOGY}" \
    --trajectory "${AA_TRAJECTORY}" \
    --config tel22_topology.json \
    --output tel22_dataset.bin \
    --priors-output cg_priors.json \
    --rb-info-output rigid_bodies_info.json

"${PYTHON_BIN}" "${TOPOLOGY_VALIDATOR}" \
    --topology cg_priors.json \
    --r0-mode numeric

# Conferma di quanti frame sono finiti nel dataset: e' il numero che conta per
# ogni discussione successiva sulla skill e sul pavimento di rumore.
built="$("${PYTHON_BIN}" -c "
import struct
print(struct.unpack('i', open('tel22_dataset.bin','rb').read(4))[0])" 2>/dev/null || echo "?")"
echo "[DONE] tel22_dataset.bin (${built} frame), cg_priors.json, rigid_bodies_info.json"
