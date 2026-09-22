#!/usr/bin/env bash
# Prepara i due file che tutto il resto della pipeline TEL26 dà per scontati:
#
#   tel26_solute.gro     la topologia ridotta ai soli atomi scritti nell'.xtc
#   tel26_topology.json  la topologia CG (mapping, bond, angoli, tetradi Morse)
#
# PERCHE' SERVE UNO STADIO CHE IL TEL22 NON AVEVA
#   Nel tutorial TEL22 la produzione GROMACS scriveva .trr completi e la
#   topologia CG era scritta a mano, nota la piega del 143D.  Le produzioni
#   TEL26 di Giulia sono diverse sotto due aspetti:
#
#     - compressed-x-grps restringe l'.xtc ai non-Water, quindi l'.xtc ha
#       8680 atomi mentre il .tpr ne descrive 168979.  MDAnalysis rifiuta la
#       coppia: serve una topologia che contenga esattamente gli atomi
#       dell'.xtc, nello stesso ordine.  La estrae extract_solute_topology.py.
#
#     - il 2JPZ e' ibrido (3+1), non antiparallelo: il registro delle tetradi
#       del TEL22 non vale.  build_g4_topology.py lo ricava dalla geometria e
#       lo verifica con un criterio indipendente (ogni tetrade prende una
#       guanina da ciascuno dei quattro tratti).
#
# USO
#   AA_TPR=/percorso/prod-1.tpr AA_XTC=/percorso/prod-1.part0001.xtc \
#       bash 01_prepare_topology.sh
#
# L'OUTPUT VA GUARDATO
#   Lo script stampa le tetradi trovate con lo scarto dal piano e la lunghezza
#   dei lati, e poi i versi dei quattro tratti.  Per il 2JPZ ci si aspetta
#   tre tetradi planari con lati 0.6-1.1 nm e UN tratto controcorrente: e' la
#   firma della piega ibrida.  Quattro versi concordi (parallela) o due e due
#   (antiparallela) vogliono dire che il registro e' sbagliato -- in quel caso
#   si passa a mano con TETRADS="2,10,14,22 ...".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TEMPLATE="${TEMPLATE:-${FRAMEWORK_ROOT}/tutorials/tel22/tel22_topology.json}"
RESIDUES_PER_COPY="${RESIDUES_PER_COPY:-26}"
COPIES="${COPIES:-10}"
N_TETRADS="${N_TETRADS:-3}"
SELECTION="${SELECTION:-not resname SOL WAT HOH TIP3 T3P}"
SOLUTE="${SOLUTE:-tel26_solute.gro}"
TOPOLOGY="${TOPOLOGY:-tel26_topology.json}"

cd "${SCRIPT_DIR}"

: "${AA_TPR:?indica AA_TPR (il .tpr della produzione, con TUTTI gli atomi)}"
: "${AA_XTC:?indica AA_XTC (l'.xtc ristretto dai compressed-x-grps)}"

for path in "${AA_TPR}" "${AA_XTC}" "${TEMPLATE}"; do
    [ -f "${path}" ] || { echo "[ERROR] manca: ${path}" >&2; exit 1; }
done

echo "== 1/2  topologia ridotta agli atomi dell'.xtc =========================="
"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/preprocessing/extract_solute_topology.py" \
    --tpr "${AA_TPR}" \
    --out "${SOLUTE}" \
    --selection "${SELECTION}" \
    --check-xtc "${AA_XTC}"

echo
echo "== 2/2  topologia CG dalla geometria ===================================="
tetrad_args=()
if [ -n "${TETRADS:-}" ]; then
    tetrad_args+=(--tetrads "${TETRADS}")
else
    tetrad_args+=(--n-tetrads "${N_TETRADS}")
fi

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/preprocessing/build_g4_topology.py" \
    --template "${TEMPLATE}" \
    --structure "${SOLUTE}" \
    --residues-per-copy "${RESIDUES_PER_COPY}" \
    --copies "${COPIES}" \
    --output "${TOPOLOGY}" \
    --pdb 2JPZ \
    --fold "ibrida 3+1" \
    ${tetrad_args[@]+"${tetrad_args[@]}"}

echo
echo "[DONE] ${SOLUTE}, ${TOPOLOGY}"
echo
echo "Prossimo passo -- il dataset, con le posizioni dall'.xtc e le forze dal .trr:"
echo "  AA_TOPOLOGY=${SCRIPT_DIR}/${SOLUTE} \\"
echo "  AA_TRAJECTORY=${AA_XTC} \\"
echo "  AA_FORCES_TRAJECTORY=/percorso/prod-1.part0001.trr \\"
echo "  AA_FORCES_TOPOLOGY=${AA_TPR} \\"
echo "      bash 02_build_dataset.sh"
