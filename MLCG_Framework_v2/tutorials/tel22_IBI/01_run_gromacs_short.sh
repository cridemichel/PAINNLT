#!/usr/bin/env bash
set -euo pipefail

# Short TEL22 all-atom smoke run for validating the AA -> CG pipeline.
# This is NOT intended as a production-quality equilibrated trajectory.
# Override durations from the environment if desired, e.g.:
#   NVT_PS=50 NPT_PS=50 MD_PS=200 TRAJ_PS=1 bash 01_run_gromacs_short.sh

NVT_PS="${NVT_PS:-20}"
NPT_PS="${NPT_PS:-20}"
MD_PS="${MD_PS:-50}"
TRAJ_PS="${TRAJ_PS:-1}"
SHORT_MDP_DIR="${SHORT_MDP_DIR:-.short_mdp}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "======================================================"
echo " 01. GROMACS SHORT ALL-ATOM RUN (TEL22 G-Quadruplex)  "
echo "======================================================"
echo "[INFO] NVT:        ${NVT_PS} ps"
echo "[INFO] NPT:        ${NPT_PS} ps"
echo "[INFO] Production: ${MD_PS} ps"
echo "[INFO] TRR output: every ${TRAJ_PS} ps"
echo "[WARN] Short-run mode is for pipeline/NVE smoke testing, not final training data."

if ! command -v gmx >/dev/null 2>&1; then
    echo "gmx non trovato! Assicurati di aver installato GROMACS e di aver fatto il source di GMXRC." >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${PYTHON_BIN} non trovato." >&2
    exit 1
fi

for f in mdp/minim.mdp mdp/nvt.mdp mdp/npt.mdp mdp/md.mdp; do
    if [ ! -f "${f}" ]; then
        echo "ERRORE: file MDP mancante: ${f}" >&2
        exit 1
    fi
done

mkdir -p "${SHORT_MDP_DIR}"

# Create a copy of an MDP with selected keys overridden, without modifying the
# original files in mdp/. Duration and output intervals are converted to steps
# using the dt already present in each source MDP.
make_short_mdp() {
    local src="$1"
    local dst="$2"
    local duration_ps="$3"
    local output_ps="${4:-}"

    "${PYTHON_BIN}" - "${src}" "${dst}" "${duration_ps}" "${output_ps}" <<'PY'
import math
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
duration_ps = float(sys.argv[3])
output_arg = sys.argv[4]
output_ps = float(output_arg) if output_arg else None

if duration_ps <= 0:
    raise SystemExit(f"duration must be > 0 ps, got {duration_ps}")
if output_ps is not None and output_ps <= 0:
    raise SystemExit(f"output interval must be > 0 ps, got {output_ps}")

text = src.read_text()

def get_value(key):
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^;\s]+)", re.I | re.M)
    m = pat.search(text)
    return m.group(1) if m else None

dt_raw = get_value("dt")
if dt_raw is None:
    raise SystemExit(f"{src}: missing required 'dt' entry")
try:
    dt = float(dt_raw)
except ValueError as exc:
    raise SystemExit(f"{src}: invalid dt={dt_raw!r}") from exc
if dt <= 0:
    raise SystemExit(f"{src}: dt must be > 0, got {dt}")

def exact_steps(time_ps, label):
    value = time_ps / dt
    steps = int(round(value))
    if steps <= 0 or not math.isclose(steps * dt, time_ps, rel_tol=1e-10, abs_tol=1e-10):
        raise SystemExit(
            f"{src}: {label}={time_ps} ps is not an integer number of steps for dt={dt} ps"
        )
    return steps

overrides = {"nsteps": str(exact_steps(duration_ps, "duration"))}

if output_ps is not None:
    stride = exact_steps(output_ps, "output interval")
    # Keep coordinates, velocities and forces synchronized in the TRR.
    overrides.update({
        "nstxout": str(stride),
        "nstvout": str(stride),
        "nstfout": str(stride),
    })

lines = text.splitlines()
seen = set()
out = []
for line in lines:
    m = re.match(r"^(\s*)([A-Za-z0-9_-]+)(\s*=).*?$", line)
    if m:
        key = m.group(2).lower()
        if key in overrides:
            out.append(f"{m.group(1)}{m.group(2)} = {overrides[key]}")
            seen.add(key)
            continue
    out.append(line)

for key, value in overrides.items():
    if key not in seen:
        out.append(f"{key} = {value}")

out.append("")
out.append("; Added by 01_run_gromacs_short.sh")
out.append(f"; requested duration = {duration_ps:g} ps")
if output_ps is not None:
    out.append(f"; synchronized TRR x/v/f interval = {output_ps:g} ps")

dst.write_text("\n".join(out) + "\n")
print(
    f"[MDP] {src} -> {dst}: dt={dt:g} ps, "
    f"nsteps={overrides['nsteps']}" +
    (f", nstxout=nstvout=nstfout={overrides['nstfout']}" if output_ps is not None else "")
)
PY
}

NVT_MDP="${SHORT_MDP_DIR}/nvt_short.mdp"
NPT_MDP="${SHORT_MDP_DIR}/npt_short.mdp"
MD_MDP="${SHORT_MDP_DIR}/md_short.mdp"

make_short_mdp mdp/nvt.mdp "${NVT_MDP}" "${NVT_PS}"
make_short_mdp mdp/npt.mdp "${NPT_MDP}" "${NPT_PS}"
make_short_mdp mdp/md.mdp  "${MD_MDP}"  "${MD_PS}" "${TRAJ_PS}"

echo "[1] Scaricamento del PDB 143D (G-quadruplex NMR)..."
if [ ! -f 143D.pdb ]; then
    curl -O https://files.rcsb.org/download/143D.pdb
else
    echo "File 143D.pdb gia presente."
fi

# Use only the first NMR model.
awk '/^MODEL/ {if (m) exit; m=1} {print} /^ENDMDL/ {exit}' 143D.pdb \
    | grep -v '^MODEL' \
    | grep -v '^ENDMDL' \
    > tel22_clean.pdb

echo "[2] Generazione della Topologia (AMBER99SB-ILDN)..."
gmx pdb2gmx -f tel22_clean.pdb -o tel22_processed.gro -water tip3p -ff amber99sb-ildn -ignh

echo "[3] Moltiplicazione: Inserimento di 10 molecole in un box da 8 nm..."
gmx insert-molecules -ci tel22_processed.gro -nmol 10 -box 8 8 8 -o box_10.gro

echo "Aggiornamento topol.top..."
sed -E -i '' 's/DNA_chain_A[[:space:]]+1$/DNA_chain_A         10/g' topol.top

echo "[4] Solvatazione..."
gmx solvate -cp box_10.gro -cs spc216.gro -o box_solvated.gro -p topol.top

echo "[5] Aggiunta Ioni (K+ e Cl-)..."
gmx grompp -f mdp/minim.mdp -c box_solvated.gro -p topol.top -o ions.tpr -maxwarn 1
echo "SOL" | gmx genion -s ions.tpr -o box_ions.gro -p topol.top -pname K -nname CL -neutral -conc 0.15

echo "[6] Minimizzazione dell'Energia..."
gmx grompp -f mdp/minim.mdp -c box_ions.gro -p topol.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em

echo "[7] Equilibrazione NVT (${NVT_PS} ps)..."
gmx grompp -f "${NVT_MDP}" -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
gmx mdrun -v -deffnm nvt

echo "[8] Equilibrazione NPT (${NPT_PS} ps)..."
gmx grompp -f "${NPT_MDP}" -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt

echo "[9] MD Production (${MD_PS} ps) per estrarre le forze..."

NSTFOUT=$(awk -F'[=;]' '
    /^[[:space:]]*nstfout[[:space:]]*=/ {
        gsub(/[[:space:]]/, "", $2);
        print $2;
        exit
    }
' "${MD_MDP}")

if [ -z "${NSTFOUT}" ] || ! [[ "${NSTFOUT}" =~ ^[0-9]+$ ]] || [ "${NSTFOUT}" -le 0 ]; then
    echo "ERRORE: ${MD_MDP} deve avere nstfout > 0 per salvare le forze nel TRR." >&2
    echo "Valore trovato: '${NSTFOUT:-non definito}'" >&2
    exit 1
fi

echo "[INFO] nstfout=${NSTFOUT}: coordinate/velocita/forze saranno sincronizzate nel TRR."

gmx grompp -f "${MD_MDP}" -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 1
gmx mdrun -v -deffnm md

if [ ! -s md.trr ]; then
    echo "ERRORE: md.trr non e stato creato oppure e vuoto." >&2
    exit 1
fi

if [ ! -s md.gro ]; then
    echo "ERRORE: md.gro non e stato creato oppure e vuoto." >&2
    exit 1
fi

echo "[10] Srotolamento della traiettoria (rimozione PBC, mantenendo le forze)..."
echo "0" | gmx trjconv -s md.tpr -f md.trr -pbc whole -force -o md_whole.trr

if [ ! -s md_whole.trr ]; then
    echo "ERRORE: md_whole.trr non e stato creato oppure e vuoto." >&2
    exit 1
fi

echo "[INFO] Verifica del TRR finale (deve riportare anche le forze):"
gmx check -f md_whole.trr

echo "======================================================"
echo " Short GROMACS run completato."
echo " File utili per il CG: md_whole.trr e md.gro"
echo " Durate: NVT=${NVT_PS} ps, NPT=${NPT_PS} ps, MD=${MD_PS} ps"
echo " ATTENZIONE: questi dati sono per smoke test, non per training finale."
echo " Ora puoi procedere con 02_build_dataset.sh"
echo "======================================================"
