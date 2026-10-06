#!/bin/bash
# Inventario delle simulazioni all-atom di riferimento per il TEL26 unfoldable.
#
# PERCHE'
#     Prima di scegliere mapping, prior e protocollo serve sapere cosa contengono
#     le corse AA: temperature, durate, frequenza dei frame, forze salvate o no,
#     ioni espliciti (specie e numero), modello d'acqua, box, prestazioni.
#
# USO (Leonardo; meglio come job, gmx check sui .trr grandi richiede minuti)
#     U=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2/tutorials/tel26_unfold
#     sbatch -A IscrB_G4MES_0 -p dcgp_usr_prod --cpus-per-task=4 --mem=8G --time=01:00:00 \
#         -J tel26_inv -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-tel26_inv-%j.out \
#         --wrap "bash $U/00_inventory_aa.sh /leonardo_work/IscrB_G4MES/GiuliaT/AllAtom > $U/inventory_aa.txt 2>&1"
#     Senza gmx disponibile salta solo la sezione dei frame.
set -u
ROOT=${1:-/leonardo_work/IscrB_G4MES/GiuliaT/AllAtom}
[ -d "$ROOT" ] || { echo "[ERROR] $ROOT non esiste o non e' leggibile"; exit 1; }

echo "# inventario di $ROOT"
date
echo
echo "## directory (profondita' 5)"
find "$ROOT" -maxdepth 5 -type d 2>/dev/null | sort

echo
echo "## file GROMACS (dimensione, data, percorso)"
find "$ROOT" -type f \( -name '*.mdp' -o -name '*.tpr' -o -name '*.xtc' -o -name '*.trr' -o -name '*.edr' \
    -o -name '*.gro' -o -name '*.pdb' -o -name '*.top' -o -name '*.itp' -o -name '*.ndx' -o -name '*.log' \
    -o -name '*.cpt' -o -name '*.sh' -o -name '*.slurm' -o -name '*.xvg' \) \
    -printf '%12s B  %TY-%Tm-%Td  %p\n' 2>/dev/null | sort -k4

echo
echo "## parametri dai .mdp"
find "$ROOT" -type f -name '*.mdp' 2>/dev/null | sort | while IFS= read -r f; do
    echo "### $f"
    grep -Ei '^[[:space:]]*(integrator|dt|nsteps|tcoupl|tc[-_]grps|tau[-_]t|ref[-_]t|pcoupl|ref[-_]p|nstxout|nstxout[-_]compressed|compressed[-_]x[-_]grps|nstfout|nstvout|nstenergy|cutoff[-_]scheme|coulombtype|rcoulomb|rvdw|constraints|gen[-_]vel|gen[-_]temp|annealing[a-z_-]*|pull[a-z_-]*)[[:space:]]*=' "$f"
done

echo
echo "## composizione ([ molecules ] dei .top)"
find "$ROOT" -type f -name '*.top' 2>/dev/null | sort | while IFS= read -r f; do
    echo "### $f"
    awk '/^\[ *molecules *\]/{p=1;next} p && /^\[/{p=0} p && NF && $1 !~ /^;/' "$f"
    grep -E '^#include' "$f" | head -20
done

echo
echo "## mdrun: versione, comando, prestazioni (dai .log)"
find "$ROOT" -type f -name '*.log' 2>/dev/null | sort | while IFS= read -r f; do
    grep -q 'GROMACS version' "$f" 2>/dev/null || continue
    echo "### $f"
    grep -m1 'GROMACS version' "$f"
    grep -m1 -A1 'Command line:' "$f" | tail -1
    grep -m1 -E 'Using [0-9]+ MPI|OpenMP threads' "$f"
    grep -E '^Performance:' "$f" | tail -1
    grep -E 'Finished mdrun|^ +Step +Time' "$f" | tail -1
done

echo
echo "## frame e durata delle traiettorie (gmx check)"
GMX=""
for c in gmx gmx_mpi gmx_d; do command -v $c >/dev/null 2>&1 && { GMX=$c; break; }; done
if [ -z "$GMX" ]; then
    for m in "profile/chem-phys" gromacs; do module load $m >/dev/null 2>&1; done
    for c in gmx gmx_mpi gmx_d; do command -v $c >/dev/null 2>&1 && { GMX=$c; break; }; done
fi
if [ -z "$GMX" ]; then
    echo "[WARN] gmx non trovato (provato module load profile/chem-phys gromacs): sezione saltata"
else
    echo "gmx: $(command -v $GMX)"
    find "$ROOT" -type f \( -name '*.xtc' -o -name '*.trr' \) 2>/dev/null | sort | while IFS= read -r f; do
        echo "### $f"
        $GMX check -f "$f" 2>&1 | grep -E '^(Last frame|Step|Time|Coords|Velocities|Forces|Box)|# Atoms|Reading frame' | tail -8
    done
    echo
    echo "## parametri dai .tpr (gmx dump; utili se mancano i .mdp)"
    find "$ROOT" -type f -name '*.tpr' 2>/dev/null | sort | while IFS= read -r f; do
        echo "### $f"
        $GMX dump -s "$f" 2>/dev/null | grep -E '^ +(integrator|nsteps|delta-t|nstxout|nstxout-compressed|nstfout|tcoupl|pcoupl|coulombtype|rcoulomb|rvdw|ref-t|tau-t|ref-p|annealing)[ :=]' | head -20
        $GMX dump -s "$f" 2>/dev/null | grep -m1 -E '^ *natoms'
    done
fi
echo
echo "[DONE]"
