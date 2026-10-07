#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p dcgp_usr_prod
#SBATCH -A IscrB_G4MES_0
#SBATCH -t 03:00:00
#SBATCH -J t400_ds
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-t400_ds-%j.out
# Dataset CG (solo posizioni) della produzione AA a 400 K, con la mappatura e i
# prior del modello di produzione lp2 ritagliati a una copia.
#
#   1. gmx trjcat delle tre parti DNA + K+ (01_extract_dna_k.sh) -> $A/cg/prod_dna_k.xtc
#   2. 05_slice_copy.py: tel26_topology.lp2_1c.json e cg_priors.lp2_1c.json
#   3. controllo dei nomi degli atomi del fosfato rispetto al riferimento a 10
#      copie (il mapping del sito S elenca O1P/O2P; nomi diversi vengono saltati
#      in silenzio dal builder: va saputo se i due sistemi sono coerenti)
#   4. build_cg_dataset.py con --priors (niente ristima) e
#      --allow-guard-violations: dataset di posizioni, un frame ogni 10 ps
#
# USO (Leonardo):  sbatch $U/06_build_t400_dataset.sh     [STRIDE=2 per un frame ogni 20 ps]
A=${A:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T=$R/tutorials/tel26
STRIDE=${STRIDE:-1}
mkdir -p $A/cg
cd $A/cg

if [ ! -s prod_dna_k.xtc ]; then
    ( module purge; module load profile/chem-phys; module load gromacs
      gmx trjcat -f $A/dnak/prod-1_dna_k.xtc $A/dnak/prod-2_dna_k.xtc $A/dnak/prod-3_dna_k.xtc -o prod_dna_k.xtc ) || exit 1
fi
module purge
source $R/hpc/env_leonardo.sh
set -euo pipefail

python3 $U/05_slice_copy.py --topology $T/tel26_topology.lp2.json --priors $T/cg_priors.lp2.json \
    --out-topology tel26_topology.lp2_1c.json --out-priors cg_priors.lp2_1c.json

python3 - $A/dnak/equil_dna_k.gro $T/tel26_solute.gro <<'PY'
import sys
for f in sys.argv[1:]:
    L = open(f).read().splitlines()
    names = [l[10:15].strip() for l in L[2:2 + int(L[1])] if l[5:10].strip() == "DG" and int(l[0:5]) == 4]
    print(f"[CHECK] {f}: residuo 4 (DG): {' '.join(names)}")
PY

python3 $R/preprocessing/build_cg_dataset.py \
    --topology $A/dnak/equil_dna_k.gro --trajectory prod_dna_k.xtc \
    --config tel26_topology.lp2_1c.json --priors cg_priors.lp2_1c.json \
    --allow-guard-violations --stride $STRIDE \
    --output tel26_lp2_1c_t400_dataset.bin \
    --priors-output cg_priors.lp2_1c.t400.json --rb-info-output rigid_bodies_info.lp2_1c.t400.json
python3 -c "import struct; print('[DONE] frame:', struct.unpack('i', open('tel26_lp2_1c_t400_dataset.bin','rb').read(4))[0])"
ls -la $A/cg
