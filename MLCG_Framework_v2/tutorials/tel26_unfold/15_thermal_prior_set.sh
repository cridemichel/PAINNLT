#!/bin/bash
# Insieme di prior TERMICI per l'allenamento di PaiNN sul sistema a 10 copie (tutorials/tel26).
#
# Dai prior lp2 (cg_priors.lp2.json, 10 copie) e dai parametri di D(T) validati con
# 13_validate_thermal.sh (insieme f: lam_tet ~ 0,08 costante, lam_oth = 1,094 (1 - T/373,7 K))
# scrive con 09_scale_contacts.py cg_priors.<SET>.json (D_H, D_S; D = null: run_cg_md lo
# risolve alla T del termostato, 299,5 K in 06_relent.sh) e crea per <SET> i nomi che
# _prior_set.sh si aspetta:
#   rigid_bodies_info.<SET>.json, tel26_topology.<SET>.json  -> link a quelli di lp2 (identici)
#   tel26_<SET>_dataset.bin                                    -> link a tel26_lp2_dataset.bin
# ATTENZIONE: il dataset di lp2 contiene le forze residue F_AA - F_prior(lp2).  L'entropia
# relativa (06_relent.sh, train_relent) usa solo le POSIZIONI e va bene; il force matching
# (03) su <SET> sarebbe sbagliato: per quello va ricostruito il dataset con i suoi prior.
#
# USO (Leonardo, login):  bash $U/15_thermal_prior_set.sh
# Variabili: SET (lp2th), BASE (lp2), H_TET T0_TET H_OTH T0_OTH (insieme f), T26 (tutorials/tel26)
set -euo pipefail
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=${T26:-$R/tutorials/tel26}
SET=${SET:-lp2th}
BASE=${BASE:-lp2}
H_TET=${H_TET:-0.09309}; T0_TET=${T0_TET:-2560.2}
H_OTH=${H_OTH:-1.09415}; T0_OTH=${T0_OTH:-373.66}
cd "$T26"
for f in cg_priors.$BASE.json rigid_bodies_info.$BASE.json tel26_topology.$BASE.json tel26_${BASE}_dataset.bin; do
    [[ -e $f ]] || { echo "[ERROR] manca $T26/$f" >&2; exit 1; }
done
for f in cg_priors.$SET.json rigid_bodies_info.$SET.json tel26_topology.$SET.json tel26_${SET}_dataset.bin; do
    [[ -e $f || -L $f ]] && { echo "[ERROR] $f esiste gia': scegli un altro SET o rinominalo" >&2; exit 1; }
done
python3 $U/09_scale_contacts.py --in cg_priors.$BASE.json --out cg_priors.$SET.json \
    --h-tet $H_TET --t0-tet $T0_TET --h-oth $H_OTH --t0-oth $T0_OTH --report-T 299.5,320,330,340
ln -s rigid_bodies_info.$BASE.json rigid_bodies_info.$SET.json
ln -s tel26_topology.$BASE.json tel26_topology.$SET.json
ln -s tel26_${BASE}_dataset.bin tel26_${SET}_dataset.bin
python3 - cg_priors.$SET.json <<'PY'
import json, sys, collections
p = json.load(open(sys.argv[1]))
m = [b for b in p["bonds"] if b.get("type") == "morse"]
c = collections.Counter("termico" if b.get("D") is None and "D_H" in b else "fisso" for b in m)
tw = [d for d in p.get("dihedrals", []) if d.get("role") == "twist"]
print(f"[INFO] {sys.argv[1]}: {len(m)} Morse ({dict(c)}), {len(tw)} torsioni twist "
      f"({sum(1 for d in tw if d.get('cbt'))} CBT, {sum(1 for d in tw if 'k_H' in d)} termiche)")
PY
ls -l cg_priors.$SET.json rigid_bodies_info.$SET.json tel26_topology.$SET.json tel26_${SET}_dataset.bin
echo "[DONE] insieme $SET in $T26.  Stati di partenza disponibili per 06_relent.sh:"
ls -t state_tel26_${BASE}_*it*.npz equilibrated*${BASE}*.npz 2>/dev/null | head -5 || true
