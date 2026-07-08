#!/bin/bash
set -e
echo "[2/4] Esecuzione Iterative Boltzmann Inversion (IBI)"

# Esegue il loop IBI (3 iterazioni di test)
python ../../ibi/run_ibi_loop.py \
    --dataset tel22_dataset.bin \
    --iterations 3 \
    --outdir ibi_priors

# Inietta il potenziale tabulato in cg_priors.json
python -c '
import json
with open("cg_priors.json", "r") as f:
    data = json.load(f)
# Sostituisce tutti i legami armonici (non di stacking/Morse) con tabelle IBI
for i, b in enumerate(data.get("bonds", [])):
    if b.get("type", "harmonic") == "harmonic":
        b["type"] = "tabulated"
        b["file"] = "ibi_priors/bond_ibi_final.dat"
        b["min"] = 0.3
        b["max"] = 0.7
with open("cg_priors.json", "w") as f:
    json.dump(data, f, indent=4)
'

# Genera il dataset residuo sottraendo le forze IBI
python ../../ibi/generate_residual_dataset.py \
    --dataset tel22_dataset.bin \
    --priors cg_priors.json \
    --output tel22_residual_dataset.bin
