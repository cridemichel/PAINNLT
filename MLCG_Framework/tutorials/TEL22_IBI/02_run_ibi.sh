#!/bin/bash
set -e
echo "[2/4] Esecuzione Iterative Boltzmann Inversion (IBI)"

# Esegue il loop IBI (3 iterazioni di test)
python ../../ibi/run_ibi_loop.py \
    --dataset tel22_dataset.bin \
    --iterations 3 \
    --outdir ibi_priors

# Inietta il potenziale tabulato in cg_priors.json generandolo ad-hoc per ogni legame
python -c '
import json
import numpy as np
import os

os.makedirs("ibi_priors", exist_ok=True)

with open("cg_priors.json", "r") as f:
    data = json.load(f)

for i, b in enumerate(data.get("bonds", [])):
    if b.get("type", "harmonic") in ["harmonic", "tabulated"]:
        r0 = b["r0"]
        k = b["k"]
        
        # Generiamo una tabella IBI ampia da 0.01 a 3.0 nm centrata sul suo esatto r0
        r_grid = np.linspace(0.01, 3.0, 500)
        V = 0.5 * k * (r_grid - r0)**2
        F = -k * (r_grid - r0)
        
        filename = f"ibi_priors/bond_ibi_spline_{i}.dat"
        np.savetxt(filename, np.column_stack((r_grid, V, F)), fmt="%.6f", header="r energy force")
        
        b["type"] = "tabulated"
        b["file"] = filename
        b["min"] = 0.01
        b["max"] = 3.0

with open("cg_priors.json", "w") as f:
    json.dump(data, f, indent=4)
'

# Genera il dataset residuo sottraendo le forze IBI
python ../../ibi/generate_residual_dataset.py \
    --dataset tel22_dataset.bin \
    --priors cg_priors.json \
    --output tel22_residual_dataset.bin
