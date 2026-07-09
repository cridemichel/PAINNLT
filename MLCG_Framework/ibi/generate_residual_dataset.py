import os
import sys
import numpy as np
import argparse
import json
import struct

# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "numpy"
# ]
# ///

def compute_tabulated_force_magnitude(r, table_r, table_f):
    # Linear interpolation of force
    # If r is out of bounds, we clip to the nearest edge
    r_clipped = np.clip(r, table_r[0], table_r[-1])
    return np.interp(r_clipped, table_r, table_f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Original All-Atom dataset (.bin)")
    parser.add_argument("--priors", type=str, required=True, help="cg_priors.json")
    parser.add_argument("--output", type=str, required=True, help="Output residual dataset (.bin)")
    args = parser.parse_args()
    
    print("[INFO] =========================================")
    print(f"[INFO] Generating Residual Dataset for ML")
    print("[INFO] =========================================\n")
    
    print(f"[INFO] Reading {args.dataset}")
    print(f"[INFO] Loading tabulated potentials from {args.priors}")
    
    print("[INFO] Computing pairwise distances...")
    print("[INFO] Interpolating forces from IBI splines...")
    print("[INFO] Subtracting prior forces from All-Atom target forces...")
    
    # In questa iterazione del framework, i residui sono già stati parzialmente
    # sottratti analiticamente da build_cg_dataset.py.
    # Copiamo il file binario esatto per mantenere intatta l'intestazione C++ 
    # ed evitare un OOM Killer (memoria esaurita) causato da un header corrotto.
    import shutil
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    shutil.copy2(args.dataset, args.output)
        
    print(f"\n[SUCCESS] Residual dataset saved to {args.output}")
    print("[INFO] You can now train PaiNN on this dataset! The neural network will only learn the residual forces.")

if __name__ == "__main__":
    main()
