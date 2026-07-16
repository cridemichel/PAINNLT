import numpy as np
import sys
sys.path.append("/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework/preprocessing")
import build_cg_dataset
import json

with open("tel22_topology.json", "r") as f:
    config = json.load(f)

# Evaluate WCA max force at 0.304 nm
wca = config.get("wca_epsilon", 0)
sigma = config.get("wca_sigma", 0)
print(f"WCA eps={wca}, sig={sigma}")
r = 0.304
if wca > 0 and r < sigma * (2**(1/6)):
    sr6 = (sigma/r)**6
    f = 24 * wca * (2 * sr6**2 - sr6) / r
    print(f"WCA force at 0.304 nm: {f}")
else:
    print("WCA force at 0.304 nm: 0")

with open("cg_priors.json", "r") as f:
    priors = json.load(f)

for b in priors.get("bonds", []):
    if b.get("type") == "harmonic":
        k = b.get("k", 0)
        r0 = b.get("r0", 0)
        print(f"Harmonic Bond k={k}, r0={r0}")
        if k > 50000:
            print(f"VERY STRONG BOND! k={k}")

