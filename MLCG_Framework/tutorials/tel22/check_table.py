import json

with open("cg_priors.json", "r") as f:
    priors = json.load(f)

for b in priors.get("bonds", []):
    if b["mol_i"] == 101 and b["mol_j"] == 102:
        print("File:", b["file"])
        with open(b["file"], "r") as tab:
            lines = tab.readlines()
            print("First 5 lines:")
            print("".join(lines[:5]))
            print("Lines around r=0.2:")
            for l in lines:
                parts = l.split()
                if len(parts) > 0 and float(parts[0]) > 0.18 and float(parts[0]) < 0.22:
                    print(l.strip())
        break
