import json
import os

with open("cg_priors.json", "r") as f:
    data = json.load(f)

for cat in ["bonds", "angles", "dihedrals"]:
    for idx, item in enumerate(data.get(cat, [])):
        if item.get("type") in ["ibi", "dbi"]:
            name = item.get("name", f"idx_{idx}")
            item["type"] = "tabulated"
            # Set the min/max bounds based on the category
            if cat == "bonds":
                item["file"] = f"ibi_priors/bond_tabulated_{name}.dat"
                item["min"] = 0.01
                item["max"] = 3.0
            elif cat == "angles":
                item["file"] = f"ibi_priors/angle_tabulated_{name}.dat"
                item["min"] = 0.0
                import numpy as np
                item["max"] = np.pi
            elif cat == "dihedrals":
                item["file"] = f"ibi_priors/dihedral_tabulated_{name}.dat"
                item["min"] = 0.0
                import numpy as np
                item["max"] = 2 * np.pi

with open("cg_priors.json", "w") as f:
    json.dump(data, f, indent=4)
print("Fixed cg_priors.json")
