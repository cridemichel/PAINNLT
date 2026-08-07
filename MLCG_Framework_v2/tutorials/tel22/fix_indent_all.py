filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"

with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # lines are 0-indexed. line 420 is index 419.
    if i >= 419:
        if line.startswith("    "):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(filepath, "w") as f:
    f.writelines(new_lines)

print("Fixed indentation for lines 420 to end.")
