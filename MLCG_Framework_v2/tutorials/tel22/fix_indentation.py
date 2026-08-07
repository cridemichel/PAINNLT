filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
in_auto_block = False
for line in lines:
    if line.startswith('    if WCA_SIGMA == "auto":'):
        in_auto_block = True
        new_lines.append(line)
    elif in_auto_block and line.startswith('print("\\n[INFO] Esecuzione allineamento Kabsch per mediare le geometrie dei corpi rigidi...")'):
        in_auto_block = False
        new_lines.append(line)
    elif in_auto_block:
        if line.startswith("    ") or line.strip() == "":
            pass
        else:
            line = "        " + line
        new_lines.append(line)
    else:
        new_lines.append(line)

with open(filepath, "w") as f:
    f.writelines(new_lines)
