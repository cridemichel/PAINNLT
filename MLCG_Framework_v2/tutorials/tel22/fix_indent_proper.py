filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    lines = f.readlines()

out = []
in_block = False
for line in lines:
    if line.startswith('    if WCA_SIGMA == "auto":'):
        in_block = True
        out.append(line)
        continue
    if in_block and line.startswith('print("\\n[INFO] Esecuzione allineamento Kabsch per mediare le geometrie dei corpi rigidi...")'):
        in_block = False
        out.append(line)
        continue
        
    if in_block:
        if not line.startswith("        ") and line.strip() != "":
            # Needs 4 spaces extra since it currently has some indentation or none?
            # Actually, let's just strip and add 8 spaces for everything? No, the indentation logic inside was correct relative to each other.
            # Let's count current leading spaces
            current_spaces = len(line) - len(line.lstrip(' '))
            if current_spaces < 8:
                out.append(" " * 4 + line)
            else:
                out.append(line)
        else:
            out.append(line)
    else:
        out.append(line)

with open(filepath, "w") as f:
    f.writelines(out)
