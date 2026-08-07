import sys

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()
    
marker = 'print("\\n[INFO] Conversione completata e forze residue salvate con successo nel dataset!")'
if marker in content:
    with open("/Users/demichel/.gemini/antigravity/brain/88f2c4a4-4efd-4e31-b158-879a8540a940/scratch/patch_build_v2.py") as f:
        patch_content = f.read()
    decoy_logic = patch_content.split('decoy_logic = """')[1].split('"""')[0]
    content = content.replace(marker, decoy_logic)
    with open(filepath, "w") as f:
        f.write(content)
    print("Injected decoy logic successfully!")
else:
    print("Marker not found!")
