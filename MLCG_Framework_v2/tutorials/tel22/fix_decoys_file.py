import sys

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()

# Replace the block that writes decoys to binary
old_write_block = """print("[INFO] Scrittura decoy nel binario...")
for d_idx, (d_sites, d_centers, d_forces, d_box) in enumerate(decoy_frames):"""

new_write_block = """print("[INFO] Scrittura decoy nel binario...")
with open(args.output, "ab") as f:
    for d_idx, (d_sites, d_centers, d_forces, d_box) in enumerate(decoy_frames):"""

content = content.replace(old_write_block, new_write_block)

old_end_block = """print(f"[INFO] Scritti {len(decoy_frames)} decoy nel binario.")"""
new_end_block = """print(f"[INFO] Scritti {len(decoy_frames)} decoy nel binario.")
with open(args.output, "r+b") as f:
    f.seek(0)
    total_frames = len(cg_centers_history) + len(decoy_frames)
    f.write(struct.pack("i", total_frames))
print("[INFO] Aggiornato il contatore dei frame totali.")
"""

content = content.replace(old_end_block, new_end_block)

with open(filepath, "w") as f:
    f.write(content)
print("Fixed decoys file writing logic!")
