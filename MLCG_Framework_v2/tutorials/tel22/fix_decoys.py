import sys

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()
    
# Replace the unpacking
content = content.replace("for (s_name, s_pos, s_type, s_mass) in sites:", "for (s_type, s_pos) in sites:")
# In the translation loop
old_translation_loop = """        for i in range(len(decoy_sites[m2_idx])):
            s_name, s_pos, s_type, s_mass = decoy_sites[m2_idx][i]
            decoy_sites[m2_idx][i] = (s_name, s_pos + translation, s_type, s_mass)"""
new_translation_loop = """        for i in range(len(decoy_sites[m2_idx])):
            s_type, s_pos = decoy_sites[m2_idx][i]
            decoy_sites[m2_idx][i] = (s_type, s_pos + translation)"""
content = content.replace(old_translation_loop, new_translation_loop)

# In the flattening loop
old_flatten_loop = """    for m_idx, sites in enumerate(d_sites):
        for (s_name, s_pos, s_type, s_mass) in sites:
            flat_pos.append(s_pos)
            flat_types.append(s_type)"""
new_flatten_loop = """    for m_idx, sites in enumerate(d_sites):
        for (s_type, s_pos) in sites:
            flat_pos.append(s_pos)
            flat_types.append(s_type)"""
content = content.replace(old_flatten_loop, new_flatten_loop)

with open(filepath, "w") as f:
    f.write(content)
print("Fixed decoys logic!")
