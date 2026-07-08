import json

def generate_topology():
    # Definiamo gli atomi pesanti (no Idrogeni)
    da_atoms = ["P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4", "C3'", "C2'", "O3'"]
    dt_atoms = ["P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'", "C1'", "N1", "C6", "C5", "C7", "C4", "O4", "N3", "C2", "O2", "C3'", "C2'", "O3'"]
    
    # Per la Guanina (DG), dividiamo nei 6 siti richiesti
    # Zucchero e fosfato nel sito 0
    dg_sugar = ["P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'", "C1'", "C3'", "C2'", "O3'"]
    # Basi negli altri 5 siti
    dg_b1 = ["N9", "C4"]
    dg_b2 = ["N3", "C2", "N2"]
    dg_b3 = ["N1", "C6", "O6"]
    dg_b4 = ["C5", "N7"]
    dg_b5 = ["C8"]
    
    config = {
        "temperature": 300.0,
        "wca_sigma": 0.3,
        "wca_epsilon": 1.0,
        "mapping": {
            "mapping_method": "COM",
            "residues": {
                "DA": {
                    "CG_DA": da_atoms
                },
                "DT": {
                    "CG_DT": dt_atoms
                },
                "DG": {
                    "CG_DG_S": dg_sugar,
                    "CG_DG_B1": dg_b1,
                    "CG_DG_B2": dg_b2,
                    "CG_DG_B3": dg_b3,
                    "CG_DG_B4": dg_b4,
                    "CG_DG_B5": dg_b5
                }
            },
            "site_types": {
                "CG_DA": 0,
                "CG_DT": 1,
                "CG_DG_S": 2,
                "CG_DG_B1": 3,
                "CG_DG_B2": 4,
                "CG_DG_B3": 5,
                "CG_DG_B4": 6,
                "CG_DG_B5": 7
            }
        },
        "bonds": []
    }
    
    # 10 chains
    # Sequence: A G G G T T A G G G T T A G G G T T A G G G (22 residues)
    sequence = ["DA", "DG", "DG", "DG", "DT", "DT", "DA", "DG", "DG", "DG", "DT", "DT", "DA", "DG", "DG", "DG", "DT", "DT", "DA", "DG", "DG", "DG"]
    
    for chain_idx in range(10):
        offset = chain_idx * 22
        for i in range(21):
            mol_i = offset + i
            mol_j = offset + i + 1
            
            res_i = sequence[i]
            res_j = sequence[i+1]
            
            # Qual è l'indice del sito da usare per il bond del backbone?
            # Se è DA o DT (1 bead), l'indice del sito è 0
            # Se è DG (6 bead), il sito dello zucchero (che connette il backbone) è il sito 0
            site_i = 0
            site_j = 0
            
            # Formato della lista bonds: [mol_i, mol_j, site_i, site_j]
            config["bonds"].append([mol_i, mol_j, site_i, site_j])

    # Write JSON
    with open("tel22_topology.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("tel22_topology.json generated successfully!")

if __name__ == "__main__":
    generate_topology()
