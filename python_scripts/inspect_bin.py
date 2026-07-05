import argparse
import struct
import sys

def read_binary_dataset(filename, target_frame=None, target_mol=None):
    try:
        f = open(filename, "rb")
    except FileNotFoundError:
        print(f"Errore: File '{filename}' non trovato.")
        sys.exit(1)

    # Leggi numero di frame
    data = f.read(4)
    if not data:
        print("File vuoto.")
        return
    num_frames = struct.unpack("i", data)[0]
    
    print(f"=== INFORMAZIONI DATASET ===")
    print(f"File: {filename}")
    print(f"Numero totale di frame: {num_frames}")
    print(f"============================\n")

    if target_frame is not None:
        if target_frame < 0 or target_frame >= num_frames:
            print(f"Errore: L'indice del frame deve essere compreso tra 0 e {num_frames-1}")
            sys.exit(1)

    # Scansioniamo i frame
    for frame_idx in range(num_frames):
        # Leggi testata frame
        data = f.read(8)
        if not data:
            break
        num_molecules, num_total_sites = struct.unpack("ii", data)
        
        # Leggi dimensioni box
        data = f.read(12)
        box_x, box_y, box_z = struct.unpack("3f", data)
        
        is_target_frame = (frame_idx == target_frame)
        
        if is_target_frame:
            print(f"--- FRAME {frame_idx} ---")
            print(f"  Molecole: {num_molecules}")
            print(f"  Siti totali: {num_total_sites}")
            print(f"  Box (nm): [{box_x:.4f}, {box_y:.4f}, {box_z:.4f}]")
            print(f"------------------")

        for mol_idx in range(num_molecules):
            data = f.read(8)
            mol_id, num_sites = struct.unpack("ii", data)
            
            data = f.read(12 * 3) # 3 float per center, 3 per force, 3 per torque
            cx, cy, cz, fx, fy, fz, tx, ty, tz = struct.unpack("9f", data)
            
            is_target_mol = is_target_frame and (target_mol is None or target_mol == mol_idx)
            
            if is_target_mol:
                print(f"  Molecola {mol_idx} (ID orig: {mol_id}):")
                print(f"    Siti: {num_sites}")
                print(f"    Centro (nm):  [{cx:.4f}, {cy:.4f}, {cz:.4f}]")
                print(f"    Forza target: [{fx:.4f}, {fy:.4f}, {fz:.4f}]")
                print(f"    Coppia target:[{tx:.4f}, {ty:.4f}, {tz:.4f}]")
                print(f"    Siti:")

            for site_idx in range(num_sites):
                data = f.read(16) # 1 int (type) + 3 float (x,y,z)
                site_type, sx, sy, sz = struct.unpack("i3f", data)
                
                if is_target_mol:
                    print(f"      Sito {site_idx} | Tipo {site_type} | Coord: [{sx:.4f}, {sy:.4f}, {sz:.4f}]")
                    
        if is_target_frame:
            print("\n")
            # Se abbiamo stampato il frame desiderato, possiamo fermarci (a meno che non vogliamo validare tutto)
            break

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ispeziona un dataset binario per CG PaiNN.")
    parser.add_argument("filename", type=str, help="Il percorso del file .bin da leggere")
    parser.add_argument("--frame", type=int, default=None, help="Indice del frame da ispezionare (es. 0)")
    parser.add_argument("--mol", type=int, default=None, help="Indice della molecola da stampare (richiede --frame)")
    
    args = parser.parse_args()
    
    if args.mol is not None and args.frame is None:
        print("Errore: Per stampare una molecola specifica devi specificare anche il --frame.")
        sys.exit(1)
        
    read_binary_dataset(args.filename, args.frame, args.mol)
