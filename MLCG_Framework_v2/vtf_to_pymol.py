import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Converte una traiettoria VTF (da ESPResSo) in PDB e genera uno script PyMOL per il video.')
    parser.add_argument('input_vtf', help='File VTF di input')
    parser.add_argument('--output_prefix', default='trajectory', help='Prefisso per i file PDB e PML di output')
    parser.add_argument('--tel22', action='store_true', help='Applica formattazione e colori specifici per tel22 (22 beads per strand)')
    args = parser.parse_args()

    atoms = []
    bonds = []
    frames = []
    current_frame = {}

    print(f"Lettura di {args.input_vtf} in corso...")
    
    if not os.path.exists(args.input_vtf):
        print(f"Errore: il file {args.input_vtf} non esiste.")
        sys.exit(1)

    with open(args.input_vtf, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'atom':
                atom_info = {}
                for i in range(2, len(parts), 2):
                    atom_info[parts[i]] = parts[i+1]
                atoms.append(atom_info)
            elif parts[0] == 'bond':
                for b in parts[1:]:
                    if ':' in b:
                        i, j = map(int, b.split(':'))
                        bonds.append((i, j))
            elif parts[0] == 'timestep':
                if current_frame:
                    frames.append(current_frame)
                    current_frame = {}
            elif len(parts) == 4:
                try:
                    idx = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    current_frame[idx] = (x, y, z)
                except ValueError:
                    pass

    if current_frame:
        frames.append(current_frame)

    print(f"Caricati {len(atoms)} atomi, {len(bonds)} legami, {len(frames)} frame.")

    pdb_file = f"{args.output_prefix}.pdb"
    pml_file = f"{args.output_prefix}.pml"

    print(f"Scrittura di {pdb_file}...")
    
    # Precalcola gli attributi degli atomi
    atom_attrs = []
    resi = 0
    chain_idx = 0
    for i, atom in enumerate(atoms):
        t = int(atom.get('name', '-1'))
        
        if args.tel22:
            if t == 9:
                resi += 1
                if resi > 22:
                    resi = 1
                    chain_idx += 1
            
            if t >= 10:
                # Ioni
                chain = 'I'
                record = 'HETATM'
                r_id = i - 1000
                name = 'ION'
                resn = 'ION'
            else:
                chain = chr(ord('A') + chain_idx) if chain_idx < 26 else 'A'
                record = 'ATOM  '
                r_id = resi
                if t == 9:
                    name = 'DUM'
                    resn = 'DUM'
                elif t == 0:
                    name = 'CA'
                    resn = 'DA'
                elif t == 1:
                    name = 'CA'
                    resn = 'DT'
                elif t == 2:
                    name = 'CA'
                    resn = 'DG'
                elif 3 <= t <= 7:
                    name = f'B{t-2}'
                    resn = 'DG'
                else:
                    name = 'UNK'
                    resn = 'UNK'
        else:
            resi = 1
            chain = 'A'
            name = atom.get('name', 'C')
            resn = 'UNK'
            record = 'ATOM  '
            r_id = resi
            
        atom_attrs.append((record, name, resn, chain, r_id))

    # Aggiungi i legami per il corpo rigido della Guanina (B1-B5)
    # per far capire a PyMOL che sono un'unica molecola (corpo rigido)
    if args.tel22:
        for i, attr in enumerate(atom_attrs):
            name = attr[1]
            resn = attr[2]
            if name == 'CA' and resn == 'DG':
                # i è CA. i+1=B1, i+2=B2, i+3=B3, i+4=B4, i+5=B5
                # Controlliamo che l'atomo i+5 esista (sicurezza)
                if i + 5 < len(atom_attrs) and atom_attrs[i+5][1] == 'B5':
                    # NON leghiamo CA a B1, altrimenti PyMOL rompe il "cartoon tube" del backbone!
                    # bonds.append((i, i+1))   # CA - B1 (rimosso)
                    bonds.append((i+1, i+2))   # B1 - B2
                    bonds.append((i+2, i+3))   # B2 - B3
                    bonds.append((i+3, i+4))   # B3 - B4
                    bonds.append((i+4, i+5))   # B4 - B5
                    bonds.append((i+5, i+1))   # B5 chiude su B1
                    bonds.append((i+1, i+4))   # B1 incrocia su B4 (anello doppio)

    with open(pdb_file, 'w') as out:
        for f_idx, frame in enumerate(frames):
            out.write(f"MODEL {f_idx + 1:4d}\n")
            for i, (record, name, resn, chain, r_id) in enumerate(atom_attrs):
                x, y, z = frame.get(i, (0.0, 0.0, 0.0))
                out.write(f"{record}{i+1:5d} {name[:4]:<4} {resn[:3]:>3} {chain}{r_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}\n")
            out.write("ENDMDL\n")
        
        for b in bonds:
            out.write(f"CONECT{b[0]+1:5d}{b[1]+1:5d}\n")
    
    print(f"Scrittura dello script PyMOL {pml_file}...")
    with open(pml_file, 'w') as out:
        out.write(f"# Script PyMOL per visualizzare il video della simulazione\n")
        abs_pdb_file = os.path.abspath(pdb_file)
        out.write(f"load {abs_pdb_file}, sim_traj\n")
        out.write("hide all\n")
        
        if args.tel22:
            # Rimuovi completamente DUM e ION
            out.write("remove resn DUM or resn ION\n")
            
            # Backbone: tubo liscio uniforme e sottile
            out.write("show cartoon, name CA\n")
            out.write("hide cartoon, not name CA\n")
            out.write("set cartoon_trace_atoms, 1\n")
            out.write("cartoon tube, name CA\n")
            out.write("set cartoon_tube_radius, 0.1\n")
            
            # Colora di arancione i loop (DA, DT)
            out.write("color orange, resn DA or resn DT\n")
            
            # Mostra le basi delle guanine (solo sferette, niente sticks)
            out.write("show spheres, resn DG and not name CA\n")
            out.write("hide sticks\n")
            out.write("set sphere_scale, 0.07, sim_traj\n")
            
            # Colora le tetradi (tutto il residuo, incluso il backbone CA)
            out.write("select tetrad_1, resi 2+8+14+20\n")
            out.write("color red, tetrad_1\n")
            out.write("select tetrad_2, resi 3+9+15+21\n")
            out.write("color green, tetrad_2\n")
            out.write("select tetrad_3, resi 4+10+16+22\n")
            out.write("color blue, tetrad_3\n")
            out.write("deselect\n")
        else:
            out.write("show spheres, sim_traj\n")
            out.write("set sphere_scale, 0.4\n")
            out.write("show sticks, sim_traj\n")
            out.write("set stick_radius, 0.15\n")
            out.write("color grey50, sim_traj\n")
            
        out.write("bg_color white\n")
        out.write(f"mset 1 - {len(frames)}\n")
        out.write("set movie_fps, 15\n")
        out.write("mplay\n")
        out.write('print("==========================================================")\n')
        out.write('print("Video della simulazione avviato. Usa i controlli in basso a destra per play/pausa.")\n')
        out.write('print("==========================================================")\n')

    print("Completato! Puoi aprire PyMOL ed eseguire lo script con:")
    print(f"  @ {pml_file}")

if __name__ == '__main__':
    main()
