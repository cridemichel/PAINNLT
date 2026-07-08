#!/bin/bash
set -e

echo "======================================================"
echo " 01. GROMACS ALL-ATOM SIMULATION (TEL22 G-Quadruplex) "
echo "======================================================"

# Controlla se GROMACS è installato
if ! command -v gmx &> /dev/null
then
    echo "gmx non trovato! Assicurati di aver installato GROMACS e di aver fatto il source di GMXRC."
    exit 1
fi

echo "[1] Scaricamento del PDB 143D (G-quadruplex NMR)..."
if [ ! -f 143D.pdb ]; then
    curl -O https://files.rcsb.org/download/143D.pdb
else
    echo "File 143D.pdb già presente."
fi

# Estraiamo le catene pulendo il file originale
grep -v "^ENDMDL" 143D.pdb | sed '/^MODEL/d' > tel22_clean.pdb

echo "[2] Generazione della Topologia (AMBER99SB-ILDN)..."
# Usiamo -ignh per ignorare gli idrogeni NMR e farli calcolare a GROMACS
gmx pdb2gmx -f tel22_clean.pdb -o tel22_processed.gro -water tip3p -ff amber99sb-ildn -ignh

echo "[3] Creazione del Box di simulazione..."
# Mettiamo le 6 molecole al centro di un box cubico, con 1.5 nm di distanza dai bordi
gmx editconf -f tel22_processed.gro -o box.gro -c -d 1.5 -bt cubic

echo "[4] Solvatazione..."
gmx solvate -cp box.gro -cs spc216.gro -o box_solvated.gro -p topol.top

echo "[5] Aggiunta Ioni (K+ e Cl-)..."
# Prepariamo un tpr provvisorio per genion
gmx grompp -f mdp/minim.mdp -c box_solvated.gro -p topol.top -o ions.tpr -maxwarn 1
# Neutralizziamo e aggiungiamo 0.15M KCl (Il K+ è vitale per il G-quadruplex!)
echo "SOL" | gmx genion -s ions.tpr -o box_ions.gro -p topol.top -pname K -nname CL -neutral -conc 0.15

echo "[6] Minimizzazione dell'Energia..."
gmx grompp -f mdp/minim.mdp -c box_ions.gro -p topol.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em

echo "[7] Equilibrazione NVT (100 ps)..."
gmx grompp -f mdp/nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
gmx mdrun -v -deffnm nvt

echo "[8] Equilibrazione NPT (100 ps)..."
gmx grompp -f mdp/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt

echo "[9] MD Production (1 ns) per estrarre le forze..."
gmx grompp -f mdp/md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 1
# Lancia la produzione! Questo richiederà un po' di tempo a seconda dei core
gmx mdrun -v -deffnm md

echo "======================================================"
echo " Simulazione GROMACS completata!"
echo " I file utili per il CG sono: md.trr (posizioni, forze) e md.gro (struttura finale)"
echo " Ora puoi procedere con 02_build_dataset.sh"
echo "======================================================"
