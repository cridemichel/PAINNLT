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

# Estraiamo solo il primo modello NMR per evitare problemi di sovrapposizione atomica
awk '/^MODEL/ {if (m) exit; m=1} {print} /^ENDMDL/ {exit}' 143D.pdb | grep -v "^MODEL" | grep -v "^ENDMDL" > tel22_clean.pdb

echo "[2] Generazione della Topologia (AMBER99SB-ILDN)..."
# Usiamo -ignh per ignorare gli idrogeni NMR e farli calcolare a GROMACS
gmx pdb2gmx -f tel22_clean.pdb -o tel22_processed.gro -water tip3p -ff amber99sb-ildn -ignh

echo "[3] Moltiplicazione: Inserimento di 10 molecole in un box da 8 nm..."
gmx insert-molecules -ci tel22_processed.gro -nmol 10 -box 8 8 8 -o box_10.gro

# NOTA: insert-molecules non aggiorna automaticamente topol.top
# Aggiungiamo le altre 9 molecole al topol.top (la prima è già lì)
echo "Aggiornamento topol.top..."
sed -E -i '' 's/DNA_chain_A[[:space:]]+1$/DNA_chain_A         10/g' topol.top

echo "[4] Solvatazione..."
gmx solvate -cp box_10.gro -cs spc216.gro -o box_solvated.gro -p topol.top

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
