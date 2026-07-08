# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///

import pymol
import os

def main():
    pymol.pymol_argv = ['pymol', '-qc']
    pymol.finish_launching()
    
    pymol.cmd.load('cg_com_trajectory.pdb', 'tel22')
    pymol.cmd.bg_color('white')
    
    # Go to last frame
    pymol.cmd.frame(100)
    
    pymol.cmd.hide('all')
    
    # La sequenza è A G G G T T A G G G T T A G G G T T A G G G
    # Le guanine sono i residui (1-indexed): 2,3,4 | 8,9,10 | 14,15,16 | 20,21,22
    pymol.cmd.select('guanines', 'chain G and resi 2-4+8-10+14-16+20-22')
    pymol.cmd.select('altri', 'chain G and not guanines')
    
    # Mostriamo lo scheletro di tutto il chain
    pymol.cmd.show('cartoon', 'chain G')
    pymol.cmd.color('gray80', 'altri')
    
    # Mostriamo le sfere SOLO per le guanine e le coloriamo di rosso
    pymol.cmd.show('spheres', 'guanines')
    pymol.cmd.set('sphere_scale', '0.8', 'guanines')
    pymol.cmd.color('red', 'guanines')
    
    pymol.cmd.orient('chain G')
    pymol.cmd.zoom('chain G')
    
    os.makedirs('output', exist_ok=True)
    pymol.cmd.png('output/strand7_last_frame.png', width=800, height=600, dpi=300, ray=1)
    print("Rendered Strand 7 to output/strand7_last_frame.png")

if __name__ == "__main__":
    main()
