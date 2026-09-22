/*
 * myconfig.hpp del framework MLCG.
 *
 * PERCHE' STA QUI E NON NELL'ALBERO DI ESPRESSO
 *   ESPResSo decide a COMPILAZIONE quali interazioni esistono: quelle non
 *   dichiarate qui vengono compilate via, e il binario non le espone affatto.
 *   Il default di ESPResSo non include MORSE, mentre la pipeline CG ci fonda
 *   i contatti fra guanine di una tetrade.  Senza, la produzione muore con
 *
 *     RuntimeError: The ESPResSo build does not expose the non-bonded Morse
 *                   interaction.
 *
 *   e l'estensione switched-Morse installata da
 *   install_switched_morse_nonbonded.py, per quanto correttamente innestata,
 *   sparisce insieme al resto perche' e' dentro #ifdef MORSE.
 *
 *   MORSE veniva gia' abilitata da install_switched_morse_nonbonded.py, che
 *   pero' scriveva il suo myconfig in espresso/build/ -- e bootstrap_leonardo
 *   cancella e rigenera la build directory quando cambiano compilatore o
 *   MLCG_TORCH_ROOT, cosa successa piu' volte mentre si sistemava la catena
 *   CUDA.  Il file spariva con la directory, e copy_plugin_files.sh, che lo
 *   riscriverebbe, gira solo nello stadio configure: una ricompilazione senza
 *   riconfigurare lasciava il binario senza MORSE.
 *
 *   Da qui la scelta della posizione: la radice dei sorgenti, che nessuno
 *   cancella.  Una copia nella build directory ha la precedenza, quindi
 *   copy_plugin_files.sh la riallinea a questa.
 *
 * CONTENUTO
 *   Il default di ESPResSo al commit fissato (84cc1d924), piu' MORSE in fondo.
 *   Il commit e' fissato, quindi la copia non diverge; se lo si sposta, va
 *   riallineata a src/config/myconfig-default.hpp della nuova versione.
 */

/*
 * Copyright (C) 2010-2026 The ESPResSo project
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
 *   Max-Planck-Institute for Polymer Research, Theory Group
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


// Geometry, equation of motion, thermostat/barostat
#define ROTATION
#define ROTATIONAL_INERTIA
#define MASS
#define PARTICLE_ANISOTROPY
#define EXTERNAL_FORCES
#define THERMOSTAT_PER_PARTICLE
#define BOND_CONSTRAINT
#define NPT
#define DPD

// Charges and dipoles
#define ELECTROSTATICS
#define DIPOLES

// Active matter
#define ENGINE

// Force/energy calculation
#define EXCLUSIONS

// Long-range interactions
#define TABULATED
#define LENNARD_JONES
#define LENNARD_JONES_GENERIC
#define LJGEN_SOFTCORE
#define LJCOS
#define LJCOS2
#define GAUSSIAN
#define HAT
#define GAY_BERNE
#define SMOOTH_STEP
#define HERTZIAN
#define SOFT_SPHERE
#define WCA

#ifdef FFTW
#define THOLE
#endif

// Further features
#define VIRTUAL_SITES_RELATIVE
#define VIRTUAL_SITES_INERTIALESS_TRACERS
#define VIRTUAL_SITES_CENTER_OF_MASS
#define COLLISION_DETECTION

// Required by MLCG reversible non-bonded Morse contacts.
#define MORSE
