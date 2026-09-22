#!/bin/bash -l
# Corpo dei job della pipeline TEL22/TEL26 su Leonardo (CINECA).
#
# NON si sottomette a mano: le risorse cambiano da stadio a stadio e le
# assegna hpc/submit_leonardo.sh, che e' anche l'unico posto dove compaiono
# account, partizioni e QOS.
#
#   bash hpc/submit_leonardo.sh <stadio>
#
# Ambiente NATIVO, senza container: Leonardo non ha le mappe subuid/subgid,
# quindi "singularity build --fakeroot" non funziona e un .def con %post non
# e' costruibile qui.  Le dipendenze vengono dai moduli piu' LibTorch e un
# venv, preparati dallo stadio setup.  Vedi hpc/env_leonardo.sh.
#
# La shell e' di login (-l) perche' altrimenti "module" non esiste.
#
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${CINECA_SCRATCH:-$HOME}/PAINNLT}"
FRAMEWORK="${FRAMEWORK:-${PROJECT_ROOT}/MLCG_Framework_v2}"
TEL22="${FRAMEWORK}/tutorials/tel22"
STAGE="${STAGE:-}"

# La traiettoria all-atom: e' GIA' su Leonardo, va solo indicata.
AA_TRAJECTORY="${AA_TRAJECTORY:-}"
AA_TOPOLOGY="${AA_TOPOLOGY:-}"

echo "[job] stadio ${STAGE} su $(hostname) — $(date)"

# shellcheck disable=SC1091
source "${FRAMEWORK}/hpc/env_leonardo.sh"

run() { srun bash -lc "source '${FRAMEWORK}/hpc/env_leonardo.sh'; $*"; }

need_env() {
    [[ -x "${FRAMEWORK}/training/build/train_painn" ]] && return 0
    echo "[ERROR] il trainer non e' compilato: ${FRAMEWORK}/training/build/train_painn" >&2
    echo "        esegui prima:  bash hpc/submit_leonardo.sh setup && bash hpc/submit_leonardo.sh bootstrap" >&2
    exit 2
}

case "$STAGE" in

setup)
    # L'unico stadio con bisogno di rete: LibTorch, i pacchetti Python e il
    # clone di ESPResSo.  Gira su lrd_all_serial, che sta sui nodi di login.
    bash "${FRAMEWORK}/hpc/setup_native.sh"
    ;;

configure)
    # Configurazione di ESPResSo e innesto del plugin.  Serve la RETE: la
    # configurazione scarica heFFTe, Kokkos e Cabana con FetchContent.
    run "STEP=configure bash '${FRAMEWORK}/hpc/bootstrap_leonardo.sh'"
    ;;

build)
    # Compilazione di ESPResSo e del trainer.  Nessuna rete, molti core.
    run "STEP=build JOBS=${JOBS:-8} bash '${FRAMEWORK}/hpc/bootstrap_leonardo.sh'"
    ;;

bootstrap)
    # Retrocompatibilita': configure e build insieme, utile solo dove ci sono
    # sia la rete sia i core.
    run "STEP=all JOBS=${JOBS:-8} bash '${FRAMEWORK}/hpc/bootstrap_leonardo.sh'"
    ;;

dataset)
    # Costruisce il dataset CG dalla traiettoria all-atom.
    [[ -n "$AA_TRAJECTORY" ]] || { echo "[ERROR] indica AA_TRAJECTORY=/percorso/della/traiettoria" >&2; exit 2; }
    [[ -n "$AA_TOPOLOGY"   ]] || { echo "[ERROR] indica AA_TOPOLOGY=/percorso/della/topologia" >&2; exit 2; }
    run "cd '${TEL22}' && AA_TRAJECTORY='${AA_TRAJECTORY}' AA_TOPOLOGY='${AA_TOPOLOGY}' \
         ${AA_FORCES_TRAJECTORY:+AA_FORCES_TRAJECTORY='${AA_FORCES_TRAJECTORY}'} \
         PYTHON_BIN=python3 bash 02_build_dataset.sh"
    echo
    echo "[NOTA] Prima di allenare, misura il pavimento di rumore sul dataset nuovo:"
    echo "       bash hpc/submit_leonardo.sh noisefloor"
    ;;

noisefloor)
    # Il criterio guida che decide TUTTO il resto: con segnale scarso la
    # selezione va fatta sulla struttura, con segnale abbondante la
    # cross-validation sull'errore di forza torna valida (come in CGnet).
    run "cd '${TEL22}' && python3 diagnostics/scripts/33_check_mean_force_signal.py tel22_dataset.bin"
    run "cd '${TEL22}' && python3 diagnostics/scripts/32_measure_noise_floor_local.py tel22_dataset.bin"
    ;;

train)
    # 03_train_model.sh legge tel22_training_config.json e basta: per allenare
    # una variante (C3 spectral, C2 weight decay) si copia il config scelto su
    # quel nome prima di sottomettere, cosi' il file che ha prodotto il modello
    # resta accanto al modello.
    need_env
    run "cd '${TEL22}' && TRAINER='${FRAMEWORK}/training/build/train_painn' bash 03_train_model.sh"
    ;;

production)
    need_env
    run "cd '${TEL22}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda bash 04_equilibrate.sh"
    run "cd '${TEL22}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
         CG_STEPS=${CG_STEPS:-10000} bash 05_run_espresso.sh"
    ;;

select)
    # Sweep strutturale sugli snapshot: 1 ps per checkpoint riproduce
    # l'ordinamento di 10 ps, quindi costa ~90 s ciascuno.
    need_env
    run "cd '${TEL22}' && MODEL_DIR='${TEL22}' PYTHON_BIN=python3 \
         PYRESSO='${FRAMEWORK}/espresso/build/pypresso' \
         bash diagnostics/scripts/45_select_checkpoint.sh"
    ;;

analysis)
    run "cd '${TEL22}' && python3 diagnostics/scripts/44_tel22_rdf.py tel22_dataset.bin run=samples.npz"
    ;;

*)
    echo "[ERROR] STAGE non indicato o non riconosciuto: '${STAGE}'" >&2
    echo "        Attesi: setup | configure | build | dataset | noisefloor | train | select | production | analysis" >&2
    echo "        Sottometti con: bash hpc/submit_leonardo.sh <stadio>" >&2
    exit 2
    ;;
esac

echo "[job] stadio ${STAGE} concluso — $(date)"
