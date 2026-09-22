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
# Il sistema da trattare.  Una tutorial directory per sistema: tutti i nomi
# dei file che gli stadi producono derivano da SYSTEM, quindi cambiarlo basta
# a spostare l'intera pipeline su un altro G-quadruplex senza toccare altro.
#
#   bash hpc/submit_leonardo.sh dataset SYSTEM=tel26 AA_TRAJECTORY=... AA_TOPOLOGY=...
#
SYSTEM="${SYSTEM:-tel22}"
TUTORIAL="${TUTORIAL:-${FRAMEWORK}/tutorials/${SYSTEM}}"
DATASET="${DATASET:-${SYSTEM}_dataset.bin}"
# Gli script di diagnostica vivono sotto tutorials/tel22 perche' sono nati
# li', ma 32 e 33 prendono il dataset come argomento e non sanno nulla del
# sistema: si riusano cosi' come sono.  44 e 45, invece, importano _tel22_cv,
# che codifica i 22 nucleotidi e le loro coordinate collettive -- restano
# legati a TEL22 finche' non se ne scrive la controparte.
DIAGNOSTICS="${DIAGNOSTICS:-${FRAMEWORK}/tutorials/tel22/diagnostics}"
STAGE="${STAGE:-}"

[[ -d "$TUTORIAL" ]] || { echo "[ERROR] directory di tutorial inesistente: ${TUTORIAL}" >&2; exit 2; }

# La traiettoria all-atom: e' GIA' su Leonardo, va solo indicata.
AA_TRAJECTORY="${AA_TRAJECTORY:-}"
AA_TOPOLOGY="${AA_TOPOLOGY:-}"

echo "[job] stadio ${STAGE} sul sistema ${SYSTEM} su $(hostname) — $(date)"

# shellcheck disable=SC1091
source "${FRAMEWORK}/hpc/env_leonardo.sh"

run() { srun bash -lc "source '${FRAMEWORK}/hpc/env_leonardo.sh'; $*"; }

only_tel22() {
    [[ "$SYSTEM" == "tel22" ]] && return 0
    echo "[ERROR] lo stadio '$1' usa le coordinate collettive di _tel22_cv, che" >&2
    echo "        descrivono i 22 nucleotidi di TEL22: non valgono per ${SYSTEM}." >&2
    echo "        Per ora su ${SYSTEM} si fermano a: dataset, noisefloor, train, production." >&2
    exit 2
}

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
    run "cd '${TUTORIAL}' && AA_TRAJECTORY='${AA_TRAJECTORY}' AA_TOPOLOGY='${AA_TOPOLOGY}' \
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
    run "cd '${TUTORIAL}' && python3 '${DIAGNOSTICS}/scripts/33_check_mean_force_signal.py' '${DATASET}'"
    run "cd '${TUTORIAL}' && python3 '${DIAGNOSTICS}/scripts/32_measure_noise_floor_local.py' '${DATASET}'"
    ;;

train)
    # 03_train_model.sh legge <sistema>_training_config.json e basta: per allenare
    # una variante (C3 spectral, C2 weight decay) si copia il config scelto su
    # quel nome prima di sottomettere, cosi' il file che ha prodotto il modello
    # resta accanto al modello.
    need_env
    run "cd '${TUTORIAL}' && TRAINER='${FRAMEWORK}/training/build/train_painn' bash 03_train_model.sh"
    ;;

production)
    need_env
    # MODEL sceglie il checkpoint, CLASSICAL/DISABLE_ML il controllo solo-prior:
    #   bash hpc/submit_leonardo.sh production SYSTEM=tel26 MODEL=tel26_model.ep20.pt
    #   bash hpc/submit_leonardo.sh production SYSTEM=tel26 CLASSICAL=1 DISABLE_ML=1
    run "cd '${TUTORIAL}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
         ${MODEL:+MODEL='${MODEL}'} ${CLASSICAL:+CLASSICAL='${CLASSICAL}'} \
         ${EQ_DT:+EQ_DT='${EQ_DT}'} bash 04_equilibrate.sh"
    run "cd '${TUTORIAL}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
         ${MODEL:+MODEL='${MODEL}'} ${DISABLE_ML:+DISABLE_ML='${DISABLE_ML}'} \
         ${SAMPLE_NPZ:+SAMPLE_NPZ='${SAMPLE_NPZ}'} \
         CG_STEPS=${CG_STEPS:-10000} bash 05_run_espresso.sh"
    ;;

select)
    # Sweep strutturale sugli snapshot: 1 ps per checkpoint riproduce
    # l'ordinamento di 10 ps, quindi costa ~90 s ciascuno.
    need_env
    only_tel22 "select"
    run "cd '${TUTORIAL}' && MODEL_DIR='${TUTORIAL}' PYTHON_BIN=python3 \
         PYRESSO='${FRAMEWORK}/espresso/build/pypresso' \
         bash diagnostics/scripts/45_select_checkpoint.sh"
    ;;

analysis)
    only_tel22 "analysis"
    run "cd '${TUTORIAL}' && python3 diagnostics/scripts/44_tel22_rdf.py '${DATASET}' run=samples.npz"
    ;;

*)
    echo "[ERROR] STAGE non indicato o non riconosciuto: '${STAGE}'" >&2
    echo "        Attesi: setup | configure | build | dataset | noisefloor | train | select | production | analysis" >&2
    echo "        Sottometti con: bash hpc/submit_leonardo.sh <stadio>" >&2
    exit 2
    ;;
esac

echo "[job] stadio ${STAGE} (${SYSTEM}) concluso — $(date)"
