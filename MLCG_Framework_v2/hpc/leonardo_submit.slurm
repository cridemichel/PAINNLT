#!/bin/bash
# Sottomissione a stadi della pipeline TEL22 su Leonardo (CINECA).
#
# USO
#   sbatch --export=ALL,STAGE=bootstrap leonardo_submit.slurm
#   sbatch --export=ALL,STAGE=dataset   leonardo_submit.slurm
#   sbatch --export=ALL,STAGE=train     leonardo_submit.slurm
#   sbatch --export=ALL,STAGE=production leonardo_submit.slurm
#   sbatch --export=ALL,STAGE=select    leonardo_submit.slurm
#
# A STADI e non tutto insieme, perche' ogni passo va guardato prima del
# successivo: il conteggio dei frame del dataset, il picco della skill, la
# g(r) per canale.  Su questo problema una pipeline che gira dritta fino in
# fondo produce un numero che non si sa leggere.
#
#SBATCH --job-name=mlcg_tel22
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --account=CAMBIA_QUESTO
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ── configurazione ──────────────────────────────────────────────────────────
# ATTENZIONE ALLE QUOTE.  $HOME su Leonardo ha una quota piccola: l'immagine
# .sif pesa alcuni GB e la build di ESPResSo altrettanto, quindi entrambe
# stanno sotto $CINECA_SCRATCH o $WORK, non in home.
PROJECT_ROOT="${PROJECT_ROOT:-${CINECA_SCRATCH:-$HOME}/PAINNLT}"
IMAGE="${IMAGE:-${CINECA_SCRATCH:-$HOME}/painn.sif}"
FRAMEWORK="${PROJECT_ROOT}/MLCG_Framework_v2"
TEL22="${FRAMEWORK}/tutorials/tel22"
STAGE="${STAGE:-}"

# La traiettoria all-atom lunga: e' GIA' su Leonardo, va solo indicata.
AA_TRAJECTORY="${AA_TRAJECTORY:-}"
AA_TOPOLOGY="${AA_TOPOLOGY:-}"

module load apptainer

if [[ ! -e "$IMAGE" ]]; then
    echo "[ERROR] immagine assente: $IMAGE" >&2
    echo "        costruiscila sul nodo di login:" >&2
    echo "          module load apptainer" >&2
    echo "          apptainer build --fakeroot $IMAGE ${FRAMEWORK}/hpc/painn_leonardo.def" >&2
    exit 2
fi

# --nv espone le GPU NVIDIA al container.  Il bind monta il progetto con lo
# STESSO percorso dentro e fuori: cosi' i path assoluti nei log, nei manifest e
# negli errori restano validi da entrambi i lati, cosa che un bind su /app
# rende falsa e che complica ogni diagnosi a posteriori.
run() { srun apptainer exec --nv --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "$IMAGE" bash -lc "$*"; }

case "$STAGE" in

bootstrap)
    # Clona ESPResSo alla versione fissata, innesta il plugin, compila tutto.
    # Non serve GPU per compilare, ma il job la chiede comunque per uniformita'.
    run "bash '${FRAMEWORK}/hpc/bootstrap_leonardo.sh'"
    ;;

dataset)
    # Costruisce il dataset CG dalla traiettoria lunga.  Lo script 02 sceglie
    # long_run/md_whole.trr quando esiste, ma qui la traiettoria e' altrove,
    # quindi va passata esplicitamente.
    [[ -n "$AA_TRAJECTORY" ]] || { echo "[ERROR] indica AA_TRAJECTORY=/percorso/della/traiettoria.trr" >&2; exit 2; }
    [[ -n "$AA_TOPOLOGY"   ]] || { echo "[ERROR] indica AA_TOPOLOGY=/percorso/della/topologia.gro" >&2; exit 2; }
    run "cd '${TEL22}' && AA_TRAJECTORY='${AA_TRAJECTORY}' AA_TOPOLOGY='${AA_TOPOLOGY}' \
         PYTHON_BIN=python3 bash 02_build_dataset.sh"
    echo
    echo "[NOTA] Prima di allenare, misura il pavimento di rumore sul dataset nuovo:"
    echo "       lo script 33 dice se il segnale di forza media e' cresciuto con i dati."
    echo "       Con ~1% di varianza apprendibile la validation loss non ordina i"
    echo "       modelli; con dati alla scala di CGnet (10^6 frame) torna a farlo."
    ;;

noisefloor)
    # Il criterio guida che decide TUTTO il resto: con segnale scarso la
    # selezione va fatta sulla struttura, con segnale abbondante la
    # cross-validation sull'errore di forza torna valida (come in CGnet).
    run "cd '${TEL22}' && python3 diagnostics/scripts/33_check_mean_force_signal.py tel22_dataset.bin"
    run "cd '${TEL22}' && python3 diagnostics/scripts/32_measure_noise_floor_local.py tel22_dataset.bin"
    ;;

train)
    run "cd '${TEL22}' && TRAINER='${FRAMEWORK}/training/build/train_painn' bash 03_train_model.sh"
    ;;

production)
    run "cd '${TEL22}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda bash 04_equilibrate.sh"
    run "cd '${TEL22}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
         CG_STEPS=${CG_STEPS:-10000} bash 05_run_espresso.sh"
    ;;

select)
    # Sweep strutturale sugli snapshot: 1 ps per checkpoint riproduce
    # l'ordinamento di 10 ps, quindi costa ~90 s ciascuno.
    run "cd '${TEL22}' && MODEL_DIR='${TEL22}' PYTHON_BIN=python3 \
         PYRESSO='${FRAMEWORK}/espresso/build/pypresso' \
         bash diagnostics/scripts/45_select_checkpoint.sh"
    ;;

analysis)
    run "cd '${TEL22}' && python3 diagnostics/scripts/44_tel22_rdf.py tel22_dataset.bin run=samples.npz"
    ;;

*)
    echo "[ERROR] STAGE non indicato o non riconosciuto: '${STAGE}'" >&2
    echo "        Attesi: bootstrap | dataset | noisefloor | train | production | select | analysis" >&2
    echo "        Esempio: sbatch --export=ALL,STAGE=bootstrap $0" >&2
    exit 2
    ;;
esac
