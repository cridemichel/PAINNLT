#!/bin/bash -l
# Corpo dei job della pipeline TEL22 su Leonardo (CINECA).
#
# NON si sottomette a mano: le risorse cambiano da stadio a stadio e le
# assegna hpc/submit_leonardo.sh, che e' anche l'unico posto dove compaiono
# account, partizioni e QOS.
#
#   bash hpc/submit_leonardo.sh <stadio>
#
# A STADI e non tutto insieme, perche' ogni passo va guardato prima del
# successivo: il conteggio dei frame del dataset, il segnale di forza media,
# il picco della skill, la g(r) per canale.  Su questo problema una pipeline
# che gira dritta fino in fondo produce un numero che non si sa leggere.
#
# La shell e' di login (-l) perche' altrimenti `module` non esiste.
#
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${CINECA_SCRATCH:-$HOME}/PAINNLT}"
FRAMEWORK="${FRAMEWORK:-${PROJECT_ROOT}/MLCG_Framework_v2}"
IMAGE="${IMAGE:-${PROJECT_ROOT}/painn.sif}"
TEL22="${FRAMEWORK}/tutorials/tel22"
STAGE="${STAGE:-}"

# La traiettoria all-atom: e' GIA' su Leonardo, va solo indicata.
AA_TRAJECTORY="${AA_TRAJECTORY:-}"
AA_TOPOLOGY="${AA_TOPOLOGY:-}"

# Il runtime dei container: su Leonardo puo' essere un modulo (apptainer o
# singularity) oppure gia' installato di sistema.  "module load apptainer" da
# solo fa fallire il job in pochi secondi se quel modulefile non esiste, come
# e' successo al primo tentativo.  Qui si prova nell'ordine e si verifica che
# il comando ci sia davvero.
find_container_runtime() {
    local m
    for m in apptainer singularity; do
        if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
            module load "$m" >/dev/null 2>&1 || true
        fi
    done
    APPTAINER="$(command -v apptainer || command -v singularity || true)"
    if [[ -z "$APPTAINER" ]]; then
        echo "[ERROR] nessun runtime di container trovato (ne' apptainer ne' singularity)." >&2
        echo "        Moduli disponibili:" >&2
        module av 2>&1 | grep -i -E 'apptainer|singularity' >&2 || echo "        (nessuno)" >&2
        exit 2
    fi
    echo "[job] runtime container: $APPTAINER ($("$APPTAINER" --version 2>&1 | head -1))"
}
find_container_runtime

echo "[job] stadio ${STAGE} su $(hostname) — $(date)"

# --nv espone le GPU NVIDIA al container; sugli stadi senza GPU e' inerte.
# Il bind monta il progetto con lo STESSO percorso dentro e fuori: cosi' i
# path assoluti nei log, nei manifest e negli errori restano validi da
# entrambi i lati, cosa che un bind su /app rende falsa e che complica ogni
# diagnosi a posteriori.
run() { srun "$APPTAINER" exec --nv --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "$IMAGE" bash -lc "$*"; }

need_image() {
    [[ -e "$IMAGE" ]] && return 0
    echo "[ERROR] immagine assente: $IMAGE" >&2
    echo "        costruiscila prima:  bash hpc/submit_leonardo.sh image" >&2
    exit 2
}

case "$STAGE" in

image)
    # L'unico stadio con bisogno di internet, e per questo gira su
    # lrd_all_serial (nodi di login).  --fakeroot serve perche' la def file
    # installa pacchetti; APPTAINER_CACHEDIR va fuori dalla home, che ha una
    # quota piccola e verrebbe riempita dai layer scaricati.
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_ROOT}/.apptainer_cache}"
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${PROJECT_ROOT}/.apptainer_tmp}"
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
    echo "[image] cache in $APPTAINER_CACHEDIR"
    # Su Leonardo il runtime e' SingularityCE, non Apptainer, e la sezione
    # %arguments con {{ TORCH_TAG }} e' un'estensione di Apptainer che
    # SingularityCE puo' non riconoscere.  Il .def effettivo si genera qui,
    # con il tag gia' sostituito e la sezione %arguments rimossa: cosi' il
    # file versionato resta leggibile e parametrico, e il build non dipende
    # da quale dei due runtime c'e'.
    TORCH_TAG="${TORCH_TAG:-2.5.1-cuda12.4-cudnn9-devel}"
    DEF_GEN="${PROJECT_ROOT}/.painn_leonardo.generated.def"
    awk -v tag="$TORCH_TAG" '
        /^%arguments/ { skip = 1; next }
        /^%/          { skip = 0 }
        skip          { next }
        { gsub(/\{\{[[:space:]]*TORCH_TAG[[:space:]]*\}\}/, tag); print }
    ' "${FRAMEWORK}/hpc/painn_leonardo.def" > "$DEF_GEN"
    echo "[image] tag PyTorch: ${TORCH_TAG}"
    grep -m1 '^From:' "$DEF_GEN"
    "$APPTAINER" build --fakeroot "$IMAGE" "$DEF_GEN"
    ls -lh "$IMAGE"
    ;;

bootstrap)
    # Innesta il plugin, compila ESPResSo e il trainer.  ESPResSo e' gia'
    # stato clonato dal wrapper sul nodo di login: qui non serve rete.
    need_image
    run "JOBS=${JOBS:-8} bash '${FRAMEWORK}/hpc/bootstrap_leonardo.sh'"
    ;;

dataset)
    # Costruisce il dataset CG dalla traiettoria lunga.  Lo script 02 sceglie
    # long_run/md_whole.trr quando esiste, ma qui la traiettoria e' altrove,
    # quindi va passata esplicitamente.
    need_image
    [[ -n "$AA_TRAJECTORY" ]] || { echo "[ERROR] indica AA_TRAJECTORY=/percorso/della/traiettoria.trr" >&2; exit 2; }
    [[ -n "$AA_TOPOLOGY"   ]] || { echo "[ERROR] indica AA_TOPOLOGY=/percorso/della/topologia.gro" >&2; exit 2; }
    # La traiettoria puo' stare fuori dal progetto: allora serve un bind in piu'.
    extra_bind=""
    for p in "$AA_TRAJECTORY" "$AA_TOPOLOGY"; do
        d="$(cd "$(dirname "$p")" && pwd)"
        [[ "$d" == "${PROJECT_ROOT}"* ]] || extra_bind="${extra_bind} --bind ${d}:${d}"
    done
    srun "$APPTAINER" exec --nv --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" ${extra_bind} "$IMAGE" \
        bash -lc "cd '${TEL22}' && AA_TRAJECTORY='${AA_TRAJECTORY}' AA_TOPOLOGY='${AA_TOPOLOGY}' \
                  PYTHON_BIN=python3 bash 02_build_dataset.sh"
    echo
    echo "[NOTA] Prima di allenare, misura il pavimento di rumore sul dataset nuovo:"
    echo "       bash hpc/submit_leonardo.sh noisefloor"
    ;;

noisefloor)
    # Il criterio guida che decide TUTTO il resto: con segnale scarso la
    # selezione va fatta sulla struttura, con segnale abbondante la
    # cross-validation sull'errore di forza torna valida (come in CGnet).
    need_image
    run "cd '${TEL22}' && python3 diagnostics/scripts/33_check_mean_force_signal.py tel22_dataset.bin"
    run "cd '${TEL22}' && python3 diagnostics/scripts/32_measure_noise_floor_local.py tel22_dataset.bin"
    ;;

train)
    need_image
    # 03_train_model.sh legge tel22_training_config.json e basta: per allenare
    # una variante (C3 spectral, C2 weight decay) si copia il config scelto su
    # quel nome prima di sottomettere, cosi' il file che ha prodotto il modello
    # resta accanto al modello.
    run "cd '${TEL22}' && TRAINER='${FRAMEWORK}/training/build/train_painn' bash 03_train_model.sh"
    ;;

production)
    need_image
    run "cd '${TEL22}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda bash 04_equilibrate.sh"
    run "cd '${TEL22}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
         CG_STEPS=${CG_STEPS:-10000} bash 05_run_espresso.sh"
    ;;

select)
    # Sweep strutturale sugli snapshot: 1 ps per checkpoint riproduce
    # l'ordinamento di 10 ps, quindi costa ~90 s ciascuno.
    need_image
    run "cd '${TEL22}' && MODEL_DIR='${TEL22}' PYTHON_BIN=python3 \
         PYRESSO='${FRAMEWORK}/espresso/build/pypresso' \
         bash diagnostics/scripts/45_select_checkpoint.sh"
    ;;

analysis)
    need_image
    run "cd '${TEL22}' && python3 diagnostics/scripts/44_tel22_rdf.py tel22_dataset.bin run=samples.npz"
    ;;

*)
    echo "[ERROR] STAGE non indicato o non riconosciuto: '${STAGE}'" >&2
    echo "        Attesi: image | bootstrap | dataset | noisefloor | train | select | production | analysis" >&2
    echo "        Sottometti con: bash hpc/submit_leonardo.sh <stadio>" >&2
    exit 2
    ;;
esac

echo "[job] stadio ${STAGE} concluso — $(date)"
