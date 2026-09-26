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
# PRIOR_SET: l'insieme di prior (tutorials/tel26/_prior_set.sh).  Cambia il
# dataset, perche' le forze allenate sono il residuo rispetto a quei prior.
PRIOR_SET="${PRIOR_SET:-}"
DATASET="${DATASET:-${SYSTEM}${PRIOR_SET:+_${PRIOR_SET}}_dataset.bin}"
# Gli script di diagnostica vivono sotto tutorials/tel22 perche' sono nati
# li', ma 32 e 33 prendono il dataset come argomento e non sanno nulla del
# sistema: si riusano cosi' come sono.  44 e 45, invece, importano _tel22_cv,
# che codifica i 22 nucleotidi e le loro coordinate collettive -- restano
# legati a TEL22 finche' non se ne scrive la controparte.
DIAGNOSTICS="${DIAGNOSTICS:-${FRAMEWORK}/tutorials/tel22/diagnostics}"
# Residui per copia: lo script 44 ne ha bisogno per spezzare le molecole in
# copie, ed e' l'unica cosa che nella g(r) dipendeva dal sistema.  Si legge
# dalla topologia generata invece di farselo dire, cosi' non puo' divergere.
if [[ -z "${NUC:-}" && -f "${TUTORIAL}/${SYSTEM}_topology.json" ]]; then
    NUC=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1]))['g4_topology']['residues_per_copy'])
except Exception:
    pass" "${TUTORIAL}/${SYSTEM}_topology.json" 2>/dev/null)
fi
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
         ${AA_FORCES_TOPOLOGY:+AA_FORCES_TOPOLOGY='${AA_FORCES_TOPOLOGY}'} \
         ${PRIOR_SET:+PRIOR_SET='${PRIOR_SET}'} ${PART:+PART='${PART}'} \
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
    # Una variante si allena con RUN=<nome>: legge <sistema>_training_config.<nome>.json
    # e scrive <sistema>_<nome>_model.pt piu' i suoi snapshot.  NON si
    # sovrascrive la config canonica: la produzione valida l'architettura contro
    # la config, e i modelli gia' allenati diventerebbero inutilizzabili.
    #   bash hpc/submit_leonardo.sh train SYSTEM=tel26 RUN=d64
    need_env
    run "cd '${TUTORIAL}' && TRAINER='${FRAMEWORK}/training/build/train_painn' \
         ${RUN:+RUN='${RUN}'} ${PRIOR_SET:+PRIOR_SET='${PRIOR_SET}'} bash 03_train_model.sh"
    ;;

production)
    need_env
    # MODEL sceglie il checkpoint, CLASSICAL/DISABLE_ML il controllo solo-prior:
    #   bash hpc/submit_leonardo.sh production SYSTEM=tel26 MODEL=tel26_model.ep20.pt
    #   bash hpc/submit_leonardo.sh production SYSTEM=tel26 CLASSICAL=1 DISABLE_ML=1
    #
    # I nomi dello stato equilibrato e della traiettoria si DERIVANO dal modello.
    # Erano fissi, e uno sweep di produzioni in parallelo nella stessa directory
    # faceva sovrascrivere equilibrated.npz a tutti: la produzione di un modello
    # poteva partire dallo stato equilibrato di un altro, in silenzio.  Cosi'
    # invece basta passare MODEL e la collisione non e' possibile.
    if [[ -n "${MODEL:-}" ]]; then
        _tag="${MODEL##*/}"; _tag="${_tag%.pt}"
        # RUN_TAG distingue piu' corse dello stesso modello (protocolli diversi)
        # senza sovrascriverne i file.
        [[ -n "${RUN_TAG:-}" ]] && _tag="${_tag}_${RUN_TAG}"
        CHECKPOINT="${CHECKPOINT:-equilibrated_${_tag}.npz}"
        SAMPLE_NPZ="${SAMPLE_NPZ:-samples_${_tag}.npz}"
    fi
    if [[ -n "${DISABLE_ML:-}${CLASSICAL:-}" ]]; then
        CHECKPOINT="${CHECKPOINT:-equilibrated_priors${PRIOR_SET:+_${PRIOR_SET}}${RUN_TAG:+_${RUN_TAG}}.npz}"
        SAMPLE_NPZ="${SAMPLE_NPZ:-samples_priors${PRIOR_SET:+_${PRIOR_SET}}${RUN_TAG:+_${RUN_TAG}}.npz}"
    fi
    echo "[job] modello ${MODEL:-tel26_model.pt} -> ${CHECKPOINT:-equilibrated.npz}, ${SAMPLE_NPZ:-samples.npz}"
    run "cd '${TUTORIAL}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
         ${MODEL:+MODEL='${MODEL}'} ${CLASSICAL:+CLASSICAL='${CLASSICAL}'} \
         ${PRIOR_SET:+PRIOR_SET='${PRIOR_SET}'} \
         ${CHECKPOINT:+CHECKPOINT='${CHECKPOINT}'} \
         ${EQ_DT:+EQ_DT='${EQ_DT}'} ${EQ_GAMMA:+EQ_GAMMA='${EQ_GAMMA}'} \
         ${EQ_ML_STEPS:+EQ_ML_STEPS='${EQ_ML_STEPS}'} bash 04_equilibrate.sh"
    run "cd '${TUTORIAL}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
         ${MODEL:+MODEL='${MODEL}'} ${DISABLE_ML:+DISABLE_ML='${DISABLE_ML}'} \
         ${PRIOR_SET:+PRIOR_SET='${PRIOR_SET}'} \
         ${CHECKPOINT:+CHECKPOINT='${CHECKPOINT}'} \
         ${SAMPLE_NPZ:+SAMPLE_NPZ='${SAMPLE_NPZ}'} \
         ${GAMMA:+GAMMA='${GAMMA}'} \
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

iterate)
    # Ciclo automatico di aggiustamento dei contatti (iterate_contacts.py), a
    # soli prior, senza nessuno nel mezzo: il collo di bottiglia non era il
    # calcolo ma il giro umano fra un'iterazione e l'altra.  Un job = una
    # catena; catene diverse (insiemi di partenza, DAMP) girano in parallelo,
    # una GPU ciascuna.
    #
    #   bash hpc/submit_leonardo.sh iterate SYSTEM=tel26 \
    #        START_TOPOLOGY=tel26_topology.b3tw.json \
    #        START_RUN=samples_priors_b3tw_100ps.npz PREFIX=b3twit NITER=5
    #
    # Iterazione n: iterate_contacts.py sulla corsa precedente -> topologia
    # <PREFIX>n, derive_prior_set.py (base BASE_SET, default b3stack), 04 e 05
    # con i soli prior.  Si ferma quando iterate_contacts.py stampa
    # [CONVERGIUTO] (la topologia buona e' allora quella della corsa appena
    # valutata) o dopo NITER iterazioni.
    need_env
    : "${START_TOPOLOGY:?indica START_TOPOLOGY, la topologia della corsa di partenza}"
    : "${START_RUN:?indica START_RUN, il samples_*.npz della corsa di partenza}"
    : "${PREFIX:?indica PREFIX, il nome base degli insiemi (es. b3twit)}"
    _topo="${START_TOPOLOGY}"
    _npz="${START_RUN}"
    for _n in $(seq 1 "${NITER:-5}"); do
        _ps="${PREFIX}${_n}"
        echo "[job] iterazione ${_n}: ${_topo} + ${_npz} -> ${_ps}"
        _out="$(run "cd '${TUTORIAL}' && python3 iterate_contacts.py --topology '${_topo}' \
                 --run '${_npz}' --dataset '${SYSTEM}_dataset.bin' \
                 --out '${SYSTEM}_topology.${_ps}.json' --damp '${DAMP:-1.0}'")"
        echo "${_out}"
        if grep -q "\[CONVERGIUTO\]" <<<"${_out}"; then
            echo "[job] convergiuto alla corsa di ${_topo}"
            break
        fi
        run "cd '${TUTORIAL}' && python3 derive_prior_set.py --base '${BASE_SET:-b3stack}' \
             --set '${_ps}' --force"
        run "cd '${TUTORIAL}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
             CLASSICAL=1 PRIOR_SET='${_ps}' EQ_GAMMA=20 \
             CHECKPOINT='equilibrated_priors_${_ps}_100ps.npz' bash 04_equilibrate.sh"
        run "cd '${TUTORIAL}' && PYRESSO='${FRAMEWORK}/espresso/build/pypresso' DEVICE=cuda \
             DISABLE_ML=1 PRIOR_SET='${_ps}' GAMMA=20 CG_STEPS=${CG_STEPS:-100000} \
             CHECKPOINT='equilibrated_priors_${_ps}_100ps.npz' \
             SAMPLE_NPZ='samples_priors_${_ps}_100ps.npz' bash 05_run_espresso.sh"
        _topo="${SYSTEM}_topology.${_ps}.json"
        _npz="samples_priors_${_ps}_100ps.npz"
    done
    echo "[job] ultima corsa: ${_topo}, ${_npz}"
    ;;

analysis)
    # La g(r) vale per qualunque G-quadruplex da 1c84f30: serve solo --nuc.
    # Gira qui e non sul nodo di login perche' load_reference legge il dataset
    # con un ciclo Python per molecola -- 1.76 milioni di iterazioni sui 6788
    # frame del TEL26 -- e sfonderebbe il limite di 600 s di CPU del login.
    [[ -n "${NUC:-}" ]] || { echo "[ERROR] indica NUC=<residui per copia>" >&2; exit 2; }
    runs="${RUNS:-run=samples.npz}"
    run "cd '${TUTORIAL}' && python3 '${DIAGNOSTICS}/scripts/44_tel22_rdf.py' \
         '${DATASET}' ${runs} --nuc ${NUC} ${RDF_ARGS:-}"
    ;;

*)
    echo "[ERROR] STAGE non indicato o non riconosciuto: '${STAGE}'" >&2
    echo "        Attesi: setup | configure | build | dataset | noisefloor | train | select | production | analysis" >&2
    echo "        Sottometti con: bash hpc/submit_leonardo.sh <stadio>" >&2
    exit 2
    ;;
esac

echo "[job] stadio ${STAGE} (${SYSTEM}) concluso — $(date)"
