#!/usr/bin/env bash
# Sottomissione degli stadi della pipeline su Leonardo con le risorse giuste
# per ciascuno.
#
#   bash hpc/submit_leonardo.sh image
#   bash hpc/submit_leonardo.sh bootstrap
#   bash hpc/submit_leonardo.sh dataset AA_TRAJECTORY=/percorso/md.trr AA_TOPOLOGY=/percorso/md.gro
#   bash hpc/submit_leonardo.sh noisefloor
#   bash hpc/submit_leonardo.sh train
#   bash hpc/submit_leonardo.sh select
#   bash hpc/submit_leonardo.sh production
#
# PERCHE' UN WRAPPER E NON LE DIRETTIVE #SBATCH
#   Gli stadi hanno bisogni opposti.  Costruire l'immagine vuole rete verso
#   Docker Hub e nessuna GPU; compilare vuole molti core e nessuna GPU;
#   allenare vuole una A100 e pochi core.  Un header unico con --gres=gpu:1 fa
#   pagare ore GPU per una compilazione e per ore di MDAnalysis, che sono la
#   parte piu' lunga e piu' inutile da mettere su un acceleratore.
#
# LA RETE
#   I nodi di calcolo di Leonardo non raggiungono internet; i nodi di login si.
#   La partizione lrd_all_serial gira SUI nodi di login (login08, login13),
#   quindi e' l'unico posto dove un job puo' scaricare l'immagine base.  Per lo
#   stesso motivo il clone di ESPResSo lo fa questo script, qui sul nodo di
#   login, prima di sottomettere il bootstrap: cosi' il job di compilazione puo'
#   girare su DCGP, che ha molti piu' core ma nessuna rete.
#
# IL LIMITE DEI 600 SECONDI
#   Sul nodo di login ogni processo viene ucciso dopo 10 minuti di CPU time
#   (ulimit -t 600).  La costruzione del .sif li supera, per questo e' un job e
#   non un comando interattivo.
set -euo pipefail

STAGE="${1:-}"
[[ -n "$STAGE" ]] || { echo "uso: $0 <image|bootstrap|dataset|noisefloor|train|select|production|analysis> [VAR=valore ...]" >&2; exit 2; }
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$FRAMEWORK/.." && pwd)}"

# ── account e aree ──────────────────────────────────────────────────────────
# Su Leonardo il progetto ISCRA B ha due associazioni: quella Booster (GPU) e
# quella con suffisso _0 sulla partizione DCGP (CPU).  Gli stadi senza GPU
# vanno sulla seconda, che non consuma il budget di acceleratori.
ACCOUNT_GPU="${ACCOUNT_GPU:-IscrB_G4MES}"
ACCOUNT_CPU="${ACCOUNT_CPU:-IscrB_G4MES_0}"
# lrd_all_serial non accetta necessariamente le stesse associazioni delle
# partizioni di calcolo: l'account _0 e' legato a DCGP e viene rifiutato con
# "Invalid account or account/partition combination".  Lasciato vuoto, nessun
# --account viene passato e vale quello di default dell'utente.
ACCOUNT_SERIAL="${ACCOUNT_SERIAL-}"
IMAGE="${IMAGE:-${PROJECT_ROOT}/painn.sif}"
DEFFILE="${FRAMEWORK}/hpc/painn_leonardo.def"
ESPRESSO_SRC="${ESPRESSO_SRC:-${FRAMEWORK}/espresso}"
ESPRESSO_COMMIT="${ESPRESSO_COMMIT:-84cc1d924}"
SUBMIT="${FRAMEWORK}/hpc/leonardo_submit.slurm"

# Le assegnazioni VAR=valore passate sulla riga di comando finiscono
# nell'ambiente esportato al job.
extra_exports=()
for kv in "$@"; do
    [[ "$kv" == *=* ]] || { echo "[ERROR] argomento non riconosciuto: $kv (atteso VAR=valore)" >&2; exit 2; }
    extra_exports+=("$kv")
done

export_list="ALL,STAGE=${STAGE},PROJECT_ROOT=${PROJECT_ROOT},IMAGE=${IMAGE},FRAMEWORK=${FRAMEWORK}"
for kv in ${extra_exports[@]+"${extra_exports[@]}"}; do
    export_list+=",${kv}"
done

case "$STAGE" in

image)
    # L'unico stadio che ha bisogno di internet: scarica l'immagine PyTorch da
    # Docker Hub e la converte in SIF.  Deve girare su lrd_all_serial.
    if [[ -e "$IMAGE" ]]; then echo "[INFO] immagine gia' presente: $IMAGE"; exit 0; fi
    res=(--partition=lrd_all_serial --time=04:00:00
         --cpus-per-task=4 --mem=30G)
    if [[ -n "$ACCOUNT_SERIAL" ]]; then res+=(--account="$ACCOUNT_SERIAL"); fi
    ;;

bootstrap)
    # Compilazione: molti core, nessuna GPU, nessuna rete richiesta -- a patto
    # che ESPResSo sia gia' stato clonato.  Lo facciamo adesso, sul nodo di
    # login, perche' il job girera' dove la rete non c'e'.
    if [[ ! -d "$ESPRESSO_SRC/.git" ]]; then
        echo "[pre] clono ESPResSo al commit ${ESPRESSO_COMMIT} (serve la rete del nodo di login)"
        git clone https://github.com/espressomd/espresso.git "$ESPRESSO_SRC"
        git -C "$ESPRESSO_SRC" checkout "$ESPRESSO_COMMIT"
    else
        echo "[pre] ESPResSo gia' presente in $ESPRESSO_SRC ($(git -C "$ESPRESSO_SRC" rev-parse --short HEAD))"
    fi
    res=(--partition=dcgp_usr_prod --time=02:00:00
         --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=100G
         --account="$ACCOUNT_CPU")
    export_list+=",JOBS=32"
    ;;

dataset|noisefloor|analysis)
    # MDAnalysis e numpy: CPU e memoria, nessuna GPU.  La lettura di una
    # traiettoria lunga e' il passo piu' lento dell'intera pipeline.
    res=(--partition=dcgp_usr_prod --time=08:00:00
         --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=200G
         --account="$ACCOUNT_CPU")
    ;;

train|select|production)
    # Una A100 per nodo: il trainer e il driver di simulazione usano una GPU
    # sola.  Su Booster il rapporto e' 8 core per GPU.
    res=(--partition=boost_usr_prod --time=24:00:00
         --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
         --account="$ACCOUNT_GPU")
    ;;

*)
    echo "[ERROR] stadio non riconosciuto: $STAGE" >&2
    exit 2
    ;;
esac

# I nomi delle QOS cambiano da progetto a progetto: non se ne impone nessuna e
# si usa quella di default dell'account.  Per forzarla:  QOS=boost_qos_dbg ...
if [[ -n "${QOS:-}" ]]; then res+=(--qos="$QOS"); fi

echo "[submit] stadio   ${STAGE}"
echo "[submit] risorse  ${res[*]}"
echo "[submit] progetto ${PROJECT_ROOT}"
echo "[submit] immagine ${IMAGE}"
sbatch "${res[@]}" --job-name="mlcg_${STAGE}" --export="${export_list}" "$SUBMIT"
