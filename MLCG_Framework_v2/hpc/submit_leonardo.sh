#!/usr/bin/env bash
# Sottomissione degli stadi della pipeline su Leonardo con le risorse giuste
# per ciascuno.
#
#   bash hpc/submit_leonardo.sh setup
#   bash hpc/submit_leonardo.sh configure
#   bash hpc/submit_leonardo.sh build
#   bash hpc/submit_leonardo.sh dataset AA_TRAJECTORY=/percorso/md.trr AA_TOPOLOGY=/percorso/md.gro
#   bash hpc/submit_leonardo.sh noisefloor
#   bash hpc/submit_leonardo.sh train
#   bash hpc/submit_leonardo.sh select
#   bash hpc/submit_leonardo.sh production
#
# IL SISTEMA
#   Di default si lavora su TEL22.  Per un altro G-quadruplex basta indicare
#   la sua tutorial directory, che da' anche i nomi dei file prodotti:
#
#   bash hpc/submit_leonardo.sh dataset SYSTEM=tel26 AA_TRAJECTORY=... AA_TOPOLOGY=...
#   bash hpc/submit_leonardo.sh train   SYSTEM=tel26
#
# PERCHE' UN WRAPPER E NON LE DIRETTIVE #SBATCH
#   Gli stadi hanno bisogni opposti.  Il setup vuole rete e nessuna GPU;
#   compilare vuole molti core e nessuna GPU;
#   allenare vuole una A100 e pochi core.  Un header unico con --gres=gpu:1 fa
#   pagare ore GPU per una compilazione e per ore di MDAnalysis, che sono la
#   parte piu' lunga e piu' inutile da mettere su un acceleratore.
#
# LA RETE
#   I nodi di calcolo di Leonardo non raggiungono internet; i nodi di login si.
#   La partizione lrd_all_serial gira SUI nodi di login (login08, login13),
#   quindi e' l'unico posto dove un job puo' scaricare LibTorch, i pacchetti
#   Python e il sorgente di ESPResSo: e' lo stadio setup.  Il bootstrap, che
#   solo compila, puo' andare su DCGP, che ha molti piu' core ma nessuna rete.
#
# IL LIMITE DEI 600 SECONDI
#   Sul nodo di login ogni processo viene ucciso dopo 10 minuti di CPU time
#   (ulimit -t 600).  La costruzione del .sif li supera, per questo e' un job e
#   non un comando interattivo.
set -euo pipefail

STAGE="${1:-}"
[[ -n "$STAGE" ]] || { echo "uso: $0 <setup|configure|build|dataset|noisefloor|train|select|production|analysis> [VAR=valore ...]" >&2; exit 2; }
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
ESPRESSO_SRC="${ESPRESSO_SRC:-${FRAMEWORK}/espresso}"
ESPRESSO_COMMIT="${ESPRESSO_COMMIT:-84cc1d924}"
SUBMIT="${FRAMEWORK}/hpc/leonardo_submit.slurm"

# Le assegnazioni VAR=valore passate sulla riga di comando finiscono
# nell'ambiente esportato al job.
extra_exports=()
for kv in "$@"; do
    [[ "$kv" == *=* ]] || { echo "[ERROR] argomento non riconosciuto: $kv (atteso VAR=valore)" >&2; exit 2; }
    extra_exports+=("$kv")
    # I file di input si controllano QUI, sul login, prima di occupare un nodo:
    # una variabile di shell vuota ($R non definita in una shell nuova) produce
    # percorsi come "/prod-1.part0001.xtc", e il job fallisce dopo la coda.
    case "${kv%%=*}" in
        AA_TOPOLOGY|AA_TRAJECTORY|AA_FORCES_TRAJECTORY|AA_FORCES_TOPOLOGY)
            [[ -f "${kv#*=}" ]] || { echo "[ERROR] ${kv%%=*}: file inesistente: '${kv#*=}'" >&2; exit 2; }
            ;;
    esac
done
if [[ -n "${LD_PRELOAD:-}" ]]; then
    echo "[submit] ATTENZIONE: LD_PRELOAD=${LD_PRELOAD} finira' nel job (--export=ALL)." >&2
    echo "[submit]             Se era solo per un test sul login: unset LD_PRELOAD" >&2
fi

export_list="ALL,STAGE=${STAGE},PROJECT_ROOT=${PROJECT_ROOT},FRAMEWORK=${FRAMEWORK}"
for kv in ${extra_exports[@]+"${extra_exports[@]}"}; do
    export_list+=",${kv}"
done

case "$STAGE" in

setup)
    # L'unico stadio che ha bisogno di internet: scarica LibTorch, i pacchetti
    # Python e il sorgente di ESPResSo.  Deve girare su lrd_all_serial, che sta
    # sui nodi di login: i nodi di calcolo non hanno rete.
    res=(--partition=lrd_all_serial --time=04:00:00
         --cpus-per-task=4 --mem=30G)
    if [[ -n "$ACCOUNT_SERIAL" ]]; then res+=(--account="$ACCOUNT_SERIAL"); fi
    ;;

configure)
    # La configurazione di ESPResSo scarica heFFTe, Kokkos e Cabana con
    # FetchContent, quindi vuole la rete: lrd_all_serial, come il setup.
    if [[ ! -d "$ESPRESSO_SRC/.git" ]]; then
        echo "[ERROR] ESPResSo non e' stato clonato: esegui prima" >&2
        echo "          bash hpc/submit_leonardo.sh setup" >&2
        exit 2
    fi
    res=(--partition=lrd_all_serial --time=04:00:00
         --cpus-per-task=4 --mem=30G)
    if [[ -n "$ACCOUNT_SERIAL" ]]; then res+=(--account="$ACCOUNT_SERIAL"); fi
    ;;

build|bootstrap)
    # Compilazione: molti core, nessuna GPU, nessuna rete -- ESPResSo e
    # LibTorch sono gia' stati scaricati dallo stadio setup.
    if [[ ! -d "$ESPRESSO_SRC/.git" ]]; then
        echo "[ERROR] ESPResSo non e' stato clonato: esegui prima" >&2
        echo "          bash hpc/submit_leonardo.sh setup" >&2
        exit 2
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

train|select|production|iterate)
    # Una A100 per nodo: il trainer e il driver di simulazione usano una GPU
    # sola.  Su Booster il rapporto e' 8 core per GPU.
    res=(--partition=boost_usr_prod --time=24:00:00
         --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
         --account="$ACCOUNT_GPU")
    ;;

*)
    echo "[ERROR] stadio non riconosciuto: $STAGE" >&2
    echo "        Attesi: setup | configure | build | dataset | noisefloor | train | select | production | analysis" >&2
    exit 2
    ;;
esac

# I nomi delle QOS cambiano da progetto a progetto: non se ne impone nessuna e
# si usa quella di default dell'account.  Per forzarla:  QOS=boost_qos_dbg ...
if [[ -n "${QOS:-}" ]]; then res+=(--qos="$QOS"); fi

# Gli stadi sono in sequenza: ognuno ha bisogno del precedente.  Sottometterli
# tutti insieme non funziona -- il configure partirebbe mentre il setup sta
# ancora installando, e il build prima che la configurazione esista.  AFTER
# incatena il job al precedente, che e' il modo giusto di metterli in coda
# tutti e tre senza stare a guardare:
#
#   S=$(bash hpc/submit_leonardo.sh setup     | awk "/Submitted/{print \$4}")
#   C=$(AFTER=$S bash hpc/submit_leonardo.sh configure | awk "/Submitted/{print \$4}")
#   AFTER=$C bash hpc/submit_leonardo.sh build
#
# afterok e non after: se uno stadio fallisce, i successivi non partono e
# restano in coda come DependencyNeverSatisfied, da cancellare con scancel.
if [[ -n "${AFTER:-}" ]]; then
    res+=(--dependency="afterok:${AFTER}")
    echo "[submit] parte dopo il job ${AFTER}"
fi

SYSTEM="${SYSTEM:-tel22}"
for kv in ${extra_exports[@]+"${extra_exports[@]}"}; do
    [[ "$kv" == SYSTEM=* ]] && SYSTEM="${kv#SYSTEM=}"
done
if [[ ! -d "${FRAMEWORK}/tutorials/${SYSTEM}" ]]; then
    echo "[ERROR] non esiste tutorials/${SYSTEM} sotto ${FRAMEWORK}" >&2
    echo "        sistemi disponibili: $(ls "${FRAMEWORK}/tutorials" | tr '\n' ' ')" >&2
    exit 2
fi
export_list+=",SYSTEM=${SYSTEM}"

# I log in un posto solo.  Le direttive #SBATCH --output dentro il file di job
# sono relative alla directory da cui si lancia sbatch, quindi i log finivano
# sparsi fra la radice del progetto e le tutorial directory a seconda di dove
# ci si trovava.  Qui si passano espliciti e assoluti, e vincono sulle
# direttive del file.
LOGDIR="${LOGDIR:-${PROJECT_ROOT}/logs}"
mkdir -p "$LOGDIR"
res+=(--output="${LOGDIR}/slurm-%x-%j.out" --error="${LOGDIR}/slurm-%x-%j.err")

echo "[submit] stadio   ${STAGE}"
case "$STAGE" in
    setup|configure|build|bootstrap) ;;
    *) echo "[submit] sistema  ${SYSTEM}" ;;
esac
echo "[submit] risorse  ${res[*]}"
echo "[submit] progetto ${PROJECT_ROOT}"
echo "[submit] log      ${LOGDIR}/slurm-<job-name>-<jobid>.out"
# setup, configure e build compilano il framework e non guardano il sistema:
# etichettarli con SYSTEM fa cercare i loro log sotto il nome sbagliato.
case "$STAGE" in
    setup|configure|build|bootstrap) job_name="mlcg_${STAGE}" ;;
    *)                               job_name="${SYSTEM}_${STAGE}" ;;
esac

echo "[submit] job-name ${job_name}"
sbatch "${res[@]}" --job-name="${job_name}" --export="${export_list}" "$SUBMIT"
