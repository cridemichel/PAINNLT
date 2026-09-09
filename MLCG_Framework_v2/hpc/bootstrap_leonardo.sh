#!/usr/bin/env bash
# Bootstrap del framework su Leonardo: clona ESPResSo alla versione fissata,
# innesta il plugin PaiNN, compila ESPResSo e il trainer.
#
# Va eseguito DENTRO il container, perche' serve LibTorch:
#   apptainer exec --nv --bind "$PWD:$PWD" painn.sif \
#       bash MLCG_Framework_v2/hpc/bootstrap_leonardo.sh
#
# PERCHE' ESPRESSO SI CLONA E NON SI TRASFERISCE
#   ESPResSo non e' tracciato nel repo PAINNLT: e' un clone separato di
#   espressomd/espresso, con zero file sotto controllo di versione qui.  Quindi
#   sul cluster si clona dal suo repository e si applica il plugin, invece di
#   trasferire 900 MB di albero sorgente piu' build.
#
# LA VERSIONE VA FISSATA
#   Il plugin innesta file dentro src/core/nonbonded_interactions e
#   src/python/espressomd, e tocca le liste di CMake di ESPResSo.  Una versione
#   diversa di ESPResSo puo' aver spostato quei file o cambiato le firme, e
#   l'innesto fallirebbe in modi non ovvi.  Questo e' il commit su cui il
#   plugin e' stato sviluppato e validato.
set -euo pipefail

ESPRESSO_COMMIT="${ESPRESSO_COMMIT:-84cc1d924}"   # 5.0.0-58-g84cc1d924
ESPRESSO_REPO="${ESPRESSO_REPO:-https://github.com/espressomd/espresso.git}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ESPRESSO_SRC="${ESPRESSO_SRC:-$FRAMEWORK_ROOT/espresso}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

say() { printf '\n[bootstrap] %s\n' "$*"; }

# ── 0. controlli preliminari ────────────────────────────────────────────────
say "verifica dell'ambiente"
command -v cmake >/dev/null || { echo "[ERROR] cmake assente: sei dentro il container?" >&2; exit 2; }
python3 -c 'import torch' 2>/dev/null || { echo "[ERROR] torch non importabile: sei dentro il container?" >&2; exit 2; }
TORCH_PREFIX="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
echo "  torch      $(python3 -c 'import torch;print(torch.__version__)')  in ${TORCH_PREFIX}"
echo "  CUDA vista $(python3 -c 'import torch;print(torch.cuda.is_available())')"
echo "  core       ${JOBS}"

# La disponibilita' di CUDA e' informativa: sul nodo di login puo' essere false
# anche quando sui nodi di calcolo funziona, perche' i login non hanno GPU.
# Compilare non la richiede; eseguire si'.

# ── 1. ESPResSo alla versione fissata ──────────────────────────────────────
if [[ -d "$ESPRESSO_SRC/.git" ]]; then
    say "ESPResSo gia' presente in $ESPRESSO_SRC"
    have="$(git -C "$ESPRESSO_SRC" rev-parse --short HEAD)"
    if [[ "$have" != "${ESPRESSO_COMMIT:0:${#have}}" ]]; then
        echo "  [ATTENZIONE] commit presente ${have}, atteso ${ESPRESSO_COMMIT}."
        echo "               Il plugin e' validato sul secondo.  Per allinearlo:"
        echo "                 git -C $ESPRESSO_SRC fetch && git -C $ESPRESSO_SRC checkout ${ESPRESSO_COMMIT}"
    else
        echo "  commit ${have}: corretto"
    fi
else
    say "clono ESPResSo al commit ${ESPRESSO_COMMIT}"
    git clone "$ESPRESSO_REPO" "$ESPRESSO_SRC"
    git -C "$ESPRESSO_SRC" checkout "$ESPRESSO_COMMIT"
fi

# ── 2. innesto del plugin PaiNN ────────────────────────────────────────────
say "innesto del plugin PaiNN nell'albero ESPResSo"
ESPRESSO_SRC="$ESPRESSO_SRC" PYTHON_BIN=python3 \
    bash "$FRAMEWORK_ROOT/simulation/espresso_plugin/copy_plugin_files.sh"

# ── 3. build di ESPResSo ───────────────────────────────────────────────────
say "compilo ESPResSo (${JOBS} core)"
cmake -S "$ESPRESSO_SRC" -B "$ESPRESSO_SRC/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
      -DPython_EXECUTABLE="$(command -v python3)"
cmake --build "$ESPRESSO_SRC/build" -j "$JOBS"

# ── 4. build del trainer ───────────────────────────────────────────────────
# MLCG_TORCH_ROOT e' il meccanismo del CMakeLists della v2 per selezionare UNA
# distribuzione LibTorch: passarlo evita che CMake ne trovi due e linki header
# e librerie di versioni diverse.
say "compilo il trainer PaiNN"
cmake -S "$FRAMEWORK_ROOT/training" -B "$FRAMEWORK_ROOT/training/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLCG_TORCH_ROOT="$TORCH_PREFIX"
cmake --build "$FRAMEWORK_ROOT/training/build" -j "$JOBS"

# ── 5. verifica ────────────────────────────────────────────────────────────
say "verifica"
ok=1
for f in "$ESPRESSO_SRC/build/pypresso" \
         "$ESPRESSO_SRC/build/src/python/espressomd/painn.so" \
         "$FRAMEWORK_ROOT/training/build/train_painn"; do
    if [[ -e "$f" ]]; then printf '  ok        %s\n' "$f"
    else printf '  ASSENTE   %s\n' "$f"; ok=0; fi
done

# Il plugin compilato deve conoscere i kwargs ordered-geometry, altrimenti
# equilibrate.py e run_cg_md.py li scarteranno con un WARN e un modello
# shared-geometry non sarebbe simulabile.
if [[ -e "$ESPRESSO_SRC/build/src/python/espressomd/painn.so" ]]; then
    if strings "$ESPRESSO_SRC/build/src/python/espressomd/painn.so" \
       | grep -q ordered_geometry_copies; then
        echo "  ok        painn.so include il supporto ordered-geometry"
    else
        echo "  [NOTA]    painn.so senza ordered-geometry: PaiNN puro funziona,"
        echo "            un modello shared-geometry no.  Riapplica il plugin e ricompila."
    fi
fi

(( ok )) || { echo "[ERROR] bootstrap incompleto" >&2; exit 1; }
say "bootstrap completato"
