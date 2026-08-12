#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_SELF_FULL_RERUN_DIR:-dna_self_full_reruns}"
TARGET_FRAMES="${DNA_SELF_FULL_TARGET_FRAMES:-0}"
REUSE="${DNA_SELF_FULL_REUSE:-1}"
KEEP_INPUTS="${DNA_SELF_FULL_KEEP_INPUTS:-0}"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in md.tpr md.gro md.trr analyze_dna_self_vs_intercopy.py analyze_force_source_decomposition.py; do
    require_file "$f"
done
command -v gmx >/dev/null 2>&1 || { echo "[ERROR] gmx non trovato nel PATH." >&2; exit 1; }
mkdir -p "$OUTDIR"

case "$TARGET_FRAMES" in ''|*[!0-9]*) echo "[ERROR] DNA_SELF_FULL_TARGET_FRAMES deve essere intero >=0" >&2; exit 2;; esac
case "$REUSE" in 0|1) ;; *) echo "[ERROR] DNA_SELF_FULL_REUSE deve essere 0 o 1" >&2; exit 2;; esac
case "$KEEP_INPUTS" in 0|1) ;; *) echo "[ERROR] DNA_SELF_FULL_KEEP_INPUTS deve essere 0 o 1" >&2; exit 2;; esac

"$PYTHON_BIN" analyze_dna_self_vs_intercopy.py --make-index --topology md.gro \
  --index-output "$OUTDIR/copy_seed.ndx" --index-manifest "$OUTDIR/copy_groups.json" \
  > "$OUTDIR/index_builder.log"
printf 'q\n' | gmx make_ndx -f md.tpr -n "$OUTDIR/copy_seed.ndx" -o "$OUTDIR/copy_groups.ndx" \
  > "$OUTDIR/make_ndx.log" 2>&1

NFRAMES=$($PYTHON_BIN - <<'PY'
import MDAnalysis as mda
u=mda.Universe('md.gro','md.trr')
print(len(u.trajectory))
PY
)
if [ "$NFRAMES" -lt 2 ]; then
    echo "[ERROR] md.trr contiene meno di due frame" >&2
    exit 2
fi
if [ "$TARGET_FRAMES" -eq 0 ] || [ "$TARGET_FRAMES" -ge "$NFRAMES" ]; then
    SKIP=1
    EFFECTIVE=$NFRAMES
else
    if [ "$TARGET_FRAMES" -lt 2 ]; then TARGET_FRAMES=2; fi
    SKIP=$(( (NFRAMES - 1) / (TARGET_FRAMES - 1) ))
    [ "$SKIP" -ge 1 ] || SKIP=1
    EFFECTIVE=$(( (NFRAMES - 1) / SKIP + 1 ))
fi

SAMPLING_MATCH=0
if [ -f "$OUTDIR/subset_skip.txt" ]; then
    OLD_SKIP=$(tr -d '[:space:]' < "$OUTDIR/subset_skip.txt")
    if [ "$OLD_SKIP" = "$SKIP" ]; then SAMPLING_MATCH=1; fi
fi

NCOPIES=$($PYTHON_BIN - "$OUTDIR/copy_groups.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['copies']))
PY
)

all_reusable=1
if [ "$REUSE" -eq 1 ] && [ "$SAMPLING_MATCH" -eq 1 ]; then
    for ((ci=0; ci<NCOPIES; ci++)); do
        tag=$(printf 'copy_%02d' "$ci")
        if [ ! -s "$OUTDIR/${tag}.gro" ] || [ ! -s "$OUTDIR/${tag}_rerun.trr" ]; then
            all_reusable=0
            break
        fi
    done
else
    all_reusable=0
fi

printf '%s\n' "$SKIP" > "$OUTDIR/subset_skip.txt"
"$PYTHON_BIN" - "$OUTDIR/sampling.json" "$NFRAMES" "$SKIP" "$EFFECTIVE" <<'PY'
import json,sys
out,n,skip,effective=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
json.dump({"source_frames":n,"skip":skip,"expected_sampled_frames":effective},open(out,"w"),indent=2)
open(out,"a").write("\n")
PY

printf '%s\n' \
  "======================================================" \
  " 03n-prep. TEL22 FULL SINGLE-COPY SELF RERUNS" \
  "======================================================" \
  "md.trr frames=$NFRAMES | skip=$SKIP | expected sampled=$EFFECTIVE" \
  "copies=$NCOPIES | output=$OUTDIR" \
  "Only single-copy reruns are generated: no DNA-all rerun." \
  ""

if [ "$all_reusable" -eq 1 ]; then
    echo "[INFO] Reusing existing single-copy reruns in $OUTDIR"
else
    for ((ci=0; ci<NCOPIES; ci++)); do
        tag=$(printf 'copy_%02d' "$ci")
        gname=$(printf 'FS_COPY_%02d' "$ci")
        gid=$(awk -v name="$gname" '$1 ~ /^[0-9]+$/ && $2 == name {print $1; exit}' "$OUTDIR/make_ndx.log")
        [ -n "$gid" ] || { echo "[ERROR] Gruppo $gname non trovato" >&2; exit 1; }
        echo "[INFO] Rerun self $tag (group $gid)"
        printf '%s\n' "$gid" | gmx convert-tpr -s md.tpr -n "$OUTDIR/copy_groups.ndx" -o "$OUTDIR/${tag}.tpr" \
          > "$OUTDIR/convert_tpr_${tag}.log" 2>&1
        printf '%s\n' "$gid" | gmx trjconv -s md.tpr -f md.gro -n "$OUTDIR/copy_groups.ndx" -o "$OUTDIR/${tag}.gro" \
          > "$OUTDIR/trjconv_${tag}_topology.log" 2>&1
        printf '%s\n' "$gid" | gmx trjconv -s md.tpr -f md.trr -n "$OUTDIR/copy_groups.ndx" \
          -o "$OUTDIR/${tag}_input.trr" -skip "$SKIP" -force > "$OUTDIR/trjconv_${tag}_input.log" 2>&1
        (
          cd "$OUTDIR"
          gmx mdrun -s "${tag}.tpr" -rerun "${tag}_input.trr" -deffnm "${tag}_rerun" \
            > "mdrun_${tag}.log" 2>&1
        )
        require_file "$OUTDIR/${tag}_rerun.trr"
        if [ "$KEEP_INPUTS" -eq 0 ]; then
            rm -f "$OUTDIR/${tag}.tpr" "$OUTDIR/${tag}_input.trr"
        fi
    done
fi

"$PYTHON_BIN" - "$OUTDIR" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
import analyze_dna_self_vs_intercopy as dsi
out=Path(sys.argv[1])
manifest=json.loads((out/'copy_groups.json').read_text())
ref_times=None
counts=[]
for ci in range(int(manifest['copies'])):
    tag=f'copy_{ci:02d}'
    times,f,t,sig=dsi.load_targets(out/f'{tag}.gro',out/f'{tag}_rerun.trr')
    times=np.asarray(times)
    if ref_times is None:
        ref_times=times
    elif len(times)!=len(ref_times) or not np.allclose(times,ref_times,atol=1e-4,rtol=0.0):
        raise SystemExit(f'{tag}: rerun times differ from copy_00')
    counts.append(int(len(times)))
report={
    'copies':int(manifest['copies']),
    'residues_per_copy':int(manifest['residues_per_copy']),
    'frames_per_copy':counts,
    'times_ps':[float(x) for x in ref_times],
    'purpose':'full-resolution single-copy atomistic self targets for learning-curve diagnostics',
}
(out/'full_self_rerun_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"[VERIFY] copies={report['copies']} frames/copy={counts[0]} time={ref_times[0]:.3f}..{ref_times[-1]:.3f} ps")
PY

echo "[DONE] $OUTDIR/full_self_rerun_report.json"
