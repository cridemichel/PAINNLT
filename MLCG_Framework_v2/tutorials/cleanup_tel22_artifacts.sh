#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="${SCRIPT_DIR}/tel22"
IBI_DIR="${SCRIPT_DIR}/tel22_IBI"

MODE="dry-run"
INCLUDE_GENERATED=0
INCLUDE_ARCHIVES=0

usage() {
    cat <<'USAGE'
Usage:
  bash tutorials/cleanup_tel22_artifacts.sh [--dry-run|--run] [--archives] [--generated]

Default behavior is a dry-run of low-risk junk cleanup in tutorials/tel22 and
 tutorials/tel22_IBI.

  --dry-run    print what would be removed (default)
  --run        actually remove selected paths
  --archives   also include known local ZIP snapshots (ibival.zip, val.zip, ms.zip)
  --generated  include regenerable CG runtime outputs only. GROMACS-generated
               files are protected by policy and are never removed by this helper;
               scientific reports, priors, datasets and models are also preserved
  -h, --help   show this help

Examples:
  bash tutorials/cleanup_tel22_artifacts.sh
  bash tutorials/cleanup_tel22_artifacts.sh --run
  bash tutorials/cleanup_tel22_artifacts.sh --dry-run --archives
  bash tutorials/cleanup_tel22_artifacts.sh --dry-run --generated
  bash tutorials/cleanup_tel22_artifacts.sh --run --generated
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) MODE="dry-run" ;;
        --run) MODE="run" ;;
        --archives) INCLUDE_ARCHIVES=1 ;;
        --generated) INCLUDE_GENERATED=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

for required in "${TEL22_DIR}" "${IBI_DIR}"; do
    if [ ! -d "${required}" ]; then
        echo "[ERROR] Expected tutorial directory not found: ${required}" >&2
        exit 1
    fi
done

removed_count=0
planned_count=0

is_safe_path() {
    case "$1" in
        "${TEL22_DIR}"/*|"${IBI_DIR}"/*) return 0 ;;
        *) return 1 ;;
    esac
}

is_protected_gromacs_path() {
    local path="$1"
    local dir name
    for dir in "${TEL22_DIR}" "${IBI_DIR}"; do
        case "${path}" in
            "${dir}"/*)
                name="${path#${dir}/}"
                case "${name}" in
                    .short_mdp|143D.pdb|pdb143d.ent.gz|tel22_clean.pdb|tel22_processed.gro|\
                    box_10.gro|box_solvated.gro|box_ions.gro|ions.tpr|posre.itp|topol.top|\
                    em.edr|em.gro|em.log|em.tpr|em.trr|nvt.cpt|nvt.edr|nvt.gro|nvt.log|nvt.tpr|\
                    npt.cpt|npt.edr|npt.gro|npt.log|npt.tpr|md.cpt|md.edr|md.gro|md.log|md.tpr|\
                    md.trr|md_whole.trr|mdout.mdp) return 0 ;;
                esac
                ;;
        esac
    done
    return 1
}

remove_path() {
    local path="$1"
    local category="$2"
    [ -e "${path}" ] || [ -L "${path}" ] || return 0
    if is_protected_gromacs_path "${path}"; then
        echo "[KEEP:gromacs-protected] ${path#${SCRIPT_DIR}/}"
        return 0
    fi
    if ! is_safe_path "${path}"; then
        echo "[ERROR] Refusing path outside TEL22 tutorials: ${path}" >&2
        exit 1
    fi
    planned_count=$((planned_count + 1))
    if [ "${MODE}" = "run" ]; then
        echo "[REMOVE:${category}] ${path#${SCRIPT_DIR}/}"
        rm -rf "${path}"
        removed_count=$((removed_count + 1))
    else
        echo "[DRY-RUN:${category}] ${path#${SCRIPT_DIR}/}"
    fi
}

cleanup_junk_dir() {
    local dir="$1"
    local path

    # Editor/patch backups and platform metadata.
    for path in \
        "${dir}/#topol.top.1#" \
        "${dir}/#topol.top.2#" \
        "${dir}/06_certify_nve.sh.pre_sigma_v2" \
        "${dir}/tel22_model.pt.manifest.json.bak" \
        "${dir}/.DS_Store"; do
        remove_path "${path}" "junk"
    done

    # Retired zero-byte placeholder. Preserve it if a local user has added content.
    path="${dir}/06c_test_selective_wca12.sh"
    if [ -f "${path}" ]; then
        if [ ! -s "${path}" ]; then
            remove_path "${path}" "retired-empty-placeholder"
        else
            echo "[KEEP] ${path#${SCRIPT_DIR}/} is non-empty; refusing automatic removal"
        fi
    fi

    # Cache directories can appear below diagnostic trees as well.
    while IFS= read -r path; do
        remove_path "${path}" "cache"
    done < <(find "${dir}" -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -print 2>/dev/null | sort)

    while IFS= read -r path; do
        remove_path "${path}" "cache"
    done < <(find "${dir}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print 2>/dev/null | sort)
}

cleanup_junk_dir "${TEL22_DIR}"
cleanup_junk_dir "${IBI_DIR}"

# The tutorial backup is low-risk junk. ZIP snapshots are opt-in because their
# content may be useful to a local user even though they are not canonical.
remove_path "${IBI_DIR}/TUTORIAL.md.orig" "junk"

if [ "${INCLUDE_ARCHIVES}" -eq 1 ]; then
    for path in \
        "${IBI_DIR}/ibival.zip" \
        "${IBI_DIR}/val.zip" \
        "${IBI_DIR}/ms.zip"; do
        remove_path "${path}" "archive"
    done
fi

if [ "${INCLUDE_GENERATED}" -eq 1 ]; then
    for dir in "${TEL22_DIR}" "${IBI_DIR}"; do
        # GROMACS working products are listed for visibility but remove_path protects them unconditionally.
        for name in \
            .short_mdp \
            143D.pdb pdb143d.ent.gz tel22_clean.pdb tel22_processed.gro \
            box_10.gro box_solvated.gro box_ions.gro ions.tpr posre.itp topol.top \
            em.edr em.gro em.log em.tpr em.trr \
            nvt.cpt nvt.edr nvt.gro nvt.log nvt.tpr \
            npt.cpt npt.edr npt.gro npt.log npt.tpr \
            md.cpt md.edr md.gro md.log md.tpr md.trr md_whole.trr mdout.mdp; do
            remove_path "${dir}/${name}" "generated-aa"
        done

        # Regenerable short runtime outputs.  Dataset, priors, model weights,
        # manifests and certification/diagnostic reports are deliberately kept.
        for name in \
            cg_training_log.csv cg_trajectory.vtf energy.csv \
            equilibrated.npz smoke_equilibrated.npz md_sanity_100.log; do
            remove_path "${dir}/${name}" "generated-cg"
        done
    done
fi

if [ "${MODE}" = "run" ]; then
    echo "[DONE] Removed ${removed_count} path(s)."
else
    echo "[DONE] Dry-run: ${planned_count} path(s) would be removed."
    echo "       Re-run with --run to delete them."
fi

if [ "${INCLUDE_ARCHIVES}" -eq 0 ]; then
    echo "[NOTE] Local ZIP snapshots were preserved. Add --archives to include them."
fi
if [ "${INCLUDE_GENERATED}" -eq 0 ]; then
    echo "[NOTE] Regenerable CG runtime outputs were preserved. GROMACS-generated files are always protected."
fi
