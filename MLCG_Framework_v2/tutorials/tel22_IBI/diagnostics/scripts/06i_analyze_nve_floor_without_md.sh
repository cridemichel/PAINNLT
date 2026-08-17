#!/usr/bin/env bash
set -euo pipefail

# TEL22_IBI post-processing-only NVE floor diagnostic.
#
# ZERO new dynamics are run by this script. It reuses completed energy series
# and certification reports from 06d/06e/06g/06h and asks whether the observed
# small-dt flattening is better described by
#
#   pure Verlet:  sigma_E = C dt^2
#
# or by a quadratic integrator contribution plus a dt-independent floor:
#
#   sigma_E^2 = sigma_0^2 + A dt^4.
#
# The fit is performed separately for each microscopic state. The three
# smallest available dt values are used for the primary floor fit so that
# coarse-dt state/phase resonances do not dominate the diagnosis.
#
# Diagnostic only: no priors, checkpoints, model files, or timestep settings
# are modified.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/model_config.sh" ]]; then
    TUTORIAL_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../../model_config.sh" ]]; then
    TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    echo "[ERROR] Cannot locate tutorials/tel22_IBI from ${SCRIPT_DIR}" >&2
    exit 1
fi
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${TUTORIAL_DIR}"

SIXH_ROOT="${SIXH_ROOT:-diagnostics/nve/nve_ibi_small_dt_multistate_scaling}"
SIXG_ROOT="${SIXG_ROOT:-diagnostics/nve/nve_ibi_dt_0p001_duration_state_certification}"
SIXD_REPORT="${SIXD_REPORT:-diagnostics/nve/nve_fine_timestep_response_promoted_ibi/nve_diagnostic_report.json}"
SIXE_ROOT="${SIXE_ROOT:-diagnostics/nve/nve_ibi_timestep_state_replicates}"
OUT_ROOT="${OUT_ROOT:-diagnostics/nve/nve_ibi_sigma_floor_analysis}"
REPORT="${OUT_ROOT}/sigma_floor_analysis_report.json"

# 06h used 0.999 ps exactly. A complete 06h energy series must span this
# duration; partial files left by Ctrl-C are ignored.
SIXH_DURATION_PS="${SIXH_DURATION_PS:-0.999}"

# Evidence thresholds. These are diagnostic rather than production gates.
FLOOR_R2_MIN="${FLOOR_R2_MIN:-0.98}"
FLOOR_RMSE_IMPROVEMENT_MIN="${FLOOR_RMSE_IMPROVEMENT_MIN:-2.0}"
MAX_PRIMARY_DT="${MAX_PRIMARY_DT:-0.0015}"

usage() {
    cat <<'USAGE'
Usage:
  06i_analyze_nve_floor_without_md.sh [--dry-run | --overwrite]

Post-processing only. Runs ZERO MD steps.

It collects existing TEL22_IBI NVE points from:
  06h small-dt trajectories (preferred when complete),
  06g dt=0.001 trajectories (0.999-ps prefix fallback),
  06d reference-state fine-dt report,
  06e branch_B/branch_C state-replicate reports.

For each state it compares:
  sigma_E = C dt^2
against
  sigma_E^2 = sigma_0^2 + A dt^4
using the three smallest available dt values (primary fit), while also
reporting a wider <=0.002-ps sensitivity fit when enough points exist.

Useful overrides:
  FLOOR_R2_MIN=0.98
  FLOOR_RMSE_IMPROVEMENT_MIN=2.0
  MAX_PRIMARY_DT=0.0015
USAGE
}

MODE="normal"
case "${1:-}" in
    "") ;;
    --dry-run) MODE="dry-run"; shift ;;
    --overwrite) MODE="overwrite"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
if (($# != 0)); then
    echo "[ERROR] Unexpected extra arguments: $*" >&2
    exit 2
fi

[[ -f "${FRAMEWORK_ROOT}/simulation/nve_analysis.py" ]] || {
    echo "[ERROR] Missing ${FRAMEWORK_ROOT}/simulation/nve_analysis.py" >&2
    exit 1
}

cat <<EOF_PLAN
[TEL22_IBI NVE SIGMA FLOOR ANALYSIS -- POST-PROCESSING ONLY]
new MD steps               : 0
06h root                   : ${SIXH_ROOT}
06g root                   : ${SIXG_ROOT}
06d reference report       : ${SIXD_REPORT}
06e state-replicate root   : ${SIXE_ROOT}
primary fit                : three smallest available dt <= ${MAX_PRIMARY_DT} ps
floor model                : sigma_E^2 = sigma_0^2 + A dt^4
pure-Verlet comparison     : sigma_E = C dt^2
floor R2 threshold         : >= ${FLOOR_R2_MIN}
RMSE improvement threshold : >= ${FLOOR_RMSE_IMPROVEMENT_MIN}x
output                     : ${REPORT}
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    echo "[PLAN] Existing files only; incomplete 06h trajectories will be ignored."
    echo "[PLAN] No pypresso/certify_nve call and no checkpoint write will occur."
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ROOT}"
elif [[ -e "${REPORT}" ]]; then
    echo "[ERROR] Report already exists: ${REPORT}" >&2
    echo "        Use --overwrite to regenerate post-processing." >&2
    exit 1
fi
mkdir -p "${OUT_ROOT}"

"${PYTHON_BIN}" - \
    "${FRAMEWORK_ROOT}" \
    "${SIXH_ROOT}" \
    "${SIXG_ROOT}" \
    "${SIXD_REPORT}" \
    "${SIXE_ROOT}" \
    "${REPORT}" \
    "${SIXH_DURATION_PS}" \
    "${FLOOR_R2_MIN}" \
    "${FLOOR_RMSE_IMPROVEMENT_MIN}" \
    "${MAX_PRIMARY_DT}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np

framework_root = Path(sys.argv[1]).resolve()
sixh_root = Path(sys.argv[2]).resolve()
sixg_root = Path(sys.argv[3]).resolve()
sixd_report = Path(sys.argv[4]).resolve()
sixe_root = Path(sys.argv[5]).resolve()
out_path = Path(sys.argv[6]).resolve()
sixh_duration = float(sys.argv[7])
floor_r2_min = float(sys.argv[8])
rmse_improvement_min = float(sys.argv[9])
max_primary_dt = float(sys.argv[10])

sys.path.insert(0, str(framework_root / "simulation"))
from nve_analysis import analyze_energy_series, read_energy_csv

branches = ("reference", "branch_B", "branch_C")


def close(a, b, rel=1.0e-9, abs_=1.0e-12):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def dt_from_tag(name):
    if not name.startswith("dt_"):
        return None
    token = name[3:].replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def point(dt, sigma, drift, source, duration=None, priority=0):
    return {
        "dt_ps": float(dt),
        "sigma_E": float(sigma),
        "relative_block_mean_drift": float(drift),
        "source": str(source),
        "duration_ps": None if duration is None else float(duration),
        "priority": int(priority),
    }


def load_report_runs(path, source_label, priority):
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = data.get("runs", [])
    except Exception as exc:
        print(f"[SKIP] unreadable report {path}: {exc}")
        return []
    out = []
    for row in runs:
        try:
            dt = float(row["dt_ps"])
            sigma = float(row["sigma_E"])
            drift = float(row["relative_block_mean_drift"])
        except (KeyError, TypeError, ValueError):
            continue
        if dt > 0.0 and sigma > 0.0 and math.isfinite(sigma):
            out.append(point(dt, sigma, drift, f"{source_label}:{path}", priority=priority))
    return out


def load_complete_06h(branch):
    base = sixh_root / branch
    if not base.is_dir():
        return []
    out = []
    for energy_path in sorted(base.glob("dt_*/energy.csv")):
        dt = dt_from_tag(energy_path.parent.name)
        if dt is None or dt <= 0.0:
            continue
        try:
            t, e = read_energy_csv(energy_path)
        except Exception as exc:
            print(f"[SKIP] {branch} {energy_path}: unreadable ({exc})")
            continue
        if t.size < 3:
            print(f"[SKIP] {branch} {energy_path}: fewer than 3 samples")
            continue
        actual = float(t[-1] - t[0])
        expected_steps = int(round(sixh_duration / dt))
        expected_duration = expected_steps * dt
        expected_samples = expected_steps + 1
        tol = max(1.0e-10, 1.0e-8 * max(1.0, expected_duration))
        if t.size != expected_samples or abs(actual - expected_duration) > tol:
            print(
                f"[SKIP] {branch} dt={dt:g}: incomplete 06h series "
                f"samples={t.size}/{expected_samples} duration={actual:.9g}/{expected_duration:.9g} ps"
            )
            continue
        m = dict(analyze_energy_series(t, e))
        out.append(
            point(
                dt,
                m["sigma_E"],
                m["relative_block_mean_drift"],
                energy_path,
                duration=m.get("duration_ps", actual),
                priority=100,
            )
        )
    return out


def load_06g_prefix(branch):
    energy_path = sixg_root / branch / "energy.csv"
    if not energy_path.is_file():
        return []
    dt = 0.001
    steps = int(round(sixh_duration / dt))
    needed = steps + 1
    expected_duration = steps * dt
    try:
        t, e = read_energy_csv(energy_path)
    except Exception as exc:
        print(f"[SKIP] {branch} 06g prefix: unreadable ({exc})")
        return []
    if t.size < needed:
        print(f"[SKIP] {branch} 06g prefix: only {t.size} samples, need {needed}")
        return []
    tp, ep = t[:needed], e[:needed]
    actual = float(tp[-1] - tp[0])
    if not close(actual, expected_duration, rel=1.0e-8, abs_=1.0e-10):
        print(f"[SKIP] {branch} 06g prefix: duration={actual:g}, expected={expected_duration:g}")
        return []
    m = dict(analyze_energy_series(tp, ep))
    return [
        point(
            dt,
            m["sigma_E"],
            m["relative_block_mean_drift"],
            f"06g-prefix:{energy_path}",
            duration=m.get("duration_ps", actual),
            priority=90,
        )
    ]


def dedupe(points):
    # Prefer completed 06h, then 06g exact prefix, then older 06d/06e reports.
    chosen = {}
    for row in points:
        key = round(float(row["dt_ps"]), 12)
        old = chosen.get(key)
        if old is None or int(row["priority"]) > int(old["priority"]):
            chosen[key] = row
    return sorted(chosen.values(), key=lambda r: float(r["dt_ps"]))


def fit_power(rows):
    dt = np.array([r["dt_ps"] for r in rows], dtype=float)
    sigma = np.array([r["sigma_E"] for r in rows], dtype=float)
    x = np.log(dt)
    y = np.log(sigma)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if sst <= np.finfo(float).eps else 1.0 - sse / sst
    return {
        "p": float(beta[1]),
        "C": float(math.exp(beta[0])),
        "loglog_r2": r2,
    }


def rmse_relative(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.maximum(np.abs(obs), np.finfo(float).tiny)
    return float(np.sqrt(np.mean(((pred - obs) / denom) ** 2)))


def fit_models(rows):
    if len(rows) < 3:
        return {"available": False, "reason": "need_at_least_3_points"}
    dt = np.array([r["dt_ps"] for r in rows], dtype=float)
    sigma = np.array([r["sigma_E"] for r in rows], dtype=float)

    # Pure order-2 model in sigma space: sigma = C dt^2.
    q = dt ** 2
    c = float(np.dot(q, sigma) / np.dot(q, q))
    pred_quad = c * q
    rmse_quad = rmse_relative(sigma, pred_quad)

    # Floor model is linear in variance: sigma^2 = b + A dt^4.
    x = dt ** 4
    y = sigma ** 2
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b = float(beta[0])
    A = float(beta[1])
    physical = b >= 0.0 and A > 0.0
    if physical:
        pred_var = b + A * x
        pred_sigma = np.sqrt(np.maximum(pred_var, 0.0))
        sigma0 = math.sqrt(b)
        crossover = (b / A) ** 0.25 if b > 0.0 else 0.0
        rmse_floor = rmse_relative(sigma, pred_sigma)
        sse = float(np.sum((y - pred_var) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 if sst <= np.finfo(float).eps else 1.0 - sse / sst
        improvement = math.inf if rmse_floor <= np.finfo(float).tiny else rmse_quad / rmse_floor
    else:
        pred_sigma = np.full_like(sigma, np.nan)
        sigma0 = math.nan
        crossover = math.nan
        rmse_floor = math.inf
        r2 = -math.inf
        improvement = 0.0

    power = fit_power(rows)
    supported = bool(
        physical
        and r2 >= floor_r2_min
        and improvement >= rmse_improvement_min
    )
    return {
        "available": True,
        "n_points": len(rows),
        "dt_ps": dt.tolist(),
        "sigma_E": sigma.tolist(),
        "free_power_fit": power,
        "pure_quadratic": {
            "model": "sigma_E = C * dt^2",
            "C": c,
            "relative_rmse": rmse_quad,
        },
        "floor_model": {
            "model": "sigma_E^2 = sigma_0^2 + A * dt^4",
            "sigma0": sigma0,
            "sigma0_squared": b,
            "A": A,
            "physical_nonnegative": physical,
            "variance_space_r2": r2,
            "relative_rmse": rmse_floor,
            "rmse_improvement_vs_pure_quadratic": improvement,
            "crossover_dt_ps": crossover,
            "supported": supported,
            "thresholds": {
                "variance_space_r2_min": floor_r2_min,
                "rmse_improvement_min": rmse_improvement_min,
            },
        },
    }


all_points = {name: [] for name in branches}

# Highest-priority small-dt data, including completed trajectories left by a
# partially interrupted 06h invocation.
for name in branches:
    all_points[name].extend(load_complete_06h(name))
    all_points[name].extend(load_06g_prefix(name))

# Older completed grids provide additional context and allow a diagnostic fit
# even if 06h was intentionally stopped after the reference branch.
all_points["reference"].extend(load_report_runs(sixd_report, "06d", priority=30))
for name in ("branch_B", "branch_C"):
    path = sixe_root / name / "nve" / "nve_certification_report.json"
    all_points[name].extend(load_report_runs(path, "06e", priority=30))

results = {}
for name in branches:
    rows = dedupe(all_points[name])
    small_candidates = [r for r in rows if float(r["dt_ps"]) <= max_primary_dt + 1e-15]
    primary = small_candidates[:3]
    sensitivity = [r for r in rows if float(r["dt_ps"]) <= 0.002 + 1e-15]
    results[name] = {
        "available_points": rows,
        "primary_rows": primary,
        "primary_fit": fit_models(primary),
        "sensitivity_le_0p002_fit": fit_models(sensitivity) if len(sensitivity) >= 3 else {
            "available": False,
            "reason": "need_at_least_3_points",
        },
    }

supporting = [
    name for name in branches
    if results[name]["primary_fit"].get("available")
    and results[name]["primary_fit"].get("floor_model", {}).get("supported", False)
]
primary_available = [name for name in branches if results[name]["primary_fit"].get("available")]

if len(supporting) >= 2:
    classification = "floor_supported_multistate__no_more_md_required_for_floor_diagnosis"
elif "reference" in supporting:
    classification = "floor_supported_reference__cross_state_floor_evidence_incomplete"
elif supporting:
    classification = "floor_supported_in_one_nonreference_state__mixed_evidence"
else:
    classification = "floor_model_not_established_from_existing_data"

report = {
    "schema_version": 1,
    "purpose": "postprocess_existing_nve_sigma_floor_without_new_md",
    "new_md_steps": 0,
    "models": {
        "pure_verlet": "sigma_E = C * dt^2",
        "floor_plus_verlet": "sigma_E^2 = sigma_0^2 + A * dt^4",
    },
    "primary_fit_policy": (
        f"three smallest available dt values not exceeding {max_primary_dt:g} ps; "
        "completed 06h data preferred, then 06g prefix, then 06d/06e reports"
    ),
    "branches": results,
    "summary": {
        "states_with_primary_fit": primary_available,
        "states_supporting_floor": supporting,
        "n_supporting_floor": len(supporting),
        "classification": classification,
    },
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print("\n[TEL22_IBI NVE SIGMA FLOOR SUMMARY]")
for name in branches:
    entry = results[name]
    available = entry["available_points"]
    print(f"[DATA] {name}: " + ", ".join(
        f"dt={r['dt_ps']:.6g}:sigma={r['sigma_E']:.6g}[{Path(str(r['source']).split(':')[-1]).name}]"
        for r in available
    ))
    fit = entry["primary_fit"]
    if not fit.get("available"):
        print(f"[FIT] {name}: unavailable ({fit.get('reason')})")
        continue
    floor = fit["floor_model"]
    quad = fit["pure_quadratic"]
    power = fit["free_power_fit"]
    print(
        f"[FIT] {name}: dt={','.join(f'{v:.6g}' for v in fit['dt_ps'])} "
        f"free_p={power['p']:.6f} R2={power['loglog_r2']:.6f}"
    )
    print(
        f"[FLOOR] {name}: sigma0={floor['sigma0']:.6g} A={floor['A']:.6g} "
        f"R2var={floor['variance_space_r2']:.6f} crossover_dt={floor['crossover_dt_ps']:.6g} "
        f"relRMSE={floor['relative_rmse']:.3e} "
        f"quad_relRMSE={quad['relative_rmse']:.3e} "
        f"improvement={floor['rmse_improvement_vs_pure_quadratic']:.3g}x "
        f"supported={floor['supported']}"
    )

print(f"[CLASSIFY] states_supporting_floor={len(supporting)}/3 " + ",".join(supporting))
print(f"[CLASSIFY] {classification}")
print(f"[REPORT] {out_path}")
print("[DONE] Post-processing only: zero new MD steps were run.")
PY
