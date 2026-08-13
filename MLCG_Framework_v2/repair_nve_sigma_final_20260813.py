#!/usr/bin/env python3
"""Deterministic final repair for the NVE sigma(E) certification refactor.

Targets the source state observed on 2026-08-13:
  * certify_nve.py already passes --log_interval 1 to run_cg_md.py,
    but still computes/prints legacy log_steps metadata;
  * sigma_E is computed, but [RESULT] prints rms_delta_E;
  * report text still describes RMS(E-E0);
  * nve_analysis.certify_metrics may still fit rms_delta_E.

The repair:
  1. makes every-step sampling explicit as `log_every = 1`;
  2. uses that variable consistently in command, plan, and per-run metadata;
  3. prints sigma_E correctly;
  4. documents sigma_E as the primary scaling observable;
  5. adds sigma_E to the CSV table while retaining rms_delta_E diagnostics;
  6. changes only certify_metrics() in nve_analysis.py from rms_delta_E
     to sigma_E, leaving analyze_energy_series() and legacy diagnostics intact;
  7. validates Python syntax before writing.

The script is strict and idempotent. It creates one-time backups.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

ROOT = Path.cwd()
CERT = ROOT / "simulation" / "certify_nve.py"
ANALYSIS = ROOT / "simulation" / "nve_analysis.py"

CERT_BAK = CERT.with_name(CERT.name + ".pre_final_sigma_repair")
ANALYSIS_BAK = ANALYSIS.with_name(ANALYSIS.name + ".pre_final_sigma_repair")

EXACT_LOG_LINE = (
    "        log_every = 1  # NVE certification: sample energy every integration step"
)


def fail(msg: str) -> "NoReturn":
    raise SystemExit(f"[ERROR] {msg}")


def backup_once(path: Path, backup: Path) -> None:
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup}")


def replace_once(text: str, old: str, new: str, label: str, *, allow_new: bool = True) -> str:
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1)
    if count == 0 and allow_new and new in text:
        return text
    if count == 0:
        fail(f"could not locate expected source for {label}")
    fail(f"expected one occurrence for {label}, found {count}")


def patch_certifier(text: str) -> str:
    # Remove the known misplaced line from the faulty v2 patcher, wherever it landed.
    bad_markers = (
        "NVE certification: sample energy every integration step",
        "NVE certification: sample total energy every integration step",
    )
    cleaned = []
    for line in text.splitlines():
        if "log_every = 1" in line and any(marker in line for marker in bad_markers):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned) + "\n"

    # Source must be valid again before deterministic edits.
    try:
        ast.parse(text, filename=str(CERT))
    except SyntaxError as exc:
        fail(
            f"{CERT} is still syntactically invalid after removing the faulty override: "
            f"line {exc.lineno}: {exc.msg}"
        )

    # Current observed source: log_steps is metadata only; the runner already gets "1".
    old_log = "        log_steps = max(1, int(round(args.log_interval_ps / dt)))"
    if old_log in text:
        text = text.replace(old_log, EXACT_LOG_LINE, 1)
    elif EXACT_LOG_LINE not in text:
        fail("could not locate the observed log_steps assignment")

    text = replace_once(
        text,
        '            "--log_interval", "1",',
        '            "--log_interval", str(log_every),',
        "runner --log_interval value",
    )

    # Make plan and metrics metadata consistent with the real every-step sampling.
    text = text.replace('"log_interval_steps": log_steps', '"log_interval_steps": log_every')
    text = text.replace('f"log_every={log_steps} steps"', 'f"log_every={log_every} steps"')

    # Correct the actual terminal observable.
    text = replace_once(
        text,
        """            f"[RESULT] dt={dt:g} sigma_E={metrics['rms_delta_E']:.6g} " """.rstrip(),
        """            f"[RESULT] dt={dt:g} sigma_E={metrics['sigma_E']:.6g} " """.rstrip(),
        "[RESULT] sigma_E value",
    )

    # Correct the CLI description/help text.
    text = text.replace(
        '"multiple Velocity-Verlet time steps and fitting RMS energy error ~ dt^p."',
        '"multiple Velocity-Verlet time steps and fitting sigma_E = std(E_total) ~ dt^p."',
    )
    text = text.replace(
        'help="Time steps in ps (default: 0.002 0.001 0.0005 0.00025)"',
        'help="Time steps in ps (default: 0.001 0.002 0.005 0.01)"',
    )
    text = text.replace(
        'parser.add_argument("--log-interval-ps", type=float, default=0.01, help="Energy sampling interval")',
        'parser.add_argument("--log-interval-ps", type=float, default=0.01, '
        'help="Deprecated compatibility option; NVE certification samples energy every integration step")',
    )

    # Report definition: primary metric is std(E), RMS(E-E0) remains per-run diagnostic.
    text = text.replace(
        '"quantity": "RMS of E_total(t)-E_total(0) over fixed physical duration",',
        '"quantity": "Population standard deviation sigma_E = std(E_total) over fixed physical duration",',
    )
    text = text.replace(
        '"scaling_model": "RMS_dE = C * dt^p",',
        '"scaling_model": "sigma_E = C * dt^p",',
    )
    if '"secondary_diagnostic"' not in text:
        text = text.replace(
            '"scaling_model": "sigma_E = C * dt^p",',
            '"scaling_model": "sigma_E = C * dt^p",\n'
            '            "secondary_diagnostic": "rms_delta_E = RMS(E_total(t)-E_total(0))",',
            1,
        )
    if '"energy_sampling"' not in text:
        text = text.replace(
            '"force_cap": 0.0,',
            '"force_cap": 0.0,\n'
            '            "energy_sampling": "every integration step",',
            1,
        )

    # Add sigma_E to CSV output but retain legacy RMS diagnostic.
    if '"sigma_E",' not in text:
        text = text.replace(
            '"dt_ps", "steps", "duration_ps", "samples", "rms_delta_E",',
            '"dt_ps", "steps", "duration_ps", "samples", "sigma_E", "rms_delta_E",',
            1,
        )
    if '"sigma_E": item["sigma_E"],' not in text:
        text = text.replace(
            '                "samples": item["samples"],\n'
            '                "rms_delta_E": item["rms_delta_E"],',
            '                "samples": item["samples"],\n'
            '                "sigma_E": item["sigma_E"],\n'
            '                "rms_delta_E": item["rms_delta_E"],',
            1,
        )

    # Protocol invariants.
    if EXACT_LOG_LINE not in text:
        fail("final certifier lacks explicit log_every=1 invariant")
    if '"--log_interval", str(log_every)' not in text:
        fail("runner command is not tied to log_every")
    if "log_steps" in text:
        fail("legacy log_steps metadata still survives")
    if "sigma_E={metrics['rms_delta_E']" in text:
        fail("[RESULT] still labels rms_delta_E as sigma_E")

    try:
        ast.parse(text, filename=str(CERT))
    except SyntaxError as exc:
        fail(f"internal certifier repair produced invalid Python at line {exc.lineno}: {exc.msg}")
    return text.rstrip() + "\n"


def patch_certify_metrics(text: str) -> tuple[str, int]:
    """Change the scaling key only inside nve_analysis.certify_metrics()."""
    try:
        tree = ast.parse(text, filename=str(ANALYSIS))
    except SyntaxError as exc:
        fail(f"{ANALYSIS} is invalid at line {exc.lineno}: {exc.msg}")

    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "certify_metrics"]
    if len(funcs) != 1:
        fail(f"expected exactly one certify_metrics() in {ANALYSIS}, found {len(funcs)}")
    fn = funcs[0]

    lines = text.splitlines(keepends=True)
    start = fn.lineno - 1
    end = fn.end_lineno
    segment = "".join(lines[start:end])

    replacements = (
        ('["rms_delta_E"]', '["sigma_E"]'),
        ("['rms_delta_E']", "['sigma_E']"),
        ('"rms_delta_E"', '"sigma_E"'),
        ("'rms_delta_E'", "'sigma_E'"),
    )

    changed = 0
    new_segment = segment
    for old, new in replacements:
        c = new_segment.count(old)
        if c:
            new_segment = new_segment.replace(old, new)
            changed += c

    if changed == 0:
        if "sigma_E" in segment:
            print("[SKIP] certify_metrics() already uses sigma_E")
            return text, 0
        fail(
            "certify_metrics() contains neither rms_delta_E nor sigma_E; "
            "inspect simulation/nve_analysis.py manually"
        )

    new_text = "".join(lines[:start]) + new_segment + "".join(lines[end:])

    try:
        ast.parse(new_text, filename=str(ANALYSIS))
    except SyntaxError as exc:
        fail(f"internal nve_analysis repair produced invalid Python at line {exc.lineno}: {exc.msg}")

    # Guard: do not globally eliminate the legacy RMS diagnostic.
    if "rms_delta_E" not in new_text:
        fail(
            "repair would remove rms_delta_E globally; aborting because it must remain "
            "as a secondary diagnostic"
        )
    return new_text.rstrip() + "\n", changed


def main() -> int:
    if not CERT.is_file():
        fail(f"missing {CERT}")
    if not ANALYSIS.is_file():
        fail(f"missing {ANALYSIS}")

    cert_old = CERT.read_text(encoding="utf-8")
    analysis_old = ANALYSIS.read_text(encoding="utf-8")

    cert_new = patch_certifier(cert_old)
    analysis_new, metric_key_changes = patch_certify_metrics(analysis_old)

    # Cross-file invariant: sigma must be produced before certify_metrics consumes it.
    if 'metrics["sigma_E"]' not in cert_new:
        fail('certify_nve.py does not populate metrics["sigma_E"]')
    if "energy_standard_deviation" not in cert_new:
        fail("certify_nve.py does not import/use energy_standard_deviation")

    if cert_new != cert_old:
        backup_once(CERT, CERT_BAK)
        CERT.write_text(cert_new, encoding="utf-8")
        print(f"[PATCH] {CERT}")
    else:
        print(f"[SKIP] {CERT}: already current")

    if analysis_new != analysis_old:
        backup_once(ANALYSIS, ANALYSIS_BAK)
        ANALYSIS.write_text(analysis_new, encoding="utf-8")
        print(f"[PATCH] {ANALYSIS}: certify_metrics sigma-key replacements={metric_key_changes}")
    else:
        print(f"[SKIP] {ANALYSIS}: already current")

    print("[PASS] final NVE sigma(E) protocol is internally consistent")
    print("       sampling: every integration step")
    print("       primary fit observable: sigma_E = std(E_total)")
    print("       secondary diagnostic retained: rms_delta_E")
    print("       default dt grid: 0.001 0.002 0.005 0.01 ps")
    print("       default duration: 5.0 ps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
