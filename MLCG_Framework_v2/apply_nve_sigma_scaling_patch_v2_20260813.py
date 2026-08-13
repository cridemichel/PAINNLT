#!/usr/bin/env python3
"""Robust incremental NVE sigma(E) certification patch for MLCG_Framework_v2.

Safe after a partial run of apply_nve_sigma_scaling_patch_20260813.py.

Changes:
  * sigma_E = population std(E_total) is available from nve_analysis.py
  * certify_nve.py uses sigma_E as the timestep-scaling observable
  * energy is sampled every integration step by overriding log_every=1
    immediately before the [PLAN] line
  * default dt grid is 0.001, 0.002, 0.005, 0.01 ps
  * default physical duration is 5.0 ps where a recognizable CLI/wrapper
    default is present
  * adds/updates a regression test

The script is idempotent. It makes .pre_sigma_v2 backups before changing files.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
SIM = ROOT / "simulation"
TEL22 = ROOT / "tutorials" / "tel22"
TESTS = ROOT / "tests"

ANALYSIS = SIM / "nve_analysis.py"
CERTIFY = SIM / "certify_nve.py"
WRAPPER = TEL22 / "06_certify_nve.sh"
TEST = TESTS / "test_nve_sigma_scaling.py"

NEW_DTS_CSV = "0.001, 0.002, 0.005, 0.01"
NEW_DTS_SHELL = "0.001 0.002 0.005 0.01"


def fail(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def read(path: Path) -> str:
    if not path.is_file():
        fail(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".pre_sigma_v2")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup}")


def write_changed(path: Path, old: str, new: str) -> None:
    if new == old:
        print(f"[SKIP] {path}: already current")
        return
    backup_once(path)
    path.write_text(new, encoding="utf-8")
    print(f"[PATCH] {path}")


def patch_analysis() -> None:
    old = read(ANALYSIS)
    text = old

    # The first patcher may already have appended this helper. Keep it.
    if "def energy_standard_deviation(" not in text:
        # Prefer adding sigma_E directly to the existing analyze_energy_series
        # return dictionary if the legacy rms_dE key is present.
        m = re.search(r'(?m)^(?P<indent>[ \t]*)"rms_dE"\s*:', text)
        if m and '"sigma_E"' not in text:
            ins = f'{m.group("indent")}"sigma_E": float(np.std(energies, ddof=0)),\n'
            text = text[:m.start()] + ins + text[m.start():]
        else:
            # Generic helper fallback, compatible with the partially patched file.
            if not re.search(r"^import numpy as np\s*$", text, flags=re.M):
                imports = list(re.finditer(r"^(?:from\s+\S+\s+import\s+.*|import\s+.*)\n", text, flags=re.M))
                if not imports:
                    fail("could not locate import block in simulation/nve_analysis.py")
                pos = imports[-1].end()
                text = text[:pos] + "import numpy as np\n" + text[pos:]
            text = text.rstrip() + '''\n\n\ndef energy_standard_deviation(energies):\n    """Population standard deviation of sampled total energy."""\n    values = np.asarray(energies, dtype=np.float64)\n    if values.ndim != 1 or values.size < 3:\n        raise ValueError("At least three energy samples are required")\n    if not np.all(np.isfinite(values)):\n        raise ValueError("Energy series contains NaN or Inf")\n    return float(np.std(values, ddof=0))\n'''

    write_changed(ANALYSIS, old, text)


def replace_argument_default(text: str, option_regex: str, new_default: str) -> tuple[str, bool]:
    # Work inside the source block that begins with the requested add_argument
    # call and ends before the next parser.add_argument. This avoids trying to
    # parse nested function calls with a single parenthesis regex.
    block_re = re.compile(
        rf"(?ms)^(?P<block>[ \t]*parser\.add_argument\(\s*{option_regex}[\s\S]*?)(?=^[ \t]*parser\.add_argument\(|\Z)"
    )
    m = block_re.search(text)
    if not m:
        return text, False
    block = m.group("block")
    dm = re.search(r"default\s*=\s*(\[[^\]]*\]|[^,\n\)]+)", block)
    if not dm:
        return text, False
    new_block = block[:dm.start(1)] + new_default + block[dm.end(1):]
    return text[:m.start("block")] + new_block + text[m.end("block"):], True


def patch_certify() -> None:
    old = read(CERTIFY)
    text = old

    # Make sigma_E available. Handle both direct metric-in-dict and helper styles.
    analysis_has_helper = "def energy_standard_deviation(" in read(ANALYSIS)
    if analysis_has_helper and "energy_standard_deviation" not in text:
        m = re.search(r"^from nve_analysis import (?P<names>[^\n]+)$", text, flags=re.M)
        if m:
            names = m.group("names").strip()
            repl = f"from nve_analysis import {names}, energy_standard_deviation"
            text = text[:m.start()] + repl + text[m.end():]
        else:
            imports = list(re.finditer(r"^(?:from\s+\S+\s+import\s+.*|import\s+.*)\n", text, flags=re.M))
            if not imports:
                fail("could not locate import block in simulation/certify_nve.py")
            pos = imports[-1].end()
            text = text[:pos] + "from nve_analysis import energy_standard_deviation\n" + text[pos:]

    # Update CLI defaults when recognizable. These are conveniences; the final
    # run command below also supplies explicit environment values.
    text2, changed_dts = replace_argument_default(
        text, r"['\"]--dts['\"]", f"[{NEW_DTS_CSV}]"
    )
    text = text2

    # duration option spelling has varied; try both common forms.
    changed_duration = False
    for opt in (r"['\"]--duration-ps['\"]", r"['\"]--duration_ps['\"]"):
        text2, changed = replace_argument_default(text, opt, "5.0")
        if changed:
            text = text2
            changed_duration = True
            break

    # Critical invariant: override sparse logging immediately before the [PLAN]
    # line. The user's current output proves this anchor exists even if the
    # upstream log_every computation has a different or multiline form.
    if not re.search(
        r'(?m)^\s*log_every\s*=\s*1\s*#\s*NVE certification: sample energy every integration step\s*$',
        text,
    ):
        lines = text.splitlines(keepends=True)
        plan_index = None
        plan_indent = ""
        for i, line in enumerate(lines):
            if "[PLAN]" in line and "log_every" in line:
                plan_index = i
                plan_indent = re.match(r"^[ \t]*", line).group(0)
                break
        if plan_index is None:
            # Less strict fallback: any PLAN line in the per-dt loop.
            for i, line in enumerate(lines):
                if "[PLAN]" in line and "dt=" in line:
                    plan_index = i
                    plan_indent = re.match(r"^[ \t]*", line).group(0)
                    break
        if plan_index is None:
            nearby = "\n".join(l for l in text.splitlines() if "PLAN" in l or "log_every" in l)
            fail("could not locate [PLAN] anchor in certify_nve.py. Found:\n" + (nearby or "(none)"))
        override = (
            f"{plan_indent}log_every = 1  # NVE certification: sample energy every integration step\n"
        )
        lines.insert(plan_index, override)
        text = "".join(lines)

    # Ensure per-run sigma_E exists. If analyze_energy_series already returns
    # sigma_E, no helper call is needed. Otherwise add the helper result.
    if 'metrics["sigma_E"]' not in text and "metrics['sigma_E']" not in text:
        m = re.search(
            r'(?m)^(?P<indent>[ \t]*)metrics\s*=\s*analyze_energy_series\((?P<args>[^\n]+)\)\s*$',
            text,
        )
        if m:
            # If nve_analysis contains sigma_E directly, metrics already has it.
            if '"sigma_E"' not in read(ANALYSIS):
                if not analysis_has_helper:
                    fail("sigma_E is unavailable from nve_analysis.py")
                indent = m.group("indent")
                line = m.group(0)
                insertion = line + f'\n{indent}metrics["sigma_E"] = energy_standard_deviation(energies)'
                text = text[:m.start()] + insertion + text[m.end():]
        else:
            fail("could not locate analyze_energy_series(...) call in certify_nve.py")

    # Certification driver: sigma_E is the primary observable. nve_analysis.py
    # keeps rms_dE for backward-compatible diagnostics.
    text = text.replace('metrics["rms_dE"]', 'metrics["sigma_E"]')
    text = text.replace("metrics['rms_dE']", "metrics['sigma_E']")
    text = text.replace("rms_dE=", "sigma_E=")

    # Common report/table field names and fit extraction.
    text = text.replace('"rms_dE": metrics["sigma_E"]', '"sigma_E": metrics["sigma_E"]')
    text = text.replace("'rms_dE': metrics['sigma_E']", "'sigma_E': metrics['sigma_E']")
    text = text.replace('row["rms_dE"]', 'row["sigma_E"]')
    text = text.replace("row['rms_dE']", "row['sigma_E']")

    if "sigma_E" not in text:
        fail("sigma_E did not become part of certify_nve.py")

    write_changed(CERTIFY, old, text)
    if not changed_dts:
        print("[WARN] --dts argparse default not recognized; use explicit NVE_DTS in the run command")
    if not changed_duration:
        print("[WARN] duration argparse default not recognized; use explicit NVE_DURATION_PS in the run command")


def patch_wrapper() -> None:
    old = read(WRAPPER)
    text = old

    # Shell parameter-expansion defaults, if present.
    text = re.sub(
        r"\$\{NVE_DTS:-[^}]*\}",
        "${NVE_DTS:-" + NEW_DTS_SHELL + "}",
        text,
    )
    text = re.sub(
        r"\$\{NVE_DURATION_PS:-[^}]*\}",
        "${NVE_DURATION_PS:-5.0}",
        text,
    )

    # Literal legacy grids/durations used in earlier revisions.
    for old_grid in (
        "0.002 0.001 0.0005 0.00025",
        "0.002 0.001 0.0005",
        "0.002,0.001,0.0005,0.00025",
        "0.002,0.001,0.0005",
    ):
        text = text.replace(old_grid, NEW_DTS_SHELL if " " in old_grid else "0.001,0.002,0.005,0.01")

    # Only replace obvious NVE duration default assignments, not arbitrary 0.5.
    text = re.sub(
        r'(?m)^(?P<prefix>\s*(?:NVE_DURATION_PS|DURATION_PS)\s*=\s*["\']?)0\.5(?P<suffix>["\']?\s*)$',
        r'\g<prefix>5.0\g<suffix>',
        text,
    )

    write_changed(WRAPPER, old, text)


def write_test() -> None:
    content = r'''import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "simulation" / "nve_analysis.py"
SPEC = importlib.util.spec_from_file_location("nve_analysis_sigma_test", MODULE_PATH)
NVE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NVE)


class TestNVESigmaScaling(unittest.TestCase):
    def sigma(self, energies):
        if hasattr(NVE, "energy_standard_deviation"):
            return NVE.energy_standard_deviation(energies)
        times = np.arange(len(energies), dtype=float)
        return NVE.analyze_energy_series(times, energies)["sigma_E"]

    def test_sigma_energy_is_population_std(self):
        energies = np.array([10.0, 11.0, 9.0, 10.5, 9.5], dtype=float)
        self.assertAlmostEqual(
            self.sigma(energies),
            float(np.std(energies, ddof=0)),
            places=14,
        )

    def test_sigma_scales_as_dt_squared(self):
        dts = np.array([0.001, 0.002, 0.005, 0.01], dtype=float)
        phase = np.linspace(0.0, 20.0 * math.pi, 20001)
        carrier = np.sin(phase) + 0.25 * np.cos(0.37 * phase)
        sigmas = []
        for dt in dts:
            energies = 2000.0 + 1.0e6 * dt * dt * carrier
            sigmas.append(self.sigma(energies))
        p = float(np.polyfit(np.log(dts), np.log(sigmas), 1)[0])
        self.assertAlmostEqual(p, 2.0, places=8)

    def test_certifier_overrides_sampling_every_step(self):
        source = (ROOT / "simulation" / "certify_nve.py").read_text(encoding="utf-8")
        self.assertIn(
            "log_every = 1  # NVE certification: sample energy every integration step",
            source,
        )
        self.assertIn("sigma_E", source)


if __name__ == "__main__":
    unittest.main()
'''
    old = TEST.read_text(encoding="utf-8") if TEST.exists() else ""
    if old == content:
        print(f"[SKIP] {TEST}: already current")
    else:
        if TEST.exists():
            backup_once(TEST)
        TEST.write_text(content, encoding="utf-8")
        print(f"[WRITE] {TEST}")


def main() -> int:
    patch_analysis()
    patch_certify()
    patch_wrapper()
    write_test()
    print("[DONE] NVE sigma(E) patch v2 applied")
    print("       target dt grid: 0.001 0.002 0.005 0.01 ps")
    print("       target duration: 5.0 ps")
    print("       sampling: every integration step")
    print("       primary scaling metric: sigma_E = std(E_total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
