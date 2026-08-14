import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "simulation" / "diagnose_morse_breakage.py"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_morse_breakage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MorseBreakageDiagnosticTests(unittest.TestCase):
    def test_load_morse_prior_uses_framework_default_cutoff(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "priors.json"
            path.write_text(json.dumps({
                "bonds": [
                    {"type": "harmonic", "k": 1.0, "r0": 1.0},
                    {"type": "morse", "D": 50.0, "a": 0.3, "r0": 1.5},
                    {"type": "morse", "D": 20.0, "a": 2.0, "r0": 0.4, "r_cut": 3.0},
                ]
            }))
            first = mod.load_morse_prior(path, 0)
            second = mod.load_morse_prior(path, 1)
        self.assertEqual(first, {"D": 50.0, "a": 0.3, "r0": 1.5, "r_cut": 15.0})
        self.assertEqual(second, {"D": 20.0, "a": 2.0, "r0": 0.4, "r_cut": 3.0})

    def test_analytic_morse_sign_and_minimum(self):
        mod = load_module()
        D, a, r0 = 8.5, 3.2, 0.37
        self.assertAlmostEqual(mod.morse_energy(r0, D, a, r0), 0.0)
        self.assertAlmostEqual(mod.morse_radial_force(r0, D, a, r0), 0.0)
        self.assertLess(mod.morse_radial_force(r0 + 0.1, D, a, r0), 0.0)
        self.assertGreater(mod.morse_radial_force(r0 - 0.1, D, a, r0), 0.0)

    def test_reset_pair_reuses_existing_system(self):
        mod = load_module()
        events = []

        class FakeParticle:
            def __init__(self):
                self.bonds = []

            def add_bond(self, bond):
                self.bonds.append(bond)

        class FakeParticles:
            def clear(self):
                events.append("part.clear")

            def add(self, pos):
                events.append(("part.add", tuple(pos)))
                return FakeParticle()

        class FakeBondedInteractions:
            def clear(self):
                events.append("bonded.clear")

            def add(self, bond):
                events.append("bonded.add")

        class FakeSystem:
            def __init__(self):
                self.part = FakeParticles()
                self.bonded_inter = FakeBondedInteractions()
                self.box_l = [100.0, 100.0, 100.0]
                self.time = 7.0
                self.time_step = 9.0

        class FakeMorse:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        system = FakeSystem()
        params = {"D": 50.0, "a": 0.3, "r0": 1.5, "r_cut": 15.0}
        p0, p1, bond = mod.reset_pair(system, FakeMorse, params, 3.0)

        self.assertEqual(events[:2], ["part.clear", "bonded.clear"])
        self.assertEqual(system.time, 0.0)
        self.assertEqual(system.time_step, 1.0e-4)
        self.assertEqual(bond.kwargs["r_cut"], 15.0)
        self.assertEqual(len(p0.bonds), 1)
        self.assertEqual(len(p1.bonds), 0)

    def test_summary_distinguishes_error_from_topology_deletion(self):
        mod = load_module()

        def probe(success, before=1, after=1):
            return {
                "success": success,
                "bond_before": {"available": True, "count": before},
                "bond_after": {"available": True, "count": after},
            }

        report = {
            "static_probes": {
                "below_cutoff": {"force": probe(True), "energy": probe(True)},
                "at_cutoff": {"force": probe(False), "energy": probe(False)},
                "above_cutoff": {"force": probe(False), "energy": probe(False)},
            },
            "dynamic_crossing": {
                "raised": True,
                "bond_before": {"available": True, "count": 1},
                "bond_after": {"available": True, "count": 1},
            },
        }
        summary = mod.summarize(report)
        self.assertTrue(summary["at_cutoff_force_reports_broken"])
        self.assertTrue(summary["above_cutoff_force_reports_broken"])
        self.assertTrue(summary["bond_topology_unchanged_after_dynamic_error"])
        self.assertTrue(summary["dynamic_crossing_stops_integrator"])


if __name__ == "__main__":
    unittest.main()
