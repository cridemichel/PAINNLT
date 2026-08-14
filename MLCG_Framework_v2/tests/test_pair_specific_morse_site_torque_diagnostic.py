import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "simulation"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

SPEC = importlib.util.spec_from_file_location(
    "morse_site_torque_diag", SIM / "diagnose_pair_specific_morse_site_torque.py"
)
DIAG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DIAG)


class MorseSiteTorqueDiagnosticTests(unittest.TestCase):
    def test_expected_wrench_has_equal_opposite_forces_and_nonzero_torques(self):
        params = dict(D=6.0, a=1.2, r0=2.0, r_switch=3.0, r_cut=4.0)
        result = DIAG.expected_site_pair_wrench(
            com_i=[2.5, 5.0, 5.0],
            site_i=[2.5, 5.4, 5.0],
            com_j=[5.0, 5.0, 5.0],
            site_j=[5.0, 4.7, 5.0],
            box_l=[10.0, 10.0, 10.0],
            params=params,
        )
        for left, right in zip(result["force_i"], result["force_j"]):
            self.assertAlmostEqual(left, -right, places=13)
        self.assertGreater(math.sqrt(sum(x * x for x in result["torque_i"])), 1.0e-6)
        self.assertGreater(math.sqrt(sum(x * x for x in result["torque_j"])), 1.0e-6)

    def test_runtime_probe_uses_production_marker_and_hybrid_helpers(self):
        source = (SIM / "diagnose_pair_specific_morse_site_torque.py").read_text()
        self.assertIn("prepare_pair_specific_morse(priors, num_species)", source)
        self.assertIn("create_pair_specific_morse_markers(", source)
        self.assertIn("configure_pair_specific_morse(system, contacts, marker_types)", source)
        self.assertIn("n_square_types={com_type, *marker_types.values()}", source)
        self.assertIn("p[\"com_i\"].torque_lab", source)
        self.assertIn("p[\"com_j\"].torque_lab", source)


if __name__ == "__main__":
    unittest.main()
