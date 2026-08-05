#!/usr/bin/env python3
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation" / "espresso_plugin" / "PaiNN_ML_Potential.cpp"


class PluginSourceInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_periodic_ghosts_are_aliases_not_nodes(self):
        self.assertNotIn("cell_structure.ghost_particles()", self.source)
        self.assertIn("pid_to_idx.find(p1.id())", self.source)
        self.assertIn("pid_to_idx.find(p2.id())", self.source)
        self.assertIn("without a local physical node", self.source)

    def test_energy_and_forces_use_same_scalar(self):
        self.assertIn("const torch::Tensor total_energy = atom_energies.sum();", self.source)
        self.assertIn("m_last_energy = total_energy.item<double>();", self.source)
        self.assertIn("{total_energy}, {t_r_ij}", self.source)
        self.assertNotIn("slice(0, 0, num_local_ml_particles)", self.source)

    def test_zero_edge_energy_is_evaluated(self):
        zero_edge = self.source.index("if (num_edges == 0)")
        forward = self.source.index("model->forward_atom_energies", zero_edge)
        return_after_forward = self.source.index("return;", forward)
        self.assertLess(zero_edge, forward)
        self.assertLess(forward, return_after_forward)
        self.assertNotIn("if (num_edges == 0) return", self.source)


if __name__ == "__main__":
    unittest.main()
