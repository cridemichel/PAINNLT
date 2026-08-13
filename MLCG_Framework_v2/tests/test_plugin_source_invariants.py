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
        self.assertIn(
            "const torch::Tensor total_energy = sum_atom_energies_for_hamiltonian(atom_energies);",
            self.source,
        )
        self.assertIn("atom_energies.to(torch::kFloat64).sum()", self.source)
        self.assertIn("if (atom_energies.device().is_cpu())", self.source)
        self.assertIn("m_last_energy = total_energy.item<double>();", self.source)
        self.assertIn("{total_energy}, {t_r_ij}", self.source)
        self.assertNotIn("slice(0, 0, num_local_ml_particles)", self.source)


    def test_graph_layout_is_deterministic_and_pairs_are_unique(self):
        self.assertIn("std::sort(", self.source)
        self.assertIn("lhs->id() < rhs->id()", self.source)
        self.assertIn("std::map<PairKey, Displacement> physical_pairs", self.source)
        self.assertIn("physical_pairs.emplace", self.source)

    def test_energy_gauge_is_reported(self):
        self.assertIn("isolated_species_zero_v1", self.source)
        self.assertIn("isolated_species_reference_table", self.source)

    def test_zero_edge_energy_is_evaluated(self):
        zero_edge = self.source.index("if (num_edges == 0)")
        forward = self.source.index("model->forward_atom_energies", zero_edge)
        return_after_forward = self.source.index("return;", forward)
        self.assertLess(zero_edge, forward)
        self.assertLess(forward, return_after_forward)
        self.assertNotIn("if (num_edges == 0) return", self.source)

    def test_runtime_precision_is_selectable_without_changing_fp32_default(self):
        header = (ROOT / "simulation" / "espresso_plugin" / "PaiNN_ML_Potential.hpp").read_text(encoding="utf-8")
        pyx = (ROOT / "simulation" / "espresso_plugin" / "painn.pyx").read_text(encoding="utf-8")
        runner = (ROOT / "simulation" / "run_cg_md.py").read_text(encoding="utf-8")
        self.assertIn('precision_str = "float32"', header)
        self.assertIn('m_dtype{torch::kFloat32}', header)
        self.assertIn('model->to(m_dtype)', self.source)
        self.assertIn('dtype(m_dtype)', self.source)
        self.assertIn('accessor<double, 2>()', self.source)
        self.assertIn('precision: str = "float32"', pyx)
        self.assertIn('--ml_precision', runner)



if __name__ == "__main__":
    unittest.main()
