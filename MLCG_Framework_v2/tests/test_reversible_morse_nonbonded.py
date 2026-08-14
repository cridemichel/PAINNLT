from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "simulation"
PLUGIN = SIM / "espresso_plugin"
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(ROOT / "preprocessing"))

from prior_kernels import switched_morse_radial_force_array  # noqa: E402
from espresso_interactions import (  # noqa: E402
    configure_pair_specific_morse,
    create_pair_specific_morse_markers,
    configure_type_pair_morse,
    max_type_pair_morse_cutoff,
    prepare_pair_specific_morse,
    prepare_type_pair_morse,
    switched_morse_energy_radial_force,
)


def load_installer():
    path = PLUGIN / "install_switched_morse_nonbonded.py"
    spec = importlib.util.spec_from_file_location("install_switched_morse_nonbonded", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeMorseHandle:
    def __init__(self):
        self.calls = []

    def set_params(self, **kwargs):
        self.calls.append(kwargs)


class FakePairHandle:
    def __init__(self):
        self.morse = FakeMorseHandle()


class FakeNonBonded:
    def __init__(self):
        self.pairs = {}

    def __getitem__(self, key):
        key = tuple(key)
        return self.pairs.setdefault(key, FakePairHandle())


class FakeParticle:
    def __init__(self, pid, **kwargs):
        self.id = pid
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.virtual = False
        self.related_to = None
        self.gamma = None
        self.gamma_rot = None

    def vs_auto_relate_to(self, parent_pid):
        self.related_to = parent_pid


class FakeParticles:
    def __init__(self):
        self.items = []

    def add(self, **kwargs):
        particle = FakeParticle(len(self.items), **kwargs)
        self.items.append(particle)
        return particle

    def by_id(self, pid):
        return self.items[int(pid)]


class FakeSystem:
    def __init__(self):
        self.non_bonded_inter = FakeNonBonded()
        self.part = FakeParticles()


class ReversibleMorseTests(unittest.TestCase):
    def test_pair_mapping_is_deterministic_and_site_addressable(self):
        priors = {
            "bonds": [
                {"type": "morse", "mol_i": 8, "mol_j": 2, "site_j": 0,
                 "D": 5, "a": 2, "r0": 0.4},
                {"type": "harmonic", "mol_i": 1, "mol_j": 2, "k": 3, "r0": 0.2},
                {"type": "morse", "mol_i": 4, "site_i": 1, "mol_j": 8,
                 "D": 7, "a": 3, "r0": 0.5},
            ]
        }
        types, contacts = prepare_pair_specific_morse(priors, num_species=6)
        self.assertEqual(types, {(2, 0): 8, (4, 1): 9, (8, -1): 10})
        self.assertEqual(len(contacts), 2)
        self.assertEqual((contacts[0]["site_i"], contacts[0]["site_j"]), (-1, 0))
        self.assertGreater(contacts[0]["r_switch"], contacts[0]["r0"])
        self.assertLess(contacts[0]["r_switch"], contacts[0]["r_cut"])

    def test_duplicate_endpoint_pairs_and_invalid_site_indices_fail_closed(self):
        duplicate = {"bonds": [
            {"type": "morse", "mol_i": 1, "site_i": 0, "mol_j": 2, "site_j": 1,
             "D": 1, "a": 1, "r0": 0.5},
            {"type": "morse", "mol_i": 2, "site_i": 1, "mol_j": 1, "site_j": 0,
             "D": 2, "a": 2, "r0": 0.6},
        ]}
        with self.assertRaisesRegex(ValueError, "Duplicate pair-specific Morse endpoint pair"):
            prepare_pair_specific_morse(duplicate, 4)
        invalid_site = {"bonds": [
            {"type": "morse", "mol_i": 1, "mol_j": 2, "site_i": -2,
             "D": 1, "a": 1, "r0": 0.5},
        ]}
        with self.assertRaisesRegex(ValueError, "use -1 for COM"):
            prepare_pair_specific_morse(invalid_site, 4)

    def test_runtime_configuration_uses_marker_type_pair(self):
        priors = {"bonds": [
            {"type": "morse", "mol_i": 1, "site_i": 0, "mol_j": 3, "site_j": -1,
             "D": 8, "a": 2.5, "r0": 0.6, "r_switch": 2.0, "r_cut": 2.5},
        ]}
        types, contacts = prepare_pair_specific_morse(priors, 5)
        system = FakeSystem()
        configure_pair_specific_morse(system, contacts, types)
        handle = system.non_bonded_inter[types[(1, 0)], types[(3, -1)]].morse
        self.assertEqual(handle.calls, [{
            "eps": 8.0, "alpha": 2.5, "rmin": 0.6,
            "cutoff": 2.5, "switch_start": 2.0,
        }])

    def test_marker_creation_preserves_physical_site_types_and_attachment_point(self):
        system = FakeSystem()
        com0 = system.part.add(pos=[0.0, 0.0, 0.0], type=6, mass=10.0,
                               rinertia=[1.0, 1.0, 1.0], mol_id=0)
        site0 = system.part.add(pos=[0.2, 0.0, 0.0], type=1, mass=1.0e-5,
                                rinertia=[1.0e-5] * 3, mol_id=0)
        com1 = system.part.add(pos=[1.0, 0.0, 0.0], type=6, mass=10.0,
                               rinertia=[1.0, 1.0, 1.0], mol_id=1)
        site1 = system.part.add(pos=[1.0, 0.3, 0.0], type=2, mass=1.0e-5,
                                rinertia=[1.0e-5] * 3, mol_id=1)
        marker_types = {(0, 0): 8, (1, -1): 9}
        marker_parts = create_pair_specific_morse_markers(
            system, marker_types, {0: com0.id, 1: com1.id},
            {(0, 0): site0.id, (1, 0): site1.id},
        )
        marker_site = system.part.by_id(marker_parts[(0, 0)])
        marker_com = system.part.by_id(marker_parts[(1, -1)])
        self.assertEqual(marker_site.pos, site0.pos)
        self.assertEqual(marker_site.related_to, com0.id)
        self.assertEqual(marker_site.type, 8)
        self.assertTrue(marker_site.virtual)
        self.assertEqual(marker_com.pos, com1.pos)
        self.assertEqual(marker_com.related_to, com1.id)
        self.assertEqual(marker_com.type, 9)
        self.assertEqual(site0.type, 1)
        self.assertEqual(site1.type, 2)

    def test_marker_creation_rejects_missing_site(self):
        system = FakeSystem()
        com = system.part.add(pos=[0.0, 0.0, 0.0], type=5, mass=10.0,
                              rinertia=[1.0, 1.0, 1.0], mol_id=0)
        with self.assertRaisesRegex(ValueError, "missing CG site 0:3"):
            create_pair_specific_morse_markers(
                system, {(0, 3): 7}, {0: com.id}, {}
            )

    def test_type_pair_morse_uses_physical_site_types(self):
        priors = {"morse_type_pairs": [
            {"type_i": 3, "type_j": 1, "D": 4.0, "a": 2.0,
             "r0": 0.55, "r_switch": 0.9, "r_cut": 1.2},
            {"type_i": 2, "type_j": 2, "D": 1.5, "a": 3.0,
             "r0": 0.4, "r_cut": 0.8},
        ]}
        items = prepare_type_pair_morse(priors, num_species=4)
        self.assertEqual((items[0]["type_i"], items[0]["type_j"]), (1, 3))
        self.assertGreater(items[1]["r_switch"], items[1]["r0"])
        self.assertLess(items[1]["r_switch"], items[1]["r_cut"])
        self.assertEqual(max_type_pair_morse_cutoff(items), 1.2)

        system = FakeSystem()
        configure_type_pair_morse(system, items)
        self.assertEqual(
            system.non_bonded_inter[1, 3].morse.calls,
            [{"eps": 4.0, "alpha": 2.0, "rmin": 0.55,
              "cutoff": 1.2, "switch_start": 0.9}],
        )

    def test_type_pair_morse_rejects_duplicate_unknown_and_implicit_cutoff(self):
        with self.assertRaisesRegex(ValueError, "explicit r_cut"):
            prepare_type_pair_morse({"morse_type_pairs": [
                {"type_i": 0, "type_j": 1, "D": 1, "a": 2, "r0": 0.4}
            ]}, 3)
        with self.assertRaisesRegex(ValueError, "valid CG site types"):
            prepare_type_pair_morse({"morse_type_pairs": [
                {"type_i": 0, "type_j": 4, "D": 1, "a": 2, "r0": 0.4, "r_cut": 1.0}
            ]}, 3)
        with self.assertRaisesRegex(ValueError, "Duplicate Morse type pair"):
            prepare_type_pair_morse({"morse_type_pairs": [
                {"type_i": 0, "type_j": 1, "D": 1, "a": 2, "r0": 0.4, "r_cut": 1.0},
                {"type_i": 1, "type_j": 0, "D": 2, "a": 3, "r0": 0.5, "r_cut": 1.1},
            ]}, 3)

    def test_python_switched_morse_is_energy_force_consistent(self):
        params = dict(D=8.5, a=3.2, r0=0.37, r_switch=1.1, r_cut=1.5)
        for r in (0.51, 1.1, 1.25, 1.49):
            energy, force = switched_morse_energy_radial_force(r, **params)
            h = 1.0e-6
            e_plus = switched_morse_energy_radial_force(r + h, **params)[0]
            e_minus = switched_morse_energy_radial_force(r - h, **params)[0]
            self.assertAlmostEqual(force, -(e_plus - e_minus) / (2 * h), places=7)
        self.assertAlmostEqual(
            switched_morse_energy_radial_force(params["r0"], **params)[0],
            -params["D"], places=13)
        self.assertEqual(switched_morse_energy_radial_force(params["r_cut"], **params), (0.0, 0.0))
        self.assertEqual(switched_morse_energy_radial_force(params["r_cut"] + 0.2, **params), (0.0, 0.0))

    def test_preprocessing_vectorized_morse_matches_runtime_scalar_kernel(self):
        params = dict(D=8.5, a=3.2, r0=0.37, r_switch=1.1, r_cut=1.5)
        distances = np.asarray([0.51, 1.10, 1.25, 1.49, 1.50, 1.70])
        got = switched_morse_radial_force_array(
            distances,
            np.full_like(distances, params["D"]),
            np.full_like(distances, params["a"]),
            np.full_like(distances, params["r0"]),
            np.full_like(distances, params["r_switch"]),
            np.full_like(distances, params["r_cut"]),
        )
        expected = np.asarray([
            switched_morse_energy_radial_force(r, **params)[1]
            for r in distances
        ])
        self.assertTrue(np.allclose(got, expected, rtol=1e-13, atol=1e-13))

    def test_cpp_switched_kernel_is_energy_force_consistent(self):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No C++ compiler available")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "config").mkdir()
            (td / "config/config.hpp").write_text("#pragma once\n#define ESPRESSO_MORSE 1\n")
            (td / "nonbonded_interaction_data.hpp").write_text(r'''#pragma once
struct Morse_Parameters {
  double eps=0., alpha=-1., rmin=-1., cut=-1., switch_start=-1., rest=-1.;
};
struct IA_parameters { Morse_Parameters morse; };
''')
            source = td / "test.cpp"
            source.write_text(r'''#include "morse_switched.hpp"
#include <cmath>
#include <array>
int main() {
  IA_parameters ia;
  ia.morse.eps=8.5; ia.morse.alpha=3.2; ia.morse.rmin=0.37;
  ia.morse.switch_start=1.1; ia.morse.cut=1.5; ia.morse.rest=0.0;
  auto e = [&](double r) { return morse_pair_energy(ia, r); };
  for (double r : std::array<double, 4>{0.51, 1.10, 1.25, 1.49}) {
    double h=1e-6;
    double fd=-(e(r+h)-e(r-h))/(2*h);
    double f=morse_pair_force_factor(ia,r)*r;
    if (std::abs(fd-f) > 2e-7) return 1;
  }
  if (std::abs(e(0.37)+8.5) > 1e-13) return 2;
  if (morse_pair_energy(ia,1.5) != 0.0) return 3;
  if (morse_pair_force_factor(ia,1.5) != 0.0) return 4;
  if (morse_pair_energy(ia,1.7) != 0.0) return 5;
  if (morse_pair_force_factor(ia,1.7) != 0.0) return 6;
  // Backward-compatible stock branch: shifted energy is zero at cutoff.
  ia.morse.switch_start=-1.0;
  double y=std::exp(-ia.morse.alpha*(ia.morse.cut-ia.morse.rmin));
  ia.morse.rest=ia.morse.eps*(y*y-2*y);
  if (std::abs(morse_pair_energy(ia,ia.morse.cut-1e-8)) > 1e-5) return 7;
  return 0;
}
''')
            exe = td / "test"
            subprocess.run(
                [compiler, "-std=c++20", "-O2", "-I", str(td), "-I", str(PLUGIN), str(source), "-o", str(exe)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(exe)], check=True)

    def test_preprocessing_pair_specific_morse_is_site_and_torque_aware(self):
        source = (ROOT / "preprocessing" / "build_cg_dataset.py").read_text(encoding="utf-8")
        self.assertIn('site_i = b.get("site_i", -1)', source)
        self.assertIn('site_j = b.get("site_j", -1)', source)
        self.assertIn('resolve_site_position(frame_centers, frame_sites, i, site_i)', source)
        self.assertIn('resolve_site_position(frame_centers, frame_sites, j, site_j)', source)
        self.assertIn('if site_i != -1:', source)
        self.assertIn('if site_j != -1:', source)
        self.assertIn('np.cross(r_rel_i, f_vec)', source)
        self.assertIn('np.cross(r_rel_j, f_vec)', source)

    def test_preprocessing_subtracts_type_pair_morse_with_nonbonded_exclusions(self):
        source = (ROOT / "preprocessing" / "build_cg_dataset.py").read_text(encoding="utf-8")
        self.assertIn('config_data.get("morse_type_pairs", [])', source)
        self.assertIn('derived_priors.get("morse_type_pairs", [])', source)
        self.assertIn('morse_excluded = wca_one_three_mol_matrix', source)
        self.assertIn('_direct_site_excluded_mask(', source)
        self.assertIn('switched_morse_radial_force_array(', source)
        self.assertIn('np.add.at(res_torques, mol_i', source)

    def test_installer_is_idempotent_on_espresso_5_style_tree(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = '''struct Morse_Parameters {
  double eps = 0.;
  double alpha = inactive_cutoff;
  double rmin = inactive_cutoff;
  double cut = inactive_cutoff;
  double rest = inactive_cutoff;
  Morse_Parameters() = default;
  Morse_Parameters(double eps, double alpha, double rmin, double cutoff);
  double max_cutoff() const { return cut; }
};\n'''
            script = '''#ifdef ESPRESSO_MORSE
class InteractionMorse
    : public InteractionPotentialInterface<::Morse_Parameters> {
 protected:
  CoreInteraction IA_parameters::*get_ptr_offset() const override {
    return &::IA_parameters::morse;
  }
 public:
  InteractionMorse() {
    add_parameters({
        make_autoparameter(&CoreInteraction::eps, "eps"),
        make_autoparameter(&CoreInteraction::alpha, "alpha"),
        make_autoparameter(&CoreInteraction::rmin, "rmin"),
        make_autoparameter(&CoreInteraction::cut, "cutoff"),
    });
  }
 private:
  void make_new_instance(VariantMap const &params) override {
    m_handle =
        make_shared_from_args<CoreInteraction, double, double, double, double>(
            params, "eps", "alpha", "rmin", "cutoff");
  }
};
#endif // ESPRESSO_MORSE
'''
            py = '''class MorseInteraction(NonBondedInteraction):
    def default_params(self):
        return {"cutoff": 0.}

class Other(NonBondedInteraction):
    pass
'''
            files = {
                "src/config/myconfig-default.hpp": "#pragma once\n#define WCA\n",
                "src/core/nonbonded_interactions/nonbonded_interaction_data.hpp": data,
                "src/core/nonbonded_interactions/morse.hpp": "stock header\n",
                "src/core/nonbonded_interactions/morse.cpp": "stock cpp\n",
                "src/script_interface/interactions/NonBondedInteraction.hpp": script,
                "src/python/espressomd/interactions.py": py,
            }
            for rel, content in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            (root / "build").mkdir()
            first = installer.install(root, PLUGIN)
            build_config = (root / "build/myconfig.hpp").read_text()
            self.assertIn("#define WCA", build_config)
            self.assertIn("#define MORSE", build_config)
            self.assertTrue(first)
            installer.check(root, PLUGIN)
            snapshot = {rel: (root / rel).read_text() for rel in files}
            config_snapshot = (root / "build/myconfig.hpp").read_text()
            second = installer.install(root, PLUGIN)
            self.assertEqual(second, [])
            self.assertEqual(snapshot, {rel: (root / rel).read_text() for rel in files})
            self.assertEqual(config_snapshot, (root / "build/myconfig.hpp").read_text())

    def test_feature_enabler_preserves_existing_build_configuration(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "build").mkdir(parents=True)
            config = root / "build/myconfig.hpp"
            config.write_text("#pragma once\n#define WCA\n#define ROTATION\n")
            path, changed = installer.ensure_morse_build_feature(root)
            self.assertEqual(path, config)
            self.assertTrue(changed)
            text = config.read_text()
            self.assertIn("#define WCA", text)
            self.assertIn("#define ROTATION", text)
            self.assertEqual(text.count("#define MORSE"), 1)
            _path, changed_again = installer.ensure_morse_build_feature(root)
            self.assertFalse(changed_again)
            self.assertEqual(config.read_text().count("#define MORSE"), 1)

    def test_production_runtimes_use_site_addressable_marker_morse_not_bonded_morse(self):
        for rel in ("simulation/run_cg_md.py", "simulation/equilibrate.py"):
            text = (ROOT / rel).read_text()
            self.assertIn("create_pair_specific_morse_markers", text)
            self.assertIn("configure_pair_specific_morse", text)
            self.assertIn("morse_marker_types", text)
            self.assertNotIn("com_runtime_type", text)
            self.assertNotIn("bond = make_analytic_morse_bond", text)

    def test_hybrid_decomposition_is_active_before_long_pair_specific_morse(self):
        for rel in ("simulation/run_cg_md.py", "simulation/equilibrate.py"):
            text = (ROOT / rel).read_text()
            search_from = text.index("regular_cutoff =")
            neighbor_call = text.index(
                "configure_neighbor_search(\n    system, args.neighbor_search,",
                search_from,
            )
            type_pair_call = text.index(
                "configure_type_pair_morse(system, morse_type_pairs)",
                search_from,
            )
            morse_call = text.index(
                "configure_pair_specific_morse(system, morse_contacts, morse_marker_types)",
                search_from,
            )
            self.assertLess(
                neighbor_call,
                type_pair_call,
                f"{rel} must validate/configure the cell decomposition before "
                "registering type-pair Morse interactions",
            )
            self.assertLess(
                neighbor_call,
                morse_call,
                f"{rel} must activate the hybrid/N-square decomposition before "
                "registering long-cutoff pair-specific Morse interactions",
            )


if __name__ == "__main__":
    unittest.main()
