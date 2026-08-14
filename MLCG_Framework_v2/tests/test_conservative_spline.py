#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "ibi"))
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "training"))

from conservative_spline import (  # noqa: E402
    ConservativeSplinePrior,
    conservative_angle_forces,
    conservative_distance_forces,
    conservative_spline_value,
    load_conservative_spline,
)
from convert_to_conservative_spline import convert  # noqa: E402
from validate_conservative_spline import validate  # noqa: E402
from framework_utils import nonconservative_prior_entries  # noqa: E402
from residual_input_provenance import referenced_prior_artifacts  # noqa: E402


def fd_force(positions, energy_fn, eps=1.0e-7):
    positions = np.asarray(positions, dtype=float)
    out = np.zeros_like(positions)
    for i in range(len(positions)):
        for a in range(3):
            plus = positions.copy(); plus[i, a] += eps
            minus = positions.copy(); minus[i, a] -= eps
            out[i, a] = -(energy_fn(plus) - energy_fn(minus)) / (2.0 * eps)
    return out


class ConservativeSplineTests(unittest.TestCase):
    def make_table(self, kind="bond"):
        x = np.linspace(0.4, 2.8, 121) if kind == "bond" else np.linspace(0.0, np.pi, 121)
        center = 1.25 if kind == "bond" else 1.45
        u = 4.0 * (x - center) ** 2 + 0.2 * (x - center) ** 4
        d = PchipInterpolator(x, u)(x, 1)
        return ConservativeSplinePrior(x, u, d, float(x[0]), float(x[-1]), kind, Path("synthetic.dat"))

    def test_scalar_energy_and_derivative_are_same_polynomial(self):
        t = self.make_table("bond")
        for q in np.linspace(0.5, 2.7, 31):
            u, du = conservative_spline_value(t, q)
            eps = 1e-7
            up = conservative_spline_value(t, q + eps)[0]
            um = conservative_spline_value(t, q - eps)[0]
            self.assertAlmostEqual(du, (up - um) / (2 * eps), places=6)
            self.assertTrue(np.isfinite(u))

    def test_distance_cartesian_force_is_energy_gradient(self):
        t = self.make_table("bond")
        box = np.array([20.0, 20.0, 20.0])
        pos = np.array([[1.0, 2.0, 3.0], [2.7, 2.2, 3.0]])
        actual = np.vstack(conservative_distance_forces(pos[0], pos[1], box, t))
        def energy(coords):
            d = coords[1] - coords[0]
            d -= box * np.round(d / box)
            return conservative_spline_value(t, np.linalg.norm(d))[0]
        expected = fd_force(pos, energy)
        self.assertTrue(np.allclose(actual, expected, rtol=3e-6, atol=3e-7))

    def test_angle_cartesian_force_is_energy_gradient(self):
        t = self.make_table("angle")
        box = np.array([20.0, 20.0, 20.0])
        pos = np.array([[0.2, 0.4, 0.1], [1.0, 1.0, 1.0], [2.2, 1.4, 0.7]])
        actual = np.vstack(conservative_angle_forces(*pos, box, t))
        def energy(coords):
            a = coords[0] - coords[1]; b = coords[2] - coords[1]
            theta = np.arccos(np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))
            return conservative_spline_value(t, theta)[0]
        expected = fd_force(pos, energy)
        self.assertTrue(np.allclose(actual, expected, rtol=5e-5, atol=4e-6))

    def test_conversion_is_read_only_and_validates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            x_b = np.linspace(0.4, 2.8, 101)
            u_b = 3.2 * (x_b - 1.1) ** 2
            f_b = -6.4 * (x_b - 1.1)
            np.savetxt(root / "bond_tabulated_g.dat", np.column_stack([x_b, u_b, f_b]))
            x_a = np.linspace(0.0, np.pi, 101)
            u_a = 2.1 * (x_a - 1.4) ** 2
            f_a = 4.2 * (x_a - 1.4)
            np.savetxt(root / "angle_tabulated_g.dat", np.column_stack([x_a, u_a, f_a]))
            priors = root / "cg_priors.json"
            priors.write_text(json.dumps({
                "bonds": [{"type":"tabulated","file":"bond_tabulated_g.dat","min":0.4,"max":2.8,"ibi_mode":"ibi"}],
                "angles": [{"type":"tabulated","file":"angle_tabulated_g.dat","min":0.0,"max":float(np.pi),"ibi_mode":"ibi"}],
                "dihedrals": [],
            }))
            before = priors.read_bytes()
            out = root / "conservative"
            report = convert(priors, out)
            self.assertEqual(priors.read_bytes(), before)
            converted = json.loads((out / "cg_priors.json").read_text())
            self.assertEqual(converted["bonds"][0]["type"], "conservative_spline")
            self.assertEqual(converted["angles"][0]["type"], "conservative_spline")
            load_conservative_spline(converted["bonds"][0], kind="bond", priors_path=out / "cg_priors.json")
            result = validate(out / "conversion_report.json")
            self.assertTrue(result["pass"])
            self.assertEqual(report["converted_unique_tables"], 2)

    def test_tabulated_dihedral_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            x = np.linspace(0, 2*np.pi, 51)
            np.savetxt(root / "d.dat", np.column_stack([x, 1-np.cos(x), np.zeros_like(x)]))
            priors = root / "p.json"
            priors.write_text(json.dumps({"bonds":[],"angles":[],"dihedrals":[{"type":"tabulated","file":"d.dat","min":0.0,"max":float(2*np.pi)}]}))
            with self.assertRaisesRegex(ValueError, "bond\\+angle only"):
                convert(priors, root / "out")

    def test_conservative_splines_are_not_flagged_as_nonconservative_tables(self):
        p = {"bonds":[{"type":"conservative_spline"}], "angles":[{"type":"conservative_spline"}]}
        self.assertEqual(nonconservative_prior_entries(p), [])


    def test_installer_is_idempotent_on_espresso_5_style_tree(self):
        installer_path = ROOT / "simulation/espresso_plugin/install_conservative_spline_bond.py"
        spec = importlib.util.spec_from_file_location("install_conservative_spline_bond", installer_path)
        installer = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(installer)
        header = ROOT / "simulation/espresso_plugin/conservative_spline_bond.hpp"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            files = {
                "src/core/bonded_interactions/bonded_interaction_data.hpp":
                    '#include "harmonic.hpp"\nusing Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n',
                "src/core/forces_inline.hpp":
                    'auto calc_bond_pair_force(Bonded_IA_Parameters const &iaparams, Vec const &dx) {\n'
                    '  if (std::get_if<VirtualBond>(&iaparams)) {\n    return std::nullopt;\n  }\n'
                    '  throw BondUnknownTypeError();\n}\n'
                    'auto calc_bonded_three_body_force(Bonded_IA_Parameters const &iaparams, Vec const &vec1, Vec const &vec2) {\n'
                    '  if (auto const *iap = std::get_if<IBMTriel>(&iaparams)) {\n    return iap->forces(vec1, vec2);\n  }\n'
                    '  throw BondUnknownTypeError();\n}\n',
                "src/core/energy_inline.hpp":
                    'auto calc_pair_bonded_energy(Bonded_IA_Parameters const &iaparams, Vec const &dx) {\n'
                    '  if (std::get_if<VirtualBond>(&iaparams)) {\n    return 0.0;\n  }\n'
                    '  throw BondUnknownTypeError();\n}\n'
                    'auto calc_angle_bonded_energy(Bonded_IA_Parameters const &iaparams, Vec const &vec1, Vec const &vec2) {\n'
                    '  if (std::get_if<IBMTriel>(&iaparams)) {\n    return 0.0;\n  }\n'
                    '  throw BondUnknownTypeError();\n}\n',
                "src/script_interface/interactions/BondedInteraction.hpp":
                    'class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n',
                "src/script_interface/interactions/initialize.cpp":
                    '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n',
                "src/python/espressomd/interactions.py":
                    'class BONDED_IA(enum.IntEnum):\n    VIRTUAL_BOND = enum.auto()\n\n'
                    '@script_interface_register\nclass BondedInteractions(ScriptObjectMap):\n',
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
            changed = installer.install(root, header)
            self.assertTrue(changed)
            installer.check(root, header)
            snapshot = {rel: (root / rel).read_text() for rel in files}
            self.assertEqual(installer.install(root, header), [])
            installer.check(root, header)
            self.assertEqual(snapshot, {rel: (root / rel).read_text() for rel in files})

    def test_function_locator_ignores_later_call_sites(self):
        installer_path = ROOT / "simulation/espresso_plugin/install_conservative_spline_bond.py"
        spec = importlib.util.spec_from_file_location("install_conservative_spline_bond_locator", installer_path)
        installer = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(installer)
        text = (
            "inline std::optional<double> calc_pair_bonded_energy(\n"
            "    Bonded_IA_Parameters const &iaparams, Vec const &dx) {\n"
            "  if (std::get_if<VirtualBond>(&iaparams)) return 0.0;\n"
            "  throw BondUnknownTypeError();\n"
            "}\n"
            "inline std::optional<double> calc_bonded_energy() {\n"
            "  return calc_pair_bonded_energy(iaparams, dx);\n"
            "}\n"
        )
        start, end = installer._function_span(text, "calc_pair_bonded_energy")
        body = text[start:end]
        self.assertIn("VirtualBond", body)
        self.assertNotIn("return calc_pair_bonded_energy(iaparams, dx)", body)

    def test_function_locator_handles_multiline_espresso_signature_and_call(self):
        installer_path = ROOT / "simulation/espresso_plugin/install_conservative_spline_bond.py"
        spec = importlib.util.spec_from_file_location("install_conservative_spline_bond_multiline_locator", installer_path)
        installer = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(installer)
        text = (
            "ESPRESSO_ATTR_ALWAYS_INLINE\n"
            "inline std::optional<std::tuple<Vec, Vec, Vec>>\n"
            "calc_bonded_three_body_force(Bonded_IA_Parameters const &iaparams,\n"
            "                             Vec const &vec1, Vec const &vec2) {\n"
            "  throw BondUnknownTypeError();\n"
            "}\n"
            "auto wrapper() { return calc_bonded_three_body_force(iaparams, a, b); }\n"
        )
        start, end = installer._function_span(text, "calc_bonded_three_body_force")
        self.assertIn("BondUnknownTypeError", text[start:end])
        self.assertNotIn("auto wrapper", text[start:end])

    def test_installer_repairs_misplaced_angle_energy_dispatch(self):
        installer_path = ROOT / "simulation/espresso_plugin/install_conservative_spline_bond.py"
        spec = importlib.util.spec_from_file_location("install_conservative_spline_bond_repair", installer_path)
        installer = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(installer)
        addition = (
            "  // MLCG conservative spline bonded interactions\n"
            "  if (auto const *iap = std::get_if<ConservativeSplineAngleBond>(&iaparams)) {\n"
            "    return iap->energy(vec1, vec2);\n"
            "  }\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            energy = Path(tmpdir) / "energy_inline.hpp"
            energy.write_text(
                "auto calc_pair_bonded_energy(Bonded_IA_Parameters const &iaparams, Vec const &dx) {\n"
                + addition
                + "  if (std::get_if<VirtualBond>(&iaparams)) { return 0.0; }\n"
                + "  throw BondUnknownTypeError();\n}\n"
                + "auto calc_angle_bonded_energy(Bonded_IA_Parameters const &iaparams, Vec const &vec1, Vec const &vec2) {\n"
                + "  if (std::get_if<IBMTriel>(&iaparams)) { return 0.0; }\n"
                + "  throw BondUnknownTypeError();\n}\n"
            )
            changed = installer._ensure_dispatch_in_function(
                energy,
                function_name="calc_angle_bonded_energy",
                addition=addition,
                sentinel="std::get_if<ConservativeSplineAngleBond>",
                anchors=("  if (std::get_if<IBMTriel>(&iaparams)) {", "  throw BondUnknownTypeError();\n"),
            )
            self.assertTrue(changed)
            text = energy.read_text()
            a0, a1 = installer._function_span(text, "calc_angle_bonded_energy")
            p0, p1 = installer._function_span(text, "calc_pair_bonded_energy")
            self.assertIn("ConservativeSplineAngleBond", text[a0:a1])
            self.assertNotIn("ConservativeSplineAngleBond", text[p0:p1])

    def test_conservative_spline_files_are_hashed_for_residual_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table = root / "bond_conservative.dat"
            table.write_text("0 0 0\n1 1 2\n")
            priors = root / "cg_priors.json"
            priors.write_text(json.dumps({
                "bonds": [{"type": "conservative_spline", "file": table.name}],
                "angles": [],
                "dihedrals": [],
            }))
            hashes = referenced_prior_artifacts(priors)
            self.assertIn(str(priors.resolve()), hashes)
            self.assertIn(str(table.resolve()), hashes)
            self.assertEqual(len(hashes), 2)


    def test_cpp_kernel_compiles_and_distance_force_matches_energy_gradient(self):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No C++ compiler available")
        plugin = ROOT / "simulation/espresso_plugin"
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / "utils").mkdir()
            (td / "utils/Vector.hpp").write_text(r'''#pragma once
#include <cmath>
namespace Utils {
struct Vector3d {
  double x=0., y=0., z=0.;
  Vector3d() = default;
  Vector3d(double x, double y, double z) : x{x}, y{y}, z{z} {}
  double norm() const { return std::sqrt(x*x+y*y+z*z); }
  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
inline Vector3d operator*(double s, Vector3d const &v) { return {s*v.x,s*v.y,s*v.z}; }
}
''')
            (td / "angle_common.hpp").write_text(r'''#pragma once
#include <utils/Vector.hpp>
#include <algorithm>
#include <cmath>
#include <tuple>
inline double calc_cosine(Utils::Vector3d const &a, Utils::Vector3d const &b, bool) {
  return std::clamp((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm()*b.norm()), -1.0, 1.0);
}
template <class F>
inline std::tuple<Utils::Vector3d,Utils::Vector3d,Utils::Vector3d>
angle_generic_force(Utils::Vector3d const &, Utils::Vector3d const &, F const &, bool) {
  return {};
}
''')
            source = td / "test.cpp"
            source.write_text(r'''#include "conservative_spline_bond.hpp"
#include <cmath>
#include <vector>
int main() {
  std::vector<double> u{1.0, 0.0, 1.0};
  std::vector<double> du{-2.0, 0.0, 2.0};
  ConservativeSplineDistanceBond b(0.5, 1.5, u, du);
  auto energy = [&](double r) { return *b.energy(Utils::Vector3d{r,0,0}); };
  double const r = 0.8, h = 1.e-7;
  double const minus_dU = -(energy(r+h)-energy(r-h))/(2*h);
  auto f = b.force(Utils::Vector3d{r,0,0});
  if (!f || std::abs((*f)[0]-minus_dU) > 1.e-7) return 1;
  if (b.force(Utils::Vector3d{1.5,0,0}).has_value()) return 2;
  if (b.energy(Utils::Vector3d{1.5,0,0}).has_value()) return 3;
  ConservativeSplineAngleBond a(0.0, std::acos(-1.0), u, du);
  auto e = a.energy(Utils::Vector3d{1,0,0}, Utils::Vector3d{0,1,0});
  if (!std::isfinite(e)) return 4;
  return 0;
}
''')
            exe = td / "test"
            subprocess.run(
                [compiler, "-std=c++20", "-O2", "-I", str(td), "-I", str(plugin), str(source), "-o", str(exe)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(exe)], check=True)

    def test_cpp_kernel_contains_single_energy_source_and_derivative(self):
        source = (ROOT / "simulation/espresso_plugin/conservative_spline_bond.hpp").read_text()
        self.assertIn("std::pair<double, double> evaluate(double q) const", source)
        self.assertIn("auto const dU_dr = spline.evaluate(dist).second", source)
        self.assertIn("return spline.evaluate(dist).first", source)
        self.assertIn("auto const dU_dphi = spline.evaluate(phi).second", source)
        self.assertNotIn("force_nodes", source)


if __name__ == "__main__":
    unittest.main()
