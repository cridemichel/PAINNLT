import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "simulation" / "espresso_plugin"


def load_installer():
    path = PLUGIN / "install_analytic_morse_bond.py"
    spec = importlib.util.spec_from_file_location("install_analytic_morse_bond", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalyticMorseBondTests(unittest.TestCase):
    def test_cpp_kernel_is_energy_force_consistent(self):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No C++ compiler available")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
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
inline Vector3d operator*(double s, Vector3d const &v) {
  return {s*v.x, s*v.y, s*v.z};
}
}
''')
            source = td / "test.cpp"
            source.write_text(r'''#include "morse_bond.hpp"
#include <cmath>
int main() {
  MorseBond b(8.5, 3.2, 0.37, 15.0);
  auto energy = [&](double r) { return *b.energy(Utils::Vector3d{r,0,0}); };
  double const r = 0.51;
  double const h = 1e-6;
  double const minus_dU = -(energy(r+h)-energy(r-h))/(2*h);
  auto const f = b.force(Utils::Vector3d{r,0,0});
  if (!f || std::abs((*f)[0] - minus_dU) > 1e-7) return 1;
  if (std::abs(energy(0.37)) > 1e-14) return 2;
  if (b.force(Utils::Vector3d{15.1,0,0}).has_value()) return 3;
  if (b.energy(Utils::Vector3d{15.1,0,0}).has_value()) return 4;
  if (b.force(Utils::Vector3d{15.0,0,0}).has_value()) return 5;
  if (b.energy(Utils::Vector3d{15.0,0,0}).has_value()) return 6;
  if (b.force(Utils::Vector3d{0.0,0,0}).has_value()) return 7;
  if (b.energy(Utils::Vector3d{0.0,0,0}).has_value()) return 8;
  return 0;
}
''')
            exe = td / "test"
            subprocess.run(
                [compiler, "-std=c++20", "-O2", "-I", str(td), "-I", str(PLUGIN), str(source), "-o", str(exe)],
                check=True, capture_output=True, text=True)
            subprocess.run([str(exe)], check=True)

    def test_installer_is_idempotent_on_espresso_5_style_tree(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "src/core/bonded_interactions/bonded_interaction_data.hpp": '#include "harmonic.hpp"\nusing Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n',
                "src/core/forces_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->force(dx);\n  }\n',
                "src/core/energy_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->energy(dx);\n  }\n',
                "src/script_interface/interactions/BondedInteraction.hpp": 'class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n',
                "src/script_interface/interactions/initialize.cpp": '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n',
                "src/python/espressomd/interactions.py": 'class BONDED_IA(enum.IntEnum):\n    VIRTUAL_BOND = enum.auto()\n\n@script_interface_register\nclass BondedInteractions(ScriptObjectMap):\n',
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
            first = installer.install(root, PLUGIN / "morse_bond.hpp")
            self.assertTrue(first)
            installer.check(root, PLUGIN / "morse_bond.hpp")
            snapshot = {rel: (root / rel).read_text() for rel in files}
            second = installer.install(root, PLUGIN / "morse_bond.hpp")
            self.assertEqual(second, [])
            self.assertEqual(snapshot, {rel: (root / rel).read_text() for rel in files})

    def test_installer_remains_idempotent_after_other_bond_types_extend_variant(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "src/core/bonded_interactions/bonded_interaction_data.hpp": '#include "harmonic.hpp"\nusing Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n',
                "src/core/forces_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->force(dx);\n  }\n',
                "src/core/energy_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->energy(dx);\n  }\n',
                "src/script_interface/interactions/BondedInteraction.hpp": 'class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n',
                "src/script_interface/interactions/initialize.cpp": '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n',
                "src/python/espressomd/interactions.py": 'class BONDED_IA(enum.IntEnum):\n    VIRTUAL_BOND = enum.auto()\n\n@script_interface_register\nclass BondedInteractions(ScriptObjectMap):\n',
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)

            installer.install(root, PLUGIN / "morse_bond.hpp")
            bond_data = root / "src/core/bonded_interactions/bonded_interaction_data.hpp"
            text = bond_data.read_text()
            text = text.replace(
                "VirtualBond, MorseBond>",
                "VirtualBond, MorseBond, ConservativeSplineDistanceBond, "
                "ConservativeSplineAngleBond, ConservativeSplineDihedralBond>",
                1,
            )
            bond_data.write_text(text)

            snapshot = {rel: (root / rel).read_text() for rel in files}
            second = installer.install(root, PLUGIN / "morse_bond.hpp")
            self.assertEqual(second, [])
            installer.check(root, PLUGIN / "morse_bond.hpp")
            self.assertEqual(snapshot, {rel: (root / rel).read_text() for rel in files})

    def test_installer_remains_idempotent_after_other_registrations_follow_quartic(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "src/core/bonded_interactions/bonded_interaction_data.hpp": '#include "harmonic.hpp"\nusing Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n',
                "src/core/forces_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->force(dx);\n  }\n',
                "src/core/energy_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->energy(dx);\n  }\n',
                "src/script_interface/interactions/BondedInteraction.hpp": 'class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n',
                "src/script_interface/interactions/initialize.cpp": '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n',
                "src/python/espressomd/interactions.py": 'class BONDED_IA(enum.IntEnum):\n    VIRTUAL_BOND = enum.auto()\n\n@script_interface_register\nclass BondedInteractions(ScriptObjectMap):\n',
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)

            installer.install(root, PLUGIN / "morse_bond.hpp")
            init = root / "src/script_interface/interactions/initialize.cpp"
            quartic = '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n'
            conservative = (
                '  om->register_new<ConservativeSplineDistanceBond>("Interactions::ConservativeSplineDistanceBond");\n'
                '  om->register_new<ConservativeSplineAngleBond>("Interactions::ConservativeSplineAngleBond");\n'
            )
            init.write_text(init.read_text().replace(quartic, quartic + conservative, 1))

            snapshot = init.read_text()
            changed = installer.install(root, PLUGIN / "morse_bond.hpp")
            self.assertNotIn("initialize.cpp", changed)
            self.assertEqual(init.read_text(), snapshot)
            self.assertEqual(installer._script_registration_count(init), 1)
            installer.check(root, PLUGIN / "morse_bond.hpp")

    def test_installer_repairs_duplicate_morse_script_registration(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "src/core/bonded_interactions/bonded_interaction_data.hpp": '#include "harmonic.hpp"\nusing Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n',
                "src/core/forces_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->force(dx);\n  }\n',
                "src/core/energy_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->energy(dx);\n  }\n',
                "src/script_interface/interactions/BondedInteraction.hpp": 'class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n',
                "src/script_interface/interactions/initialize.cpp": '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n',
                "src/python/espressomd/interactions.py": 'class BONDED_IA(enum.IntEnum):\n    VIRTUAL_BOND = enum.auto()\n\n@script_interface_register\nclass BondedInteractions(ScriptObjectMap):\n',
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)

            installer.install(root, PLUGIN / "morse_bond.hpp")
            init = root / "src/script_interface/interactions/initialize.cpp"
            registration = (
                '  om->register_new<MorseBond>("Interactions::MorseBond"); '
                '// MLCG analytic MorseBond\n'
            )
            init.write_text(init.read_text() + registration)
            self.assertEqual(installer._script_registration_count(init), 2)

            changed = installer.install(root, PLUGIN / "morse_bond.hpp")
            self.assertIn("initialize.cpp", changed)
            self.assertEqual(installer._script_registration_count(init), 1)
            installer.check(root, PLUGIN / "morse_bond.hpp")

    def test_installer_remains_idempotent_after_other_python_enum_members_follow_virtual(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "src/core/bonded_interactions/bonded_interaction_data.hpp": '#include "harmonic.hpp"\nusing Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n',
                "src/core/forces_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->force(dx);\n  }\n',
                "src/core/energy_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->energy(dx);\n  }\n',
                "src/script_interface/interactions/BondedInteraction.hpp": 'class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n',
                "src/script_interface/interactions/initialize.cpp": '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n',
                "src/python/espressomd/interactions.py": 'class BONDED_IA(enum.IntEnum):\n    VIRTUAL_BOND = enum.auto()\n\n@script_interface_register\nclass BondedInteractions(ScriptObjectMap):\n',
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)

            installer.install(root, PLUGIN / "morse_bond.hpp")
            python_file = root / "src/python/espressomd/interactions.py"
            virtual = "    VIRTUAL_BOND = enum.auto()\n"
            conservative = (
                "    CONSERVATIVE_SPLINE_DISTANCE_BOND = enum.auto()\n"
                "    CONSERVATIVE_SPLINE_ANGLE_BOND = enum.auto()\n"
                "    CONSERVATIVE_SPLINE_DIHEDRAL_BOND = enum.auto()\n"
            )
            python_file.write_text(
                python_file.read_text().replace(virtual, virtual + conservative, 1)
            )

            snapshot = python_file.read_text()
            changed = installer.install(root, PLUGIN / "morse_bond.hpp")
            self.assertNotIn("BONDED_IA", changed)
            self.assertEqual(python_file.read_text(), snapshot)
            self.assertEqual(installer._python_enum_count(python_file), 1)
            installer.check(root, PLUGIN / "morse_bond.hpp")

    def test_installer_repairs_duplicate_morse_python_enum_member(self):
        installer = load_installer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "src/core/bonded_interactions/bonded_interaction_data.hpp": '#include "harmonic.hpp"\nusing Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n',
                "src/core/forces_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->force(dx);\n  }\n',
                "src/core/energy_inline.hpp": '  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n    return iap->energy(dx);\n  }\n',
                "src/script_interface/interactions/BondedInteraction.hpp": 'class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n',
                "src/script_interface/interactions/initialize.cpp": '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n',
                "src/python/espressomd/interactions.py": 'class BONDED_IA(enum.IntEnum):\n    VIRTUAL_BOND = enum.auto()\n\n@script_interface_register\nclass BondedInteractions(ScriptObjectMap):\n',
            }
            for rel, data in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)

            installer.install(root, PLUGIN / "morse_bond.hpp")
            python_file = root / "src/python/espressomd/interactions.py"
            entry = "    MORSE_BOND = enum.auto()  # MLCG analytic MorseBond\n"
            python_file.write_text(python_file.read_text().replace(entry, entry + entry, 1))
            self.assertEqual(installer._python_enum_count(python_file), 2)

            changed = installer.install(root, PLUGIN / "morse_bond.hpp")
            self.assertIn("BONDED_IA", changed)
            self.assertEqual(installer._python_enum_count(python_file), 1)
            installer.check(root, PLUGIN / "morse_bond.hpp")

    def test_runtime_factory_requires_extension_and_maps_parameters(self):
        import sys
        sys.path.insert(0, str(ROOT / "simulation"))
        try:
            from espresso_interactions import make_analytic_morse_bond
        finally:
            sys.path.pop(0)

        class FakeMorse:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class WithMorse:
            MorseBond = FakeMorse

        bond = make_analytic_morse_bond(WithMorse, {"D": 4.0, "a": 5.0, "r0": 0.3})
        self.assertEqual(bond.kwargs, {"D": 4.0, "a": 5.0, "r_0": 0.3, "r_cut": 15.0})
        with self.assertRaisesRegex(RuntimeError, "no espressomd.interactions.MorseBond"):
            make_analytic_morse_bond(object(), {"D": 4.0, "a": 5.0, "r0": 0.3})


if __name__ == "__main__":
    unittest.main()
