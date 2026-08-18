#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "simulation" / "espresso_plugin"


def _load_installer(filename: str, module_name: str):
    path = PLUGIN / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MORSE = _load_installer("install_analytic_morse_bond.py", "mlcg_install_morse_integration")
SPLINE = _load_installer("install_conservative_spline_bond.py", "mlcg_install_spline_integration")


class EspressoPluginInstallerIntegrationTests(unittest.TestCase):
    def _make_espresso_tree(self, root: Path) -> list[str]:
        files = {
            "src/core/bonded_interactions/bonded_interaction_data.hpp": (
                '#include "harmonic.hpp"\n'
                "using Bonded_IA_Parameters = std::variant<FeneBond, HarmonicBond, "
                "QuarticBond, OifGlobalForcesBond, OifLocalForcesBond, VirtualBond>;\n"
            ),
            "src/core/forces_inline.hpp": (
                "auto calc_bond_pair_force(Bonded_IA_Parameters const &iaparams, Vec const &dx) {\n"
                "  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n"
                "    return iap->force(dx);\n"
                "  }\n"
                "  if (std::get_if<VirtualBond>(&iaparams)) {\n"
                "    return std::nullopt;\n"
                "  }\n"
                "  throw BondUnknownTypeError();\n"
                "}\n"
                "auto calc_bonded_three_body_force(Bonded_IA_Parameters const &iaparams, Vec const &vec1, Vec const &vec2) {\n"
                "  if (auto const *iap = std::get_if<IBMTriel>(&iaparams)) {\n"
                "    return iap->forces(vec1, vec2);\n"
                "  }\n"
                "  throw BondUnknownTypeError();\n"
                "}\n"
                "auto calc_bonded_dihedral_force(Bonded_IA_Parameters const &iaparams, Vec const &v12, Vec const &v23, Vec const &v34) {\n"
                "  throw BondUnknownTypeError();\n"
                "}\n"
            ),
            "src/core/energy_inline.hpp": (
                "auto calc_pair_bonded_energy(Bonded_IA_Parameters const &iaparams, Vec const &dx) {\n"
                "  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n"
                "    return iap->energy(dx);\n"
                "  }\n"
                "  if (std::get_if<VirtualBond>(&iaparams)) {\n"
                "    return 0.0;\n"
                "  }\n"
                "  throw BondUnknownTypeError();\n"
                "}\n"
                "auto calc_angle_bonded_energy(Bonded_IA_Parameters const &iaparams, Vec const &vec1, Vec const &vec2) {\n"
                "  if (std::get_if<IBMTriel>(&iaparams)) {\n"
                "    return 0.0;\n"
                "  }\n"
                "  throw BondUnknownTypeError();\n"
                "}\n"
                "auto calc_dihedral_bonded_energy(Bonded_IA_Parameters const &iaparams, Vec const &v12, Vec const &v23, Vec const &v34) {\n"
                "  throw BondUnknownTypeError();\n"
                "}\n"
            ),
            "src/script_interface/interactions/BondedInteraction.hpp": (
                "class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {\n"
            ),
            "src/script_interface/interactions/initialize.cpp": (
                '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n'
            ),
            "src/python/espressomd/interactions.py": (
                "class BONDED_IA(enum.IntEnum):\n"
                "    VIRTUAL_BOND = enum.auto()\n\n"
                "@script_interface_register\n"
                "class BondedInteractions(ScriptObjectMap):\n"
            ),
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return list(files)

    @staticmethod
    def _variant_custom_types(root: Path) -> list[str]:
        text = (root / "src/core/bonded_interactions/bonded_interaction_data.hpp").read_text()
        match = re.search(
            r"using Bonded_IA_Parameters\s*=\s*std::variant<(.*?)>;",
            text,
            flags=re.DOTALL,
        )
        assert match is not None
        types = [part.strip() for part in match.group(1).split(",")]
        virtual = types.index("VirtualBond")
        return types[virtual + 1 :]

    @staticmethod
    def _python_custom_types(root: Path) -> list[str]:
        text = (root / "src/python/espressomd/interactions.py").read_text()
        mapping = {
            "MORSE_BOND": "MorseBond",
            "CONSERVATIVE_SPLINE_DISTANCE": "ConservativeSplineDistanceBond",
            "CONSERVATIVE_SPLINE_ANGLE": "ConservativeSplineAngleBond",
            "CONSERVATIVE_SPLINE_DIHEDRAL": "ConservativeSplineDihedralBond",
        }
        names = [
            match.group(1)
            for match in re.finditer(
                r"^\s+([A-Z][A-Z0-9_]*)\s*=\s*enum\.auto\(\)",
                text,
                flags=re.MULTILINE,
            )
        ]
        virtual = names.index("VIRTUAL_BOND")
        return [mapping[name] for name in names[virtual + 1 :] if name in mapping]

    def _assert_joint_install_is_stable(self, order: tuple[str, ...]) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tracked = self._make_espresso_tree(root)

            for name in order:
                if name == "morse":
                    MORSE.install(root, PLUGIN / "morse_bond.hpp")
                else:
                    SPLINE.install(root, PLUGIN / "conservative_spline_bond.hpp")

            MORSE.check(root, PLUGIN / "morse_bond.hpp")
            SPLINE.check(root, PLUGIN / "conservative_spline_bond.hpp")
            self.assertEqual(self._variant_custom_types(root), self._python_custom_types(root))

            bond_data = (root / "src/core/bonded_interactions/bonded_interaction_data.hpp").read_text()
            self.assertEqual(bond_data.count('#include "morse_bond.hpp"'), 1)
            self.assertEqual(bond_data.count('#include "conservative_spline_bond.hpp"'), 1)
            init = (root / "src/script_interface/interactions/initialize.cpp").read_text()
            self.assertEqual(init.count('Interactions::MorseBond'), 1)
            self.assertEqual(init.count('Interactions::ConservativeSplineDistanceBond'), 1)
            self.assertEqual(init.count('Interactions::ConservativeSplineAngleBond'), 1)
            self.assertEqual(init.count('Interactions::ConservativeSplineDihedralBond'), 1)

            snapshot = {
                rel: (root / rel).read_bytes()
                for rel in tracked
            }
            self.assertEqual(MORSE.install(root, PLUGIN / "morse_bond.hpp"), [])
            self.assertEqual(SPLINE.install(root, PLUGIN / "conservative_spline_bond.hpp"), [])
            self.assertEqual(
                snapshot,
                {rel: (root / rel).read_bytes() for rel in tracked},
            )

    def test_morse_then_conservative_spline_is_jointly_idempotent(self):
        self._assert_joint_install_is_stable(("morse", "spline"))

    def test_conservative_spline_then_morse_is_jointly_idempotent(self):
        self._assert_joint_install_is_stable(("spline", "morse"))

    def test_alternating_installers_remains_jointly_idempotent(self):
        self._assert_joint_install_is_stable(("morse", "spline", "morse", "spline"))


if __name__ == "__main__":
    unittest.main()
