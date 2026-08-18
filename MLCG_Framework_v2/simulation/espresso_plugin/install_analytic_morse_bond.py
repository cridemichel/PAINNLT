#!/usr/bin/env python3
"""Install the MLCG analytic Morse bonded interaction into ESPResSo 5.0.x."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _read(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required ESPResSo source file not found: {path}")
    return path.read_text()


def _replace_once(path: Path, old: str, new: str) -> bool:
    text = _read(path)
    if new in text:
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one installation anchor in {path}, found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new, 1))
    return True


def _insert_before_once(path: Path, anchor: str, addition: str, sentinel: str) -> bool:
    text = _read(path)
    if sentinel in text:
        return False
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one installation anchor in {path}, found {count}: {anchor!r}"
        )
    path.write_text(text.replace(anchor, addition + anchor, 1))
    return True


def _patch_bonded_variant(path: Path) -> bool:
    """Ensure MorseBond is present in Bonded_IA_Parameters.

    Other MLCG installers may append additional bonded interaction types after
    MorseBond.  Therefore idempotency must be decided from the variant contents
    rather than from an exact tail such as ``VirtualBond, MorseBond>``.
    """
    text = _read(path)
    match = re.search(
        r"using Bonded_IA_Parameters\s*=\s*std::variant<.*?>;",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"Could not locate Bonded_IA_Parameters variant in {path}")

    block = match.group(0)
    if re.search(r"\bMorseBond\b", block):
        return False
    if block.count("VirtualBond") != 1:
        raise RuntimeError(
            f"Could not uniquely locate VirtualBond in Bonded_IA_Parameters: {path}"
        )

    replacement = block.replace("VirtualBond", "VirtualBond, MorseBond", 1)
    path.write_text(text[: match.start()] + replacement + text[match.end() :])
    return True


def _bonded_variant_contains(path: Path, type_name: str) -> bool:
    text = _read(path)
    match = re.search(
        r"using Bonded_IA_Parameters\s*=\s*std::variant<.*?>;",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        return False
    return re.search(rf"\b{re.escape(type_name)}\b", match.group(0)) is not None


def _ensure_script_registration(path: Path) -> bool:
    """Ensure exactly one ScriptInterface registration for MorseBond.

    Other MLCG installers also insert registrations after QuarticBond.  A
    previous implementation checked for the exact adjacent text
    ``QuarticBond`` + ``MorseBond``; once another installer inserted lines
    between them, rerunning this installer could register MorseBond twice.
    """
    text = _read(path)
    pattern = re.compile(
        r'^[ \t]*om->register_new<MorseBond>\("Interactions::MorseBond"\);[^\n]*(?:\n|$)',
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) == 1:
        return False

    if len(matches) > 1:
        # Preserve the first registration exactly as written and remove later
        # duplicates.  Work backwards so match offsets remain valid.
        repaired = text
        for match in reversed(matches[1:]):
            repaired = repaired[: match.start()] + repaired[match.end() :]
        path.write_text(repaired)
        return True

    anchor = '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n'
    if text.count(anchor) != 1:
        raise RuntimeError(
            f"Could not uniquely locate ScriptInterface registration anchor in {path}"
        )
    registration = (
        '  om->register_new<MorseBond>("Interactions::MorseBond"); '
        '// MLCG analytic MorseBond\n'
    )
    path.write_text(text.replace(anchor, anchor + registration, 1))
    return True


def _script_registration_count(path: Path) -> int:
    return len(
        re.findall(
            r'om->register_new<MorseBond>\("Interactions::MorseBond"\);',
            _read(path),
        )
    )


def _ensure_python_enum_entry(path: Path) -> bool:
    """Ensure exactly one MORSE_BOND member in the BONDED_IA enum.

    Other MLCG installers also append enum members after ``VIRTUAL_BOND``.
    Checking for the exact adjacent ``VIRTUAL_BOND`` + ``MORSE_BOND`` text is
    therefore not idempotent: a later installer can move MorseBond away from
    the anchor, causing a subsequent Morse install to add the enum member a
    second time.
    """
    text = _read(path)
    pattern = re.compile(
        r'^[ \t]+MORSE_BOND\s*=\s*enum\.auto\(\)[^\n]*(?:\n|$)',
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) == 1:
        return False

    if len(matches) > 1:
        repaired = text
        for match in reversed(matches[1:]):
            repaired = repaired[: match.start()] + repaired[match.end() :]
        path.write_text(repaired)
        return True

    anchor = "    VIRTUAL_BOND = enum.auto()\n"
    if text.count(anchor) != 1:
        raise RuntimeError(
            f"Could not uniquely locate BONDED_IA VIRTUAL_BOND anchor in {path}"
        )
    entry = "    MORSE_BOND = enum.auto()  # MLCG analytic MorseBond\n"
    path.write_text(text.replace(anchor, anchor + entry, 1))
    return True


def _python_enum_count(path: Path) -> int:
    return len(
        re.findall(
            r'^\s+MORSE_BOND\s*=\s*enum\.auto\(\)',
            _read(path),
            flags=re.MULTILINE,
        )
    )


def required_paths(espresso_root: Path) -> dict[str, Path]:
    return {
        "bond_data": espresso_root / "src/core/bonded_interactions/bonded_interaction_data.hpp",
        "forces": espresso_root / "src/core/forces_inline.hpp",
        "energy": espresso_root / "src/core/energy_inline.hpp",
        "script_header": espresso_root / "src/script_interface/interactions/BondedInteraction.hpp",
        "script_init": espresso_root / "src/script_interface/interactions/initialize.cpp",
        "python": espresso_root / "src/python/espressomd/interactions.py",
        "morse_header": espresso_root / "src/core/bonded_interactions/morse_bond.hpp",
    }


def install(espresso_root: Path, source_header: Path) -> list[str]:
    paths = required_paths(espresso_root)
    for key, path in paths.items():
        if key != "morse_header":
            _read(path)

    source = _read(source_header)
    paths["morse_header"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["morse_header"].is_file() or paths["morse_header"].read_text() != source:
        shutil.copyfile(source_header, paths["morse_header"])

    changed: list[str] = []

    if _replace_once(
        paths["bond_data"],
        '#include "harmonic.hpp"\n',
        '#include "harmonic.hpp"\n#include "morse_bond.hpp" // MLCG analytic MorseBond\n',
    ):
        changed.append("bonded_interaction_data.hpp include")

    if _patch_bonded_variant(paths["bond_data"]):
        changed.append("Bonded_IA_Parameters")

    force_anchor = (
        "  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n"
        "    return iap->force(dx);\n"
        "  }\n"
    )
    if _replace_once(
        paths["forces"],
        force_anchor,
        force_anchor
        + "  // MLCG analytic MorseBond\n"
          "  if (auto const *iap = std::get_if<MorseBond>(&iaparams)) {\n"
          "    return iap->force(dx);\n"
          "  }\n",
    ):
        changed.append("forces_inline.hpp")

    energy_anchor = (
        "  if (auto const *iap = std::get_if<QuarticBond>(&iaparams)) {\n"
        "    return iap->energy(dx);\n"
        "  }\n"
    )
    if _replace_once(
        paths["energy"],
        energy_anchor,
        energy_anchor
        + "  // MLCG analytic MorseBond\n"
          "  if (auto const *iap = std::get_if<MorseBond>(&iaparams)) {\n"
          "    return iap->energy(dx);\n"
          "  }\n",
    ):
        changed.append("energy_inline.hpp")

    script_class = '''// MLCG analytic MorseBond
class MorseBond : public BondedInteractionImpl<::MorseBond> {
 public:
  MorseBond() {
    add_parameters({
        {"D", AutoParameter::read_only, [this]() { return get_struct().D; }},
        {"a", AutoParameter::read_only, [this]() { return get_struct().a; }},
        {"r_0", AutoParameter::read_only, [this]() { return get_struct().r0; }},
        {"r_cut", AutoParameter::read_only,
         [this]() { return get_struct().r_cut; }},
    });
  }

 private:
  void construct_bond(VariantMap const &params) override {
    m_bonded_ia =
        std::make_shared<::Bonded_IA_Parameters>(CoreBondedInteraction(
            get_value<double>(params, "D"), get_value<double>(params, "a"),
            get_value<double>(params, "r_0"),
            get_value<double>(params, "r_cut")));
  }
};

'''
    if _insert_before_once(
        paths["script_header"],
        "class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {",
        script_class,
        "// MLCG analytic MorseBond",
    ):
        changed.append("BondedInteraction.hpp")

    if _ensure_script_registration(paths["script_init"]):
        changed.append("initialize.cpp")

    if _ensure_python_enum_entry(paths["python"]):
        changed.append("BONDED_IA")

    python_class = '''# MLCG analytic MorseBond Python class
@script_interface_register
class MorseBond(BondedInteraction):
    """Conservative bonded Morse potential.

    U(r) = D * (1 - exp(-a * (r-r_0)))**2
    """

    _so_name = "Interactions::MorseBond"
    _type_number = BONDED_IA.MORSE_BOND

    def get_default_params(self):
        return {"r_cut": 15.0}


'''
    if _insert_before_once(
        paths["python"],
        "@script_interface_register\nclass BondedInteractions(ScriptObjectMap):",
        python_class,
        "# MLCG analytic MorseBond Python class",
    ):
        changed.append("interactions.py class")

    return changed


def check(espresso_root: Path, source_header: Path) -> None:
    paths = required_paths(espresso_root)
    checks = {
        "morse header": paths["morse_header"].is_file()
        and paths["morse_header"].read_text() == _read(source_header),
        "variant": _bonded_variant_contains(paths["bond_data"], "MorseBond"),
        "force dispatch": "std::get_if<MorseBond>" in _read(paths["forces"]),
        "energy dispatch": "std::get_if<MorseBond>" in _read(paths["energy"]),
        "script interface": "BondedInteractionImpl<::MorseBond>" in _read(paths["script_header"]),
        "script registration": _script_registration_count(paths["script_init"]) == 1,
        "python enum": _python_enum_count(paths["python"]) == 1,
        "python class": "class MorseBond(BondedInteraction):" in _read(paths["python"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError("Analytic MorseBond installation incomplete: " + ", ".join(failed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--espresso-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source_header = Path(__file__).resolve().with_name("morse_bond.hpp")
    espresso_root = args.espresso_root.expanduser().resolve()
    if args.check:
        check(espresso_root, source_header)
        print("[PASS] Analytic MorseBond is installed consistently.")
        return

    changed = install(espresso_root, source_header)
    check(espresso_root, source_header)
    if changed:
        print("[PASS] Installed analytic MorseBond: " + ", ".join(changed))
    else:
        print("[PASS] Analytic MorseBond already installed; no source edits needed.")


if __name__ == "__main__":
    main()
