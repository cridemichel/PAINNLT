#!/usr/bin/env python3
"""Install MLCG conservative bond/angle/dihedral spline interactions into ESPResSo 5.0.x.

The installer is idempotent and intentionally patches only the minimal core,
ScriptInterface and Python dispatch points required for three new bonded types:
``ConservativeSplineDistance``, ``ConservativeSplineAngle`` and
``ConservativeSplineDihedral``.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SENTINEL = "MLCG conservative spline bonded interactions"


def _read(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required ESPResSo source file not found: {path}")
    return path.read_text()


def required_paths(root: Path) -> dict[str, Path]:
    return {
        "bond_data": root / "src/core/bonded_interactions/bonded_interaction_data.hpp",
        "forces": root / "src/core/forces_inline.hpp",
        "energy": root / "src/core/energy_inline.hpp",
        "script_header": root / "src/script_interface/interactions/BondedInteraction.hpp",
        "script_init": root / "src/script_interface/interactions/initialize.cpp",
        "python": root / "src/python/espressomd/interactions.py",
        "header": root / "src/core/bonded_interactions/conservative_spline_bond.hpp",
    }


def _replace_once(path: Path, old: str, new: str) -> bool:
    text = _read(path)
    if new in text:
        return False
    if text.count(old) != 1:
        raise RuntimeError(f"Could not find unique ESPResSo installation anchor in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1))
    return True


def _ensure_include_once(path: Path, include_line: str, anchor: str) -> bool:
    """Ensure exactly one generated include independently of installer order."""
    text = _read(path)
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line == include_line]
    if len(matches) == 1:
        return False
    if len(matches) > 1:
        keep = matches[0]
        lines = [line for i, line in enumerate(lines) if line != include_line or i == keep]
        path.write_text("".join(lines))
        return True
    if text.count(anchor) != 1:
        raise RuntimeError(f"Could not find unique ESPResSo include anchor in {path}: {anchor!r}")
    path.write_text(text.replace(anchor, anchor + include_line, 1))
    return True


def _insert_before(path: Path, anchor: str, addition: str, sentinel: str) -> bool:
    text = _read(path)
    if sentinel in text:
        return False
    if text.count(anchor) != 1:
        raise RuntimeError(f"Could not find unique ESPResSo installation anchor in {path}: {anchor!r}")
    path.write_text(text.replace(anchor, addition + anchor, 1))
    return True


def _matching_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    """Return the position of the delimiter matching ``text[start]``."""
    if start < 0 or start >= len(text) or text[start] != opening:
        raise RuntimeError(f"Expected {opening!r} at position {start}")
    depth = 0
    for pos in range(start, len(text)):
        char = text[pos]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return pos
    raise RuntimeError(f"Unbalanced {opening}{closing} delimiters")


def _function_span(text: str, function_name: str) -> tuple[int, int]:
    """Return ``[start, end)`` for a C++ function definition.

    ESPResSo headers naturally contain both a function definition and later
    call sites with the same identifier (for example
    ``calc_pair_bonded_energy`` is called by ``calc_bonded_energy``).  A plain
    string-count therefore cannot identify the definition.  We instead scan
    identifier occurrences, match the argument list and accept only an
    occurrence whose closing parenthesis is followed by a function body.
    """
    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    candidates: list[tuple[int, int]] = []
    for match in pattern.finditer(text):
        open_paren = text.find("(", match.start(), match.end())
        close_paren = _matching_delimiter(text, open_paren, "(", ")")
        pos = close_paren + 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            # This is a call site (normally followed by ';', ')' or ',') or
            # a declaration without a body, not the inline definition.
            continue
        close_brace = _matching_delimiter(text, pos, "{", "}")
        candidates.append((match.start(), close_brace + 1))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely locate C++ function definition {function_name}: "
            f"found {len(candidates)} bodies"
        )
    return candidates[0]


def _ensure_dispatch_in_function(
    path: Path,
    *,
    function_name: str,
    addition: str,
    sentinel: str,
    anchors: tuple[str, ...],
) -> bool:
    """Ensure one generated dispatch block lives inside the requested function.

    Older installer revisions could place the angle-energy block in the wrong
    function. Exact generated blocks outside ``function_name`` are removed so
    rerunning the installer repairs an already-patched ESPResSo tree.
    """
    text = _read(path)
    fstart, fend = _function_span(text, function_name)
    hits = [m.start() for m in re.finditer(re.escape(addition), text)]
    if len(hits) == 1 and fstart <= hits[0] < fend:
        return False

    # Repair any exact block emitted by a previous MLCG installer revision.
    if hits:
        text = text.replace(addition, "")
        fstart, fend = _function_span(text, function_name)

    body = text[fstart:fend]
    if sentinel in body:
        raise RuntimeError(
            f"Found {sentinel!r} inside {function_name}, but not in the expected generated block"
        )
    anchor = next((candidate for candidate in anchors if body.count(candidate) == 1), None)
    if anchor is None:
        raise RuntimeError(
            f"Could not locate a unique dispatch anchor inside {function_name} in {path}"
        )
    rel = body.index(anchor)
    pos = fstart + rel
    path.write_text(text[:pos] + addition + text[pos:])
    return True


def _dispatch_is_in_function(path: Path, function_name: str, needle: str) -> bool:
    text = _read(path)
    fstart, fend = _function_span(text, function_name)
    return needle in text[fstart:fend] and needle not in (text[:fstart] + text[fend:])


def _patch_variant(path: Path) -> bool:
    text = _read(path)
    if "ConservativeSplineDihedralBond" in text:
        return False
    match = re.search(r"using Bonded_IA_Parameters\s*=\s*std::variant<.*?>;", text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not locate Bonded_IA_Parameters variant in {path}")
    block = match.group(0)
    if "ConservativeSplineAngleBond" in block:
        replacement = block.replace(
            "ConservativeSplineAngleBond",
            "ConservativeSplineAngleBond, ConservativeSplineDihedralBond",
            1,
        )
    else:
        if block.count("VirtualBond") != 1:
            raise RuntimeError(f"Could not uniquely locate VirtualBond in Bonded_IA_Parameters: {path}")
        replacement = block.replace(
            "VirtualBond",
            "VirtualBond, ConservativeSplineDistanceBond, ConservativeSplineAngleBond, ConservativeSplineDihedralBond",
            1,
        )
    path.write_text(text[:match.start()] + replacement + text[match.end():])
    return True


def _patch_python_enum(path: Path) -> bool:
    text = _read(path)
    if "CONSERVATIVE_SPLINE_DIHEDRAL" in text:
        return False
    if "CONSERVATIVE_SPLINE_ANGLE" in text:
        anchor = "    CONSERVATIVE_SPLINE_ANGLE = enum.auto()  # MLCG conservative spline\n"
        if text.count(anchor) != 1:
            raise RuntimeError(f"Could not uniquely locate conservative angle enum in {path}")
        path.write_text(text.replace(
            anchor,
            anchor + "    CONSERVATIVE_SPLINE_DIHEDRAL = enum.auto()  # MLCG conservative spline\n",
            1,
        ))
        return True
    anchor = "    VIRTUAL_BOND = enum.auto()\n"
    if text.count(anchor) != 1:
        raise RuntimeError(f"Could not locate BONDED_IA.VIRTUAL_BOND in {path}")
    addition = (
        anchor
        + "    CONSERVATIVE_SPLINE_DISTANCE = enum.auto()  # MLCG conservative spline\n"
        + "    CONSERVATIVE_SPLINE_ANGLE = enum.auto()  # MLCG conservative spline\n"
        + "    CONSERVATIVE_SPLINE_DIHEDRAL = enum.auto()  # MLCG conservative spline\n"
    )
    path.write_text(text.replace(anchor, addition, 1))
    return True


def _ensure_registration(path: Path) -> bool:
    text = _read(path)
    needle = 'Interactions::ConservativeSplineDihedralBond'
    if needle in text:
        return False
    angle_line = '  om->register_new<ConservativeSplineAngleBond>("Interactions::ConservativeSplineAngleBond"); // MLCG conservative spline\n'
    dihedral_line = '  om->register_new<ConservativeSplineDihedralBond>("Interactions::ConservativeSplineDihedralBond"); // MLCG conservative spline\n'
    if angle_line in text:
        path.write_text(text.replace(angle_line, angle_line + dihedral_line, 1))
        return True
    quartic = '  om->register_new<QuarticBond>("Interactions::QuarticBond");\n'
    if text.count(quartic) != 1:
        raise RuntimeError(f"Could not locate ScriptInterface registration anchor in {path}")
    path.write_text(text.replace(
        quartic,
        quartic
        + '  om->register_new<ConservativeSplineDistanceBond>("Interactions::ConservativeSplineDistanceBond"); // MLCG conservative spline\n'
        + angle_line
        + dihedral_line,
        1,
    ))
    return True


def _ensure_python_default_params(path: Path) -> bool:
    """Implement BondedInteraction's abstract default-parameter hook.

    ESPResSo 5 declares ``BondedInteraction.get_default_params`` abstract.
    Conservative splines intentionally have no implicit defaults: min, max,
    energy and derivative are all required, so the concrete classes return an
    empty mapping.  The repair logic is idempotent and also upgrades trees
    patched by older installer revisions.
    """
    text = _read(path)
    changed = False
    for enum_name in (
        "CONSERVATIVE_SPLINE_DISTANCE",
        "CONSERVATIVE_SPLINE_ANGLE",
        "CONSERVATIVE_SPLINE_DIHEDRAL",
    ):
        anchor = f"    _type_number = BONDED_IA.{enum_name}\n"
        if text.count(anchor) != 1:
            raise RuntimeError(
                f"Could not uniquely locate Python conservative spline class "
                f"for BONDED_IA.{enum_name} in {path}"
            )
        start = text.index(anchor)
        next_class = text.find("\n@script_interface_register\nclass ", start + len(anchor))
        end = len(text) if next_class < 0 else next_class
        body = text[start:end]
        if "def get_default_params(self):" in body:
            continue
        replacement = (
            anchor
            + "\n"
            + "    def get_default_params(self):\n"
            + "        return {}\n"
        )
        text = text[:start] + text[start:].replace(anchor, replacement, 1)
        changed = True
    if changed:
        path.write_text(text)
    return changed


def install(root: Path, source_header: Path) -> list[str]:
    paths = required_paths(root)
    for key, path in paths.items():
        if key != "header":
            _read(path)
    source = _read(source_header)
    paths["header"].parent.mkdir(parents=True, exist_ok=True)
    changed = []
    if not paths["header"].is_file() or paths["header"].read_text() != source:
        shutil.copyfile(source_header, paths["header"])
        changed.append("conservative_spline_bond.hpp")

    if _ensure_include_once(
        paths["bond_data"],
        '#include "conservative_spline_bond.hpp" // MLCG conservative spline bonded interactions\n',
        '#include "harmonic.hpp"\n',
    ):
        changed.append("bonded_interaction_data.hpp include")
    if _patch_variant(paths["bond_data"]):
        changed.append("Bonded_IA_Parameters")

    pair_force = '''  // MLCG conservative spline bonded interactions\n  if (auto const *iap = std::get_if<ConservativeSplineDistanceBond>(&iaparams)) {\n    return iap->force(dx);\n  }\n'''
    if _ensure_dispatch_in_function(
        paths["forces"],
        function_name="calc_bond_pair_force",
        addition=pair_force,
        sentinel="std::get_if<ConservativeSplineDistanceBond>",
        anchors=("  if (std::get_if<VirtualBond>(&iaparams)) {\n",),
    ):
        changed.append("pair force dispatch")

    angle_force = '''  // MLCG conservative spline bonded interactions\n  if (auto const *iap = std::get_if<ConservativeSplineAngleBond>(&iaparams)) {\n    return iap->forces(vec1, vec2);\n  }\n'''
    if _ensure_dispatch_in_function(
        paths["forces"],
        function_name="calc_bonded_three_body_force",
        addition=angle_force,
        sentinel="std::get_if<ConservativeSplineAngleBond>",
        anchors=(
            "  if (auto const *iap = std::get_if<IBMTriel>(&iaparams)) {\n",
            "  throw BondUnknownTypeError();\n",
        ),
    ):
        changed.append("angle force dispatch")

    dihedral_force = '''  // MLCG conservative spline bonded interactions\n  if (auto const *iap = std::get_if<ConservativeSplineDihedralBond>(&iaparams)) {\n    return iap->forces(v12, v23, v34);\n  }\n'''
    if _ensure_dispatch_in_function(
        paths["forces"],
        function_name="calc_bonded_dihedral_force",
        addition=dihedral_force,
        sentinel="std::get_if<ConservativeSplineDihedralBond>",
        anchors=("  throw BondUnknownTypeError();\n",),
    ):
        changed.append("dihedral force dispatch")

    pair_energy = '''  // MLCG conservative spline bonded interactions\n  if (auto const *iap = std::get_if<ConservativeSplineDistanceBond>(&iaparams)) {\n    return iap->energy(dx);\n  }\n'''
    if _ensure_dispatch_in_function(
        paths["energy"],
        function_name="calc_pair_bonded_energy",
        addition=pair_energy,
        sentinel="std::get_if<ConservativeSplineDistanceBond>",
        anchors=("  if (std::get_if<VirtualBond>(&iaparams)) {\n",),
    ):
        changed.append("pair energy dispatch")

    angle_energy = '''  // MLCG conservative spline bonded interactions\n  if (auto const *iap = std::get_if<ConservativeSplineAngleBond>(&iaparams)) {\n    return iap->energy(vec1, vec2);\n  }\n'''
    if _ensure_dispatch_in_function(
        paths["energy"],
        function_name="calc_angle_bonded_energy",
        addition=angle_energy,
        sentinel="std::get_if<ConservativeSplineAngleBond>",
        anchors=(
            "  if (std::get_if<IBMTriel>(&iaparams)) {\n",
            "  if (auto const *iap = std::get_if<IBMTriel>(&iaparams)) {\n",
            "  throw BondUnknownTypeError();\n",
        ),
    ):
        changed.append("angle energy dispatch")

    dihedral_energy = '''  // MLCG conservative spline bonded interactions\n  if (auto const *iap = std::get_if<ConservativeSplineDihedralBond>(&iaparams)) {\n    return iap->energy(v12, v23, v34);\n  }\n'''
    if _ensure_dispatch_in_function(
        paths["energy"],
        function_name="calc_dihedral_bonded_energy",
        addition=dihedral_energy,
        sentinel="std::get_if<ConservativeSplineDihedralBond>",
        anchors=("  throw BondUnknownTypeError();\n",),
    ):
        changed.append("dihedral energy dispatch")

    classes = r'''// MLCG conservative spline bonded interactions
class ConservativeSplineDistanceBond
    : public BondedInteractionImpl<::ConservativeSplineDistanceBond> {
 public:
  ConservativeSplineDistanceBond() {
    add_parameters({
        {"min", AutoParameter::read_only, [this]() { return get_struct().spline.minval; }},
        {"max", AutoParameter::read_only, [this]() { return get_struct().spline.maxval; }},
        {"energy", AutoParameter::read_only, [this]() { return get_struct().spline.energy_nodes; }},
        {"derivative", AutoParameter::read_only, [this]() { return get_struct().spline.derivative_nodes; }},
    });
  }
 private:
  void construct_bond(VariantMap const &params) override {
    m_bonded_ia = std::make_shared<::Bonded_IA_Parameters>(CoreBondedInteraction(
        get_value<double>(params, "min"), get_value<double>(params, "max"),
        get_value<std::vector<double>>(params, "energy"),
        get_value<std::vector<double>>(params, "derivative")));
  }
};

class ConservativeSplineAngleBond
    : public BondedInteractionImpl<::ConservativeSplineAngleBond> {
 public:
  ConservativeSplineAngleBond() {
    add_parameters({
        {"min", AutoParameter::read_only, [this]() { return get_struct().spline.minval; }},
        {"max", AutoParameter::read_only, [this]() { return get_struct().spline.maxval; }},
        {"energy", AutoParameter::read_only, [this]() { return get_struct().spline.energy_nodes; }},
        {"derivative", AutoParameter::read_only, [this]() { return get_struct().spline.derivative_nodes; }},
    });
  }
 private:
  void construct_bond(VariantMap const &params) override {
    m_bonded_ia = std::make_shared<::Bonded_IA_Parameters>(CoreBondedInteraction(
        get_value<double>(params, "min"), get_value<double>(params, "max"),
        get_value<std::vector<double>>(params, "energy"),
        get_value<std::vector<double>>(params, "derivative")));
  }
};

class ConservativeSplineDihedralBond
    : public BondedInteractionImpl<::ConservativeSplineDihedralBond> {
 public:
  ConservativeSplineDihedralBond() {
    add_parameters({
        {"min", AutoParameter::read_only, [this]() { return get_struct().spline.minval; }},
        {"max", AutoParameter::read_only, [this]() { return get_struct().spline.maxval; }},
        {"energy", AutoParameter::read_only, [this]() { return get_struct().spline.energy_nodes; }},
        {"derivative", AutoParameter::read_only, [this]() { return get_struct().spline.derivative_nodes; }},
    });
  }
 private:
  void construct_bond(VariantMap const &params) override {
    m_bonded_ia = std::make_shared<::Bonded_IA_Parameters>(CoreBondedInteraction(
        get_value<double>(params, "min"), get_value<double>(params, "max"),
        get_value<std::vector<double>>(params, "energy"),
        get_value<std::vector<double>>(params, "derivative")));
  }
};

'''
    if _insert_before(
        paths["script_header"],
        "class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {",
        classes,
        "class ConservativeSplineDistanceBond",
    ):
        changed.append("ScriptInterface classes")

    dihedral_class = r'''// MLCG conservative spline bonded interactions
class ConservativeSplineDihedralBond
    : public BondedInteractionImpl<::ConservativeSplineDihedralBond> {
 public:
  ConservativeSplineDihedralBond() {
    add_parameters({
        {"min", AutoParameter::read_only, [this]() { return get_struct().spline.minval; }},
        {"max", AutoParameter::read_only, [this]() { return get_struct().spline.maxval; }},
        {"energy", AutoParameter::read_only, [this]() { return get_struct().spline.energy_nodes; }},
        {"derivative", AutoParameter::read_only, [this]() { return get_struct().spline.derivative_nodes; }},
    });
  }
 private:
  void construct_bond(VariantMap const &params) override {
    m_bonded_ia = std::make_shared<::Bonded_IA_Parameters>(CoreBondedInteraction(
        get_value<double>(params, "min"), get_value<double>(params, "max"),
        get_value<std::vector<double>>(params, "energy"),
        get_value<std::vector<double>>(params, "derivative")));
  }
};

'''
    if _insert_before(
        paths["script_header"],
        "class BondedCoulomb : public BondedInteractionImpl<::BondedCoulomb> {",
        dihedral_class,
        "class ConservativeSplineDihedralBond",
    ):
        changed.append("ScriptInterface dihedral class")

    if _ensure_registration(paths["script_init"]):
        changed.append("ScriptInterface registration")

    if _patch_python_enum(paths["python"]):
        changed.append("Python enum")

    py_classes = '''# MLCG conservative spline bonded interactions
@script_interface_register
class ConservativeSplineDistance(BondedInteraction):
    """Conservative cubic-Hermite distance potential from U and dU/dr nodes."""
    _so_name = "Interactions::ConservativeSplineDistanceBond"
    _type_number = BONDED_IA.CONSERVATIVE_SPLINE_DISTANCE

    def get_default_params(self):
        return {}


@script_interface_register
class ConservativeSplineAngle(BondedInteraction):
    """Conservative cubic-Hermite angle potential from U and dU/dtheta nodes."""
    _so_name = "Interactions::ConservativeSplineAngleBond"
    _type_number = BONDED_IA.CONSERVATIVE_SPLINE_ANGLE

    def get_default_params(self):
        return {}


@script_interface_register
class ConservativeSplineDihedral(BondedInteraction):
    """Periodic conservative Hermite dihedral potential from U and dU/dphi nodes."""
    _so_name = "Interactions::ConservativeSplineDihedralBond"
    _type_number = BONDED_IA.CONSERVATIVE_SPLINE_DIHEDRAL

    def get_default_params(self):
        return {}


'''
    if _insert_before(
        paths["python"],
        "@script_interface_register\nclass BondedInteractions(ScriptObjectMap):",
        py_classes,
        "class ConservativeSplineDistance(BondedInteraction)",
    ):
        changed.append("Python classes")
    py_dihedral = '''# MLCG conservative spline bonded interactions
@script_interface_register
class ConservativeSplineDihedral(BondedInteraction):
    """Periodic conservative Hermite dihedral potential from U and dU/dphi nodes."""
    _so_name = "Interactions::ConservativeSplineDihedralBond"
    _type_number = BONDED_IA.CONSERVATIVE_SPLINE_DIHEDRAL

    def get_default_params(self):
        return {}


'''
    if _insert_before(
        paths["python"],
        "@script_interface_register\nclass BondedInteractions(ScriptObjectMap):",
        py_dihedral,
        "class ConservativeSplineDihedral(BondedInteraction)",
    ):
        changed.append("Python dihedral class")
    if _ensure_python_default_params(paths["python"]):
        changed.append("Python get_default_params")
    return changed


def check(root: Path, source_header: Path) -> None:
    p = required_paths(root)
    checks = {
        "header": p["header"].is_file() and p["header"].read_text() == _read(source_header),
        "variant distance": "ConservativeSplineDistanceBond" in _read(p["bond_data"]),
        "variant angle": "ConservativeSplineAngleBond" in _read(p["bond_data"]),
        "variant dihedral": "ConservativeSplineDihedralBond" in _read(p["bond_data"]),
        "distance force": _dispatch_is_in_function(
            p["forces"], "calc_bond_pair_force", "std::get_if<ConservativeSplineDistanceBond>"
        ),
        "angle force": _dispatch_is_in_function(
            p["forces"], "calc_bonded_three_body_force", "std::get_if<ConservativeSplineAngleBond>"
        ),
        "dihedral force": _dispatch_is_in_function(
            p["forces"], "calc_bonded_dihedral_force", "std::get_if<ConservativeSplineDihedralBond>"
        ),
        "distance energy": _dispatch_is_in_function(
            p["energy"], "calc_pair_bonded_energy", "std::get_if<ConservativeSplineDistanceBond>"
        ),
        "angle energy": _dispatch_is_in_function(
            p["energy"], "calc_angle_bonded_energy", "std::get_if<ConservativeSplineAngleBond>"
        ),
        "dihedral energy": _dispatch_is_in_function(
            p["energy"], "calc_dihedral_bonded_energy", "std::get_if<ConservativeSplineDihedralBond>"
        ),
        "script distance": "BondedInteractionImpl<::ConservativeSplineDistanceBond>" in _read(p["script_header"]),
        "script angle": "BondedInteractionImpl<::ConservativeSplineAngleBond>" in _read(p["script_header"]),
        "script dihedral": "BondedInteractionImpl<::ConservativeSplineDihedralBond>" in _read(p["script_header"]),
        "registration": "Interactions::ConservativeSplineDistanceBond" in _read(p["script_init"]),
        "python distance": "class ConservativeSplineDistance(BondedInteraction)" in _read(p["python"]),
        "python angle": "class ConservativeSplineAngle(BondedInteraction)" in _read(p["python"]),
        "python dihedral": "class ConservativeSplineDihedral(BondedInteraction)" in _read(p["python"]),
        "python distance defaults": (
            "_type_number = BONDED_IA.CONSERVATIVE_SPLINE_DISTANCE\n\n"
            "    def get_default_params(self):\n"
            "        return {}"
        ) in _read(p["python"]),
        "python angle defaults": (
            "_type_number = BONDED_IA.CONSERVATIVE_SPLINE_ANGLE\n\n"
            "    def get_default_params(self):\n"
            "        return {}"
        ) in _read(p["python"]),
        "python dihedral defaults": (
            "_type_number = BONDED_IA.CONSERVATIVE_SPLINE_DIHEDRAL\n\n"
            "    def get_default_params(self):\n"
            "        return {}"
        ) in _read(p["python"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError("Conservative spline installation incomplete: " + ", ".join(failed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--espresso-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.espresso_root.expanduser().resolve()
    source_header = Path(__file__).resolve().with_name("conservative_spline_bond.hpp")
    if args.check:
        check(root, source_header)
        print("[PASS] Conservative spline bonded interactions are installed consistently.")
        return
    changed = install(root, source_header)
    check(root, source_header)
    if changed:
        print("[PASS] Installed conservative spline bonded interactions: " + ", ".join(changed))
    else:
        print("[PASS] Conservative spline bonded interactions already installed; no edits needed.")


if __name__ == "__main__":
    main()
