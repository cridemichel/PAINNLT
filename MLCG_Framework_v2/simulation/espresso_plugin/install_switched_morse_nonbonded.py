#!/usr/bin/env python3
"""Install the MLCG switched extension of ESPResSo's non-bonded Morse potential.

The extension is deliberately backward compatible: ``switch_start < 0`` keeps
ESPResSo's stock shifted-Morse semantics; a non-negative ``switch_start``
activates a quintic C1 switching tail used by the MLCG runtime.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


SENTINEL = "MLCG switched non-bonded Morse"


def _read(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required ESPResSo source file not found: {path}")
    return path.read_text()




def _has_feature_define(text: str, feature: str) -> bool:
    return re.search(rf"^\s*#\s*define\s+{re.escape(feature)}(?:\s|$)", text, flags=re.MULTILINE) is not None


def ensure_morse_build_feature(espresso_root: Path) -> tuple[Path, bool]:
    """Ensure the ESPResSo build configuration compiles the MORSE feature.

    ESPResSo gives ``build/myconfig.hpp`` precedence.  If no explicit build
    configuration exists, create one from the active source configuration (or
    the stock default) and add only ``#define MORSE``.  Existing feature
    selections are otherwise left untouched.
    """
    build_dir = espresso_root / "build"
    if not build_dir.is_dir():
        raise RuntimeError(
            f"ESPResSo build directory not found: {build_dir}. Configure ESPResSo first."
        )

    build_config = build_dir / "myconfig.hpp"
    source_config = espresso_root / "myconfig.hpp"
    default_config = espresso_root / "src/config/myconfig-default.hpp"

    if not build_config.exists():
        template = source_config if source_config.is_file() else default_config
        if not template.is_file():
            raise RuntimeError(
                "Could not locate an ESPResSo myconfig.hpp or src/config/myconfig-default.hpp"
            )
        shutil.copyfile(template, build_config)

    text = _read(build_config)
    if _has_feature_define(text, "MORSE"):
        return build_config, False

    if not text.endswith("\n"):
        text += "\n"
    text += "\n// Required by MLCG reversible non-bonded Morse contacts.\n#define MORSE\n"
    build_config.write_text(text)
    return build_config, True

def required_paths(espresso_root: Path) -> dict[str, Path]:
    return {
        "data": espresso_root / "src/core/nonbonded_interactions/nonbonded_interaction_data.hpp",
        "morse_hpp": espresso_root / "src/core/nonbonded_interactions/morse.hpp",
        "morse_cpp": espresso_root / "src/core/nonbonded_interactions/morse.cpp",
        "script": espresso_root / "src/script_interface/interactions/NonBondedInteraction.hpp",
        "python": espresso_root / "src/python/espressomd/interactions.py",
    }


def _replace_struct(data_path: Path) -> bool:
    text = _read(data_path)
    if "double switch_start = inactive_cutoff; // MLCG switched non-bonded Morse" in text:
        return False
    old = '''struct Morse_Parameters {
  double eps = 0.;
  double alpha = inactive_cutoff;
  double rmin = inactive_cutoff;
  double cut = inactive_cutoff;
  double rest = inactive_cutoff;
  Morse_Parameters() = default;
  Morse_Parameters(double eps, double alpha, double rmin, double cutoff);
  double max_cutoff() const { return cut; }
};'''
    new = '''struct Morse_Parameters {
  double eps = 0.;
  double alpha = inactive_cutoff;
  double rmin = inactive_cutoff;
  double cut = inactive_cutoff;
  double switch_start = inactive_cutoff; // MLCG switched non-bonded Morse
  double rest = inactive_cutoff;
  Morse_Parameters() = default;
  Morse_Parameters(double eps, double alpha, double rmin, double cutoff,
                   double switch_start = inactive_cutoff);
  double max_cutoff() const { return cut; }
};'''
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not recognize ESPResSo 5.0.x Morse_Parameters in {data_path}; "
            f"expected one stock block, found {count}."
        )
    data_path.write_text(text.replace(old, new, 1))
    return True


def _replace_script_interface(path: Path) -> bool:
    text = _read(path)
    if 'CoreInteraction::switch_start, "switch_start"' in text:
        return False

    match = re.search(
        r"(#ifdef ESPRESSO_MORSE\s+class InteractionMorse.*?#endif // ESPRESSO_MORSE)",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"Could not locate InteractionMorse block in {path}")
    block = match.group(1)

    parameter_anchor = '        make_autoparameter(&CoreInteraction::cut, "cutoff"),\n'
    if block.count(parameter_anchor) != 1:
        raise RuntimeError(
            f"Could not uniquely locate InteractionMorse cutoff parameter in {path}"
        )
    block = block.replace(
        parameter_anchor,
        parameter_anchor
        + '        make_autoparameter(&CoreInteraction::switch_start, "switch_start"), // MLCG switched non-bonded Morse\n',
        1,
    )

    ctor_pattern = re.compile(
        r"make_shared_from_args<CoreInteraction,\s*double,\s*double,\s*double,\s*double>\(\s*"
        r"params,\s*\"eps\",\s*\"alpha\",\s*\"rmin\",\s*\"cutoff\"\);",
        flags=re.DOTALL,
    )
    block, count = ctor_pattern.subn(
        'make_shared_from_args<CoreInteraction, double, double, double, double, double>(\n'
        '            params, "eps", "alpha", "rmin", "cutoff", "switch_start");',
        block,
        count=1,
    )
    if count != 1:
        raise RuntimeError(
            f"Could not recognize ESPResSo 5.0.x InteractionMorse constructor in {path}"
        )

    path.write_text(text[: match.start(1)] + block + text[match.end(1) :])
    return True


def _replace_python_default(path: Path) -> bool:
    text = _read(path)
    # Python-side defaults must supply switch_start because the ScriptInterface
    # requires every registered parameter on set_params().
    class_match = re.search(
        r"(^class MorseInteraction\(NonBondedInteraction\):.*?)(?=^class |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if class_match is None:
        raise RuntimeError(f"Could not locate MorseInteraction class in {path}")
    block = class_match.group(1)
    if '"switch_start": -1.' in block or "'switch_start': -1." in block:
        return False

    patterns = [
        (r'return \{"cutoff": 0\.\}', 'return {"cutoff": 0., "switch_start": -1.}'),
        (r'return \{"cutoff": 0\.0\}', 'return {"cutoff": 0.0, "switch_start": -1.0}'),
        (r"return \{'cutoff': 0\.\}", "return {'cutoff': 0., 'switch_start': -1.}"),
        (r"return \{'cutoff': 0\.0\}", "return {'cutoff': 0.0, 'switch_start': -1.0}"),
    ]
    new_block = block
    for pattern, replacement in patterns:
        new_block, count = re.subn(pattern, replacement, new_block, count=1)
        if count:
            break
    else:
        raise RuntimeError(
            f"Could not recognize MorseInteraction.default_params() in {path}"
        )

    path.write_text(text[: class_match.start(1)] + new_block + text[class_match.end(1) :])
    return True


def _copy_if_different(src: Path, dst: Path) -> bool:
    source = _read(src)
    if dst.is_file() and dst.read_text() == source:
        return False
    shutil.copyfile(src, dst)
    return True


def install(espresso_root: Path, source_dir: Path) -> list[str]:
    paths = required_paths(espresso_root)
    for path in paths.values():
        _read(path)

    changed: list[str] = []
    config_path, feature_changed = ensure_morse_build_feature(espresso_root)
    if feature_changed:
        changed.append(f"MORSE build feature ({config_path})")
    if _replace_struct(paths["data"]):
        changed.append("Morse_Parameters")
    if _replace_script_interface(paths["script"]):
        changed.append("InteractionMorse ScriptInterface")
    if _replace_python_default(paths["python"]):
        changed.append("MorseInteraction Python default")
    if _copy_if_different(source_dir / "morse_switched.hpp", paths["morse_hpp"]):
        changed.append("morse.hpp kernel")
    if _copy_if_different(source_dir / "morse_switched.cpp", paths["morse_cpp"]):
        changed.append("morse.cpp parameters")
    return changed


def check(espresso_root: Path, source_dir: Path) -> None:
    paths = required_paths(espresso_root)
    data = _read(paths["data"])
    script = _read(paths["script"])
    py = _read(paths["python"])
    build_config = espresso_root / "build/myconfig.hpp"
    checks = {
        "MORSE build feature": build_config.is_file() and _has_feature_define(_read(build_config), "MORSE"),
        "switch_start core field": "double switch_start = inactive_cutoff; // MLCG switched non-bonded Morse" in data,
        "five-argument core constructor": "double switch_start = inactive_cutoff);" in data,
        "ScriptInterface parameter": 'CoreInteraction::switch_start, "switch_start"' in script,
        "ScriptInterface constructor": '"cutoff", "switch_start"' in script,
        "Python default": "switch_start" in py,
        "Morse kernel": paths["morse_hpp"].read_text() == _read(source_dir / "morse_switched.hpp"),
        "Morse constructor": paths["morse_cpp"].read_text() == _read(source_dir / "morse_switched.cpp"),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "Switched non-bonded Morse installation incomplete: " + ", ".join(failed)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--espresso-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.espresso_root.expanduser().resolve()
    source_dir = Path(__file__).resolve().parent
    if args.check:
        check(root, source_dir)
        print("[PASS] Switched non-bonded Morse is installed consistently.")
        return

    changed = install(root, source_dir)
    check(root, source_dir)
    if changed:
        print("[PASS] Installed switched non-bonded Morse: " + ", ".join(changed))
    else:
        print("[PASS] Switched non-bonded Morse already installed; no edits needed.")


if __name__ == "__main__":
    main()
