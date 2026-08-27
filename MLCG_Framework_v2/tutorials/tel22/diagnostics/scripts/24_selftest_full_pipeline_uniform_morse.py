#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "24_prepare_full_pipeline_uniform_morse.py"
spec = importlib.util.spec_from_file_location("prep24", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

TUTORIAL = Path(__file__).resolve().parents[2]


def main() -> int:
    topology = json.loads((TUTORIAL / "tel22_topology.json").read_text())
    priors = json.loads((TUTORIAL / "cg_priors.json").read_text())
    t, p = mod.validate_source(topology, priors)
    assert len(t) == len(p) == 180
    new_a = 0.255
    nt, ti = mod.rewrite_uniform(topology, new_a)
    np_, pi = mod.rewrite_uniform(priors, new_a)
    mod.assert_only_morse_a_changed(topology, nt, new_a)
    mod.assert_only_morse_a_changed(priors, np_, new_a)
    assert len(ti) == len(pi) == 180
    assert math.isclose((new_a / 0.3) ** 2, 0.7225, rel_tol=0.0, abs_tol=1e-15)

    # Exercise serialization and ensure source files are not touched.
    top_before = (TUTORIAL / "tel22_topology.json").read_bytes()
    pri_before = (TUTORIAL / "cg_priors.json").read_bytes()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        mod.dump_json(out / "topology.json", nt)
        mod.dump_json(out / "priors.json", np_)
        rt = json.loads((out / "topology.json").read_text())
        rp = json.loads((out / "priors.json").read_text())
        assert {float(x[1]["a"]) for x in mod.morse_records(rt)} == {new_a}
        assert {float(x[1]["a"]) for x in mod.morse_records(rp)} == {new_a}
    assert (TUTORIAL / "tel22_topology.json").read_bytes() == top_before
    assert (TUTORIAL / "cg_priors.json").read_bytes() == pri_before

    print("[PASS] production source has 180 uniform Morse contacts at a=0.3")
    print("[PASS] candidate rewrite changes only Morse a and yields a=0.255 on all 180")
    print("[PASS] production topology/priors remain untouched by candidate preparation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
