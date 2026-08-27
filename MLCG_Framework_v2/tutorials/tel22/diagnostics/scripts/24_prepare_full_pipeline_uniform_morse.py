#!/usr/bin/env python3
"""Prepare an isolated TEL22 full-pipeline candidate with uniform Morse a.

The candidate is deliberately derived from the current production topology and
priors.  Only the explicit pair-specific Morse ``a`` values are changed; all
other topology/prior fields remain byte-for-byte-equivalent at the JSON data
model level.  Production artifacts are never modified in place.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

KIND = "tel22_full_pipeline_uniform_morse_inputs"
SCHEMA_VERSION = 1
EXPECTED_MORSE = 180
EXPECTED_OLD_A = 0.3
DEFAULT_NEW_A = 0.255
EXPECTED_D = 50.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def morse_records(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for idx, entry in enumerate(data.get("bonds", [])):
        if isinstance(entry, dict) and str(entry.get("type", "")).lower() == "morse":
            out.append((idx, entry))
    return out


def endpoint_key(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(entry["mol_i"]), int(entry["mol_j"]),
        int(entry.get("site_i", -1)), int(entry.get("site_j", -1)),
    )


def validate_source(topology: dict[str, Any], priors: dict[str, Any]) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    t = morse_records(topology)
    p = morse_records(priors)
    if len(t) != EXPECTED_MORSE or len(p) != EXPECTED_MORSE:
        raise ValueError(f"expected {EXPECTED_MORSE} explicit Morse records; topology={len(t)} priors={len(p)}")
    if topology.get("morse_type_pairs", []) or priors.get("morse_type_pairs", []):
        raise ValueError("TEL22 uniform-stabilizer pipeline requires morse_type_pairs to be empty")

    def validate_arm(label: str, records: list[tuple[int, dict[str, Any]]]) -> None:
        for idx, rec in records:
            if abs(float(rec["a"]) - EXPECTED_OLD_A) > 1e-15:
                raise ValueError(f"{label} Morse bonds[{idx}] has a={rec['a']}, expected production {EXPECTED_OLD_A}")
            if abs(float(rec["D"]) - EXPECTED_D) > 1e-12:
                raise ValueError(f"{label} Morse bonds[{idx}] has D={rec['D']}, expected {EXPECTED_D}")
    validate_arm("topology", t)
    validate_arm("priors", p)

    t_by_key = {endpoint_key(rec): rec for _, rec in t}
    p_by_key = {endpoint_key(rec): rec for _, rec in p}
    if set(t_by_key) != set(p_by_key):
        raise ValueError("production topology/prior Morse endpoint sets differ")
    for key in sorted(t_by_key):
        tr, pr = t_by_key[key], p_by_key[key]
        for field in ("D", "a", "r0"):
            if abs(float(tr[field]) - float(pr[field])) > 1e-12:
                raise ValueError(f"topology/prior Morse mismatch at {key} field {field}: {tr[field]} vs {pr[field]}")
    return t, p


def rewrite_uniform(data: dict[str, Any], new_a: float) -> tuple[dict[str, Any], list[int]]:
    out = copy.deepcopy(data)
    changed: list[int] = []
    for idx, rec in morse_records(out):
        rec["a"] = float(new_a)
        changed.append(idx)
    return out, changed


def assert_only_morse_a_changed(old: dict[str, Any], new: dict[str, Any], new_a: float) -> None:
    old_copy = copy.deepcopy(old)
    new_copy = copy.deepcopy(new)
    old_m = morse_records(old_copy)
    new_m = morse_records(new_copy)
    if [idx for idx, _ in old_m] != [idx for idx, _ in new_m]:
        raise ValueError("Morse record indices changed")
    for (oi, o), (ni, n) in zip(old_m, new_m):
        assert oi == ni
        if abs(float(n["a"]) - new_a) > 1e-15:
            raise ValueError(f"candidate Morse bonds[{ni}] a is not {new_a}")
        o["a"] = float(new_a)
    if old_copy != new_copy:
        raise ValueError("candidate differs from source in fields other than explicit Morse a")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True, type=Path)
    ap.add_argument("--priors", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--new-a", type=float, default=DEFAULT_NEW_A)
    ap.add_argument("--abc-summary", type=Path, default=None)
    ap.add_argument("--closure-b-report", type=Path, default=None)
    ap.add_argument("--closure-c-report", type=Path, default=None)
    args = ap.parse_args()

    if args.new_a <= 0.0:
        raise ValueError("--new-a must be positive")
    topology_path = args.topology.resolve()
    priors_path = args.priors.resolve()
    topology = load_json(topology_path)
    priors = load_json(priors_path)
    validate_source(topology, priors)

    # Require the already-completed ABC decision if supplied by the runner.
    provenance: dict[str, Any] = {}
    if args.abc_summary is not None:
        abc_path = args.abc_summary.resolve()
        abc = load_json(abc_path)
        if abc.get("kind") != "tel22_morse_stabilizer_abc_10ps_fullgrid":
            raise ValueError("unexpected ABC summary kind")
        if abc.get("recommended_numerical_stabilizer_for_next_step") != "C_uniform_a0p85":
            raise ValueError("ABC summary did not recommend C_uniform_a0p85")
        if abs(float(abc["arms"]["C_uniform_a0p85"]["a_uniform"]) - args.new_a) > 1e-15:
            raise ValueError("ABC recommended a differs from requested pipeline a")
        provenance["morse_uniform_abc_summary"] = {"path": str(abc_path), "sha256": sha256(abc_path)}

    # Require the old-PaiNN closure to show that retraining is warranted.
    if (args.closure_b_report is None) != (args.closure_c_report is None):
        raise ValueError("closure B and C reports must be supplied together")
    if args.closure_b_report is not None:
        b_path = args.closure_b_report.resolve(); c_path = args.closure_c_report.resolve()
        b = load_json(b_path); c = load_json(c_path)
        if b.get("definition", {}).get("hamiltonian_mode") != "model_active":
            raise ValueError("closure B is not model_active")
        if c.get("definition", {}).get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
            raise ValueError("closure C is not the ML-disabled control")
        if b.get("inputs_sha256", {}).get("priors") != c.get("inputs_sha256", {}).get("priors"):
            raise ValueError("closure B/C do not use identical uniform priors")
        pb = float(b["certification"]["scaling"]["exponent_p"])
        pc = float(c["certification"]["scaling"]["exponent_p"])
        dtmin_b = min(b["runs"], key=lambda x: float(x["dt_ps"]))
        dtmin_c = min(c["runs"], key=lambda x: float(x["dt_ps"]))
        floor_ratio = float(dtmin_b["sigma_E"]) / float(dtmin_c["sigma_E"])
        if not (abs(pc - 2.0) < abs(pb - 2.0) and floor_ratio > 1.5):
            raise ValueError(
                f"closure does not support retraining: p_oldML={pb}, p_noML={pc}, dtmin_floor_ratio={floor_ratio}"
            )
        provenance["old_painn_closure_B"] = {"path": str(b_path), "sha256": sha256(b_path), "p": pb}
        provenance["old_painn_closure_C"] = {"path": str(c_path), "sha256": sha256(c_path), "p": pc, "dtmin_sigma_ratio_B_over_C": floor_ratio}

    candidate_topology, top_changed = rewrite_uniform(topology, args.new_a)
    candidate_priors, pri_changed = rewrite_uniform(priors, args.new_a)
    assert_only_morse_a_changed(topology, candidate_topology, args.new_a)
    assert_only_morse_a_changed(priors, candidate_priors, args.new_a)
    if len(top_changed) != EXPECTED_MORSE or len(pri_changed) != EXPECTED_MORSE:
        raise AssertionError("unexpected changed Morse count")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    top_out = out / "tel22_topology_uniform_a0p255.json"
    pri_out = out / "cg_priors.json"
    dump_json(top_out, candidate_topology)
    dump_json(pri_out, candidate_priors)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "stabilizer_context": "TEL22 Morse terms are empirical structural/numerical stabilizers; all 180 explicit contacts use one uniform a.",
        "source_topology": str(topology_path),
        "source_topology_sha256": sha256(topology_path),
        "source_priors": str(priors_path),
        "source_priors_sha256": sha256(priors_path),
        "candidate_topology": str(top_out),
        "candidate_topology_sha256": sha256(top_out),
        "candidate_priors": str(pri_out),
        "candidate_priors_sha256": sha256(pri_out),
        "morse_count": EXPECTED_MORSE,
        "source_uniform_a": EXPECTED_OLD_A,
        "candidate_uniform_a": float(args.new_a),
        "k_at_r0_ratio": float((args.new_a / EXPECTED_OLD_A) ** 2),
        "topology_changed_indices": top_changed,
        "priors_changed_indices": pri_changed,
        "only_explicit_morse_a_changed": True,
        "provenance": provenance,
    }
    manifest_path = out / "full_pipeline_input_manifest.json"
    dump_json(manifest_path, manifest)

    print(f"[PREPARED] {EXPECTED_MORSE} Morse contacts: a={EXPECTED_OLD_A} -> {args.new_a}")
    print(f"[TOPOLOGY] {top_out}")
    print(f"[PRIORS]   {pri_out}")
    print(f"[MANIFEST] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
