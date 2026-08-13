"""Shared validation and rigid-body helpers for MLCG_Framework_v2.

This module intentionally depends only on NumPy and the Python standard library,
so it can be imported by ESPResSo's ``pypresso`` interpreter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 3
MODEL_MANIFEST_SCHEMA_VERSION = 3
ENERGY_GAUGE = "isolated_species_zero_v1"
PAINN_ARCHITECTURE_VARIANT = "painn_canonical_context_silu_v2"


def validate_wca_exclusion_policy(priors: dict[str, Any]) -> None:
    """Require the selective 1-2 / all-sites 1-3 WCA policy (schema v3)."""
    meta = priors.get("wca_exclusions", {})
    if not (
        meta.get("policy_version") == 3
        and meta.get("exclude_12") is True
        and meta.get("exclude_13") is True
        and meta.get("direct_scope") == "bonded_site_pairs_only"
        and meta.get("one_three_scope") == "molecule_pair_all_sites"
        and meta.get("pair_source") == "explicit_topology_pairs_v3"
        and isinstance(meta.get("direct_pairs"), list)
        and isinstance(meta.get("direct_site_pairs"), list)
        and isinstance(meta.get("one_three_pairs"), list)
    ):
        raise ValueError(
            "cg_priors.json does not declare the WCA policy v3 "
            "(1-2 bonded-site-only, 1-3 all-sites). Rebuild the dataset and "
            "cg_priors.json, retrain the model, and re-equilibrate before runtime use."
        )


def wca_topology_exclusion_pairs(
    priors: dict[str, Any], num_molecules: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return explicit topological 1-2 and 1-3 molecule pairs from cg_priors."""
    validate_wca_exclusion_policy(priors)
    meta = priors["wca_exclusions"]

    def parse_pairs(name: str) -> set[tuple[int, int]]:
        result: set[tuple[int, int]] = set()
        raw_pairs = meta.get(name, [])
        for raw in raw_pairs:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"Invalid WCA exclusion pair in {name}: {raw!r}")
            mi, mj = int(raw[0]), int(raw[1])
            if not (0 <= mi < num_molecules and 0 <= mj < num_molecules and mi != mj):
                raise ValueError(f"Out-of-range WCA exclusion pair in {name}: {(mi, mj)}")
            result.add((min(mi, mj), max(mi, mj)))
        if len(result) != len(raw_pairs):
            raise ValueError(f"Duplicate WCA exclusion pair in {name}")
        return result

    direct_pairs = parse_pairs("direct_pairs")
    one_three_pairs = parse_pairs("one_three_pairs")
    if direct_pairs & one_three_pairs:
        raise ValueError("WCA direct_pairs and one_three_pairs must be disjoint")
    if len(direct_pairs) != int(meta.get("direct_pair_count", -1)):
        raise ValueError("WCA direct_pair_count does not match explicit direct_pairs")
    if len(one_three_pairs) != int(meta.get("one_three_pair_count", -1)):
        raise ValueError("WCA one_three_pair_count does not match explicit one_three_pairs")
    return direct_pairs, one_three_pairs


def wca_direct_bonded_site_exclusions(
    priors: dict[str, Any], num_molecules: int
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Return production 1-2 virtual-site exclusions stored in cg_priors.

    Under policy v3, topological 1-2 molecule pairs retain WCA on every
    cross-body virtual-site pair except the site pair(s) that are explicitly
    bonded with ``exclude_wca=true``.  The metadata is authoritative at runtime
    and is cross-checked against the bonded priors when those records are
    available.
    """
    direct_pairs, _ = wca_topology_exclusion_pairs(priors, num_molecules)
    meta = priors["wca_exclusions"]
    result: dict[tuple[int, int], set[tuple[int, int]]] = {pair: set() for pair in direct_pairs}
    raw_records = meta.get("direct_site_pairs", [])
    seen_records: set[tuple[int, int, int, int]] = set()

    for raw in raw_records:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f"Invalid WCA direct_site_pairs record: {raw!r}")
        mi, mj, si, sj = map(int, raw)
        if not (0 <= mi < num_molecules and 0 <= mj < num_molecules and mi != mj):
            raise ValueError(f"Out-of-range WCA direct site-pair molecule indices: {raw!r}")
        if si < 0 or sj < 0:
            raise ValueError(f"WCA direct_site_pairs requires non-negative site indices: {raw!r}")
        if mi < mj:
            record = (mi, mj, si, sj)
        else:
            record = (mj, mi, sj, si)
        pair = record[:2]
        if pair not in direct_pairs:
            raise ValueError(
                f"WCA direct site-pair {record!r} does not belong to an explicit direct_pair"
            )
        if record in seen_records:
            raise ValueError(f"Duplicate WCA direct site-pair record: {record!r}")
        seen_records.add(record)
        result[pair].add(record[2:])

    if len(seen_records) != int(meta.get("direct_site_pair_count", -1)):
        raise ValueError(
            "WCA direct_site_pair_count does not match explicit direct_site_pairs"
        )

    # Cross-check the stored site-level policy against the bonded priors.  COM-level
    # bonds (negative/missing site indices) create a 1-2 topology relation but do
    # not suppress WCA between any virtual-site pair.
    expected: set[tuple[int, int, int, int]] = set()
    for bond in priors.get("bonds", []):
        if isinstance(bond, dict):
            excludes = bool(
                bond.get("exclude_wca", str(bond.get("type", "harmonic")).lower() != "morse")
            )
            if not excludes or "mol_i" not in bond or "mol_j" not in bond:
                continue
            mi, mj = int(bond["mol_i"]), int(bond["mol_j"])
            si, sj = int(bond.get("site_i", -1)), int(bond.get("site_j", -1))
        elif isinstance(bond, (list, tuple)) and len(bond) >= 2:
            mi, mj = int(bond[0]), int(bond[1])
            si = int(bond[2]) if len(bond) > 2 else -1
            sj = int(bond[3]) if len(bond) > 3 else -1
        else:
            continue
        pair = (min(mi, mj), max(mi, mj))
        if pair not in direct_pairs or si < 0 or sj < 0:
            continue
        expected.add((mi, mj, si, sj) if mi < mj else (mj, mi, sj, si))

    if expected != seen_records:
        missing = sorted(expected - seen_records)[:8]
        extra = sorted(seen_records - expected)[:8]
        raise ValueError(
            "WCA direct_site_pairs metadata disagrees with bonded priors; "
            f"missing={missing}, extra={extra}. Rebuild cg_priors.json."
        )
    return result

def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_manifest_path(model_path: str | Path) -> Path:
    return Path(f"{Path(model_path)}.manifest.json")


def _effective_architecture(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "architecture_variant", "num_species", "hidden_channels",
        "n_layers", "num_rbf", "cutoff", "toxvaerd_alpha"
    )
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Model config is missing required architecture keys: {missing}")
    variant = str(config["architecture_variant"])
    if variant != PAINN_ARCHITECTURE_VARIANT:
        raise ValueError(
            f"Unsupported PaiNN architecture_variant {variant!r}; "
            f"expected {PAINN_ARCHITECTURE_VARIANT!r}."
        )
    return {
        "variant": variant,
        "num_species": int(config["num_species"]),
        "hidden_channels": int(config["hidden_channels"]),
        "n_layers": int(config["n_layers"]),
        "num_rbf": int(config["num_rbf"]),
        "cutoff": float(config["cutoff"]),
        "toxvaerd_alpha": float(config["toxvaerd_alpha"]),
    }


def validate_model_manifest(
    model_path: str | Path,
    config: dict[str, Any],
    *,
    allow_missing: bool = False,
) -> dict[str, Any] | None:
    """Validate that the runtime architecture matches the training manifest."""
    model_path = Path(model_path)
    manifest_path = model_manifest_path(model_path)
    if not manifest_path.is_file():
        if allow_missing:
            print(f"[WARNING] Missing model manifest: {manifest_path}")
            return None
        raise FileNotFoundError(
            f"Missing model manifest {manifest_path}. Retrain the model with the patched trainer "
            "or pass --allow_missing_model_manifest explicitly."
        )

    with manifest_path.open() as handle:
        manifest = json.load(handle)

    if int(manifest.get("schema_version", -1)) != MODEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model manifest schema {manifest.get('schema_version')} in {manifest_path}"
        )
    if manifest.get("framework") != "MLCG_Framework_v2":
        raise ValueError(f"Unexpected framework identifier in {manifest_path}: {manifest.get('framework')}")
    if manifest.get("energy_gauge") != ENERGY_GAUGE:
        raise ValueError(
            f"Unsupported or missing energy gauge in {manifest_path}: "
            f"{manifest.get('energy_gauge')!r}; expected {ENERGY_GAUGE!r}. "
            "Regenerate the manifest with training/create_model_manifest.py."
        )

    expected = _effective_architecture(config)
    recorded = manifest.get("architecture", {})
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in recorded:
            mismatches.append(f"{key}: missing from manifest")
            continue
        actual = recorded[key]
        if isinstance(expected_value, float):
            # Architecture floats may be serialized after a float32 round-trip
            # (e.g. 1.2616 -> 1.2616000175476074).  Treat that representation
            # noise as equal while still rejecting physically meaningful changes.
            if not math.isclose(float(actual), expected_value, rel_tol=1e-6, abs_tol=1e-8):
                mismatches.append(f"{key}: manifest={actual}, runtime={expected_value}")
        elif isinstance(expected_value, str):
            if str(actual) != expected_value:
                mismatches.append(f"{key}: manifest={actual}, runtime={expected_value}")
        elif int(actual) != expected_value:
            mismatches.append(f"{key}: manifest={actual}, runtime={expected_value}")

    recorded_size = manifest.get("model_file_size_bytes")
    if recorded_size is not None and int(recorded_size) != model_path.stat().st_size:
        mismatches.append(
            f"model_file_size_bytes: manifest={recorded_size}, runtime={model_path.stat().st_size}"
        )
    recorded_hash = manifest.get("model_sha256")
    if recorded_hash is not None:
        current_hash = sha256_file(model_path)
        if recorded_hash != current_hash:
            mismatches.append(f"model_sha256: manifest={recorded_hash}, runtime={current_hash}")

    if mismatches:
        raise ValueError("Model manifest mismatch:\n  - " + "\n  - ".join(mismatches))

    print(f"[INFO] Model manifest validated: {manifest_path}")
    return manifest


def input_hashes(
    *,
    dataset: str | Path,
    config: str | Path,
    priors: str | Path,
    rb_info: str | Path,
    model: str | Path | None,
) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {
        "dataset_sha256": sha256_file(dataset),
        "config_sha256": sha256_file(config),
        "priors_sha256": sha256_file(priors),
        "rb_info_sha256": sha256_file(rb_info),
        "model_sha256": sha256_file(model) if model is not None else None,
    }
    if model is not None:
        manifest = model_manifest_path(model)
        hashes["model_manifest_sha256"] = sha256_file(manifest) if manifest.is_file() else None
    else:
        hashes["model_manifest_sha256"] = None
    return hashes


def particle_signature(system: Any) -> dict[str, np.ndarray]:
    particles = [system.part.by_id(i) for i in range(len(system.part))]
    return {
        "particle_ids": np.asarray([int(p.id) for p in particles], dtype=np.int64),
        "particle_types": np.asarray([int(p.type) for p in particles], dtype=np.int64),
        "particle_mol_ids": np.asarray([int(p.mol_id) for p in particles], dtype=np.int64),
        "particle_is_virtual": np.asarray([bool(p.is_virtual) for p in particles], dtype=np.bool_),
    }


def save_checkpoint(
    path: str | Path,
    *,
    system: Any,
    pos: np.ndarray,
    vel: np.ndarray,
    quat: np.ndarray,
    omega: np.ndarray,
    hashes: dict[str, str | None],
    config: dict[str, Any],
    dt: float,
    kT: float,
) -> None:
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "framework": "MLCG_Framework_v2",
        "energy_gauge": ENERGY_GAUGE,
        "input_hashes": hashes,
        "architecture": _effective_architecture(config),
        "created_with_dt_ps": float(dt),
        "created_with_kT_kJ_mol": float(kT),
    }
    signature = particle_signature(system)
    np.savez_compressed(
        path,
        pos=np.asarray(pos, dtype=float),
        v=np.asarray(vel, dtype=float),
        quat=np.asarray(quat, dtype=float),
        omega=np.asarray(omega, dtype=float),
        box_l=np.asarray(system.box_l, dtype=float),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **signature,
    )


def _read_scalar_string(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("Expected a scalar string in checkpoint metadata")
    return str(array.item())


def validate_checkpoint(
    checkpoint: Any,
    *,
    system: Any,
    expected_hashes: dict[str, str | None],
    expected_config: dict[str, Any],
    allow_legacy: bool = False,
    allow_mismatch: bool = False,
) -> dict[str, Any] | None:
    """Validate checkpoint provenance and particle identity before loading state."""
    required = {
        "metadata_json",
        "box_l",
        "particle_ids",
        "particle_types",
        "particle_mol_ids",
        "particle_is_virtual",
    }
    missing = sorted(required.difference(checkpoint.files))
    if missing:
        if allow_legacy:
            print(f"[WARNING] Legacy checkpoint without provenance fields: {missing}")
            return None
        raise ValueError(
            "Checkpoint lacks provenance metadata. Regenerate it with patched equilibrate.py "
            "or pass --allow_legacy_checkpoint explicitly. Missing: " + ", ".join(missing)
        )

    metadata = json.loads(_read_scalar_string(checkpoint["metadata_json"]))
    if int(metadata.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {metadata.get('schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )

    mismatches: list[str] = []
    if metadata.get("energy_gauge") != ENERGY_GAUGE:
        mismatches.append(
            f"energy_gauge: checkpoint={metadata.get('energy_gauge')!r}, runtime={ENERGY_GAUGE!r}"
        )
    recorded_architecture = metadata.get("architecture", {})
    expected_architecture = _effective_architecture(expected_config)
    for key, expected in expected_architecture.items():
        actual = recorded_architecture.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
                mismatches.append(f"architecture.{key}: checkpoint={actual}, runtime={expected}")
        elif isinstance(expected, str):
            if actual is None or str(actual) != expected:
                mismatches.append(f"architecture.{key}: checkpoint={actual}, runtime={expected}")
        elif actual is None or int(actual) != expected:
            mismatches.append(f"architecture.{key}: checkpoint={actual}, runtime={expected}")

    recorded_hashes = metadata.get("input_hashes", {})
    for key, expected in expected_hashes.items():
        actual = recorded_hashes.get(key)
        if actual != expected:
            mismatches.append(f"{key}: checkpoint={actual}, runtime={expected}")

    if not np.allclose(np.asarray(checkpoint["box_l"], dtype=float), np.asarray(system.box_l, dtype=float), rtol=0.0, atol=1e-6):
        mismatches.append(
            f"box_l: checkpoint={np.asarray(checkpoint['box_l']).tolist()}, runtime={list(system.box_l)}"
        )

    current = particle_signature(system)
    for key, expected in current.items():
        actual = np.asarray(checkpoint[key])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            mismatches.append(f"{key}: checkpoint particle identity differs from runtime")

    if mismatches:
        message = "Checkpoint provenance mismatch:\n  - " + "\n  - ".join(mismatches)
        if allow_mismatch:
            print("[WARNING] " + message.replace("\n", "\n[WARNING] "))
        else:
            raise ValueError(message)
    else:
        print("[INFO] Checkpoint provenance and particle identity validated.")
    return metadata


def ensure_single_rank(system: Any, *, allow_unsafe_mpi: bool = False) -> None:
    try:
        node_grid = tuple(int(v) for v in system.cell_system.node_grid)
        ranks = int(np.prod(node_grid))
    except Exception:
        return
    if ranks > 1 and not allow_unsafe_mpi:
        raise RuntimeError(
            f"PaiNN multi-rank execution is not certified (node_grid={node_grid}). "
            "Use one MPI rank, or pass --allow_unsafe_mpi only for an explicit parity experiment."
        )
    if ranks > 1:
        print(f"[WARNING] Running uncertified PaiNN multi-rank path with node_grid={node_grid}")


def nonconservative_prior_entries(priors: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    for index, bond in enumerate(priors.get("bonds", [])):
        if bond.get("type", "harmonic") == "tabulated":
            entries.append(f"bond[{index}]={bond.get('type')}")
    for index, angle in enumerate(priors.get("angles", [])):
        if angle.get("type", "harmonic") == "tabulated":
            entries.append(f"angle[{index}]=tabulated")
    for index, dihedral in enumerate(priors.get("dihedrals", [])):
        if dihedral.get("type", "cosine") == "tabulated":
            entries.append(f"dihedral[{index}]=tabulated")
    return entries


def _validate_rb_template(resname: str, data: dict[str, Any]) -> None:
    if int(data.get("schema_version", -1)) != 2 or data.get("body_frame") != "principal_axes":
        raise ValueError(
            f"Rigid-body template {resname} is legacy or lacks principal-axis metadata. "
            "Regenerate rigid_bodies_info.json with the patched preprocessing pipeline."
        )
    inertia = np.asarray(data.get("inertia_amu_nm2", []), dtype=float)
    if inertia.shape != (3,) or not np.isfinite(inertia).all() or np.any(inertia < 0.0):
        raise ValueError(f"Invalid principal moments for rigid-body template {resname}: {inertia}")
    for site_name, site in data.get("sites", {}).items():
        offset = np.asarray(site.get("relative_pos_nm", []), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(f"Invalid offset for {resname}/{site_name}: {offset}")
        int(site["type"])


def get_rb_data_by_sites(site_types: Iterable[int], rb_info: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    site_types = [int(value) for value in site_types]
    exact: list[tuple[str, dict[str, Any]]] = []
    for resname, data in rb_info.items():
        if not isinstance(data, dict) or "sites" not in data:
            continue
        expected = [int(site["type"]) for site in data["sites"].values()]
        if expected == site_types:
            exact.append((resname, data))
    if len(exact) == 1:
        _validate_rb_template(*exact[0])
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"Ambiguous rigid-body templates for ordered site types {site_types}: "
            + ", ".join(name for name, _ in exact)
        )
    raise ValueError(
        f"No rigid-body template matches ordered site types {site_types}. "
        "Site order is part of the dataset schema; regenerate inconsistent artifacts."
    )

def _kabsch_rotation(body_points: np.ndarray, space_points: np.ndarray) -> np.ndarray:
    if body_points.shape != space_points.shape or body_points.ndim != 2 or body_points.shape[1] != 3:
        raise ValueError("Kabsch inputs must both have shape (N, 3)")
    covariance = body_points.T @ space_points
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or np.linalg.det(rotation) < 0.0:
        raise ValueError("Failed to construct a proper rigid-body rotation")
    return rotation


def _matrix_to_quat_fallback(matrix: np.ndarray) -> np.ndarray:
    """ESPResSo-compatible scalar-first quaternion conversion.

    This mirrors ``espressomd.rotation.matrix_to_quat`` for installations where
    that helper is unavailable.
    """
    matrix = np.asarray(matrix, dtype=float)
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-7):
        raise ValueError("Only proper rotations are supported")
    m = matrix.copy().T
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = np.array([m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]])
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = np.array([m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]])
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = np.array([m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t])
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = np.array([t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]])
    q *= 0.5 / math.sqrt(t)
    return q / np.linalg.norm(q)


def matrix_to_espresso_quat(rotation: np.ndarray) -> np.ndarray:
    try:
        from espressomd.rotation import matrix_to_quat  # type: ignore

        quat = np.asarray(matrix_to_quat(rotation), dtype=float)
    except (ImportError, AttributeError):
        quat = _matrix_to_quat_fallback(rotation)
    return quat / np.linalg.norm(quat)


def rigid_body_quaternion(
    center: Iterable[float],
    site_positions: Iterable[Iterable[float]],
    box_l: Iterable[float],
    rb_data: dict[str, Any],
) -> np.ndarray:
    """Infer the COM orientation from principal-frame template sites."""
    if rb_data.get("body_frame") != "principal_axes":
        return np.asarray([1.0, 0.0, 0.0, 0.0])

    body_offsets = np.asarray(
        [site["relative_pos_nm"] for site in rb_data["sites"].values()], dtype=float
    )
    space_positions = np.asarray(list(site_positions), dtype=float)
    if body_offsets.shape != space_positions.shape:
        raise ValueError(
            f"Rigid-body template shape {body_offsets.shape} does not match sites {space_positions.shape}"
        )
    if len(body_offsets) < 2:
        return np.asarray([1.0, 0.0, 0.0, 0.0])

    center = np.asarray(center, dtype=float)
    box_l = np.asarray(box_l, dtype=float)
    space_offsets = space_positions - center
    space_offsets -= box_l * np.round(space_offsets / box_l)
    rotation = _kabsch_rotation(body_offsets, space_offsets)
    rmsd = float(np.sqrt(np.mean(np.sum(((rotation @ body_offsets.T).T - space_offsets) ** 2, axis=1))))
    if rmsd > 1.0e-3:
        raise ValueError(
            f"Rigid-body template alignment RMSD is {rmsd:.6g} nm; "
            "dataset and rigid_bodies_info.json are inconsistent"
        )
    if rmsd > 1.0e-5:
        print(f"[WARNING] Rigid-body template alignment RMSD is {rmsd:.6g} nm")
    return matrix_to_espresso_quat(rotation)
