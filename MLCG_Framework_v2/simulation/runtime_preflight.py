#!/usr/bin/env python3
"""Fail-closed provenance check for a complete MLCG runtime Hamiltonian.

This check binds a trained PaiNN model to the exact residual dataset/config used
at training time and, when supplied, to the residual-build provenance that
certifies the selected priors and rigid-body metadata.  It is intentionally
system-agnostic: tutorial wrappers only provide paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import residual_input_provenance as residual_provenance  # noqa: E402

SCHEMA_VERSION = 1
FRAMEWORK = "MLCG_Framework_v2"
KIND = "runtime_hamiltonian_preflight"
MODEL_MANIFEST_SCHEMA_VERSION = 3
ENERGY_GAUGE = "isolated_species_zero_v1"


def canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def sha256_file(path: str | Path) -> str:
    path = canonical(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(value: Any, manifest_path: Path) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Manifest {manifest_path} contains an empty path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _require_artifact(label: str, path: str | Path) -> Path:
    resolved = canonical(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _check_manifest_artifact(
    label: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    selected: Path,
    *,
    path_key: str,
    size_key: str,
    hash_key: str,
) -> None:
    recorded_path = _resolve_manifest_path(manifest.get(path_key), manifest_path)
    if recorded_path != selected:
        raise ValueError(
            f"{label} path mismatch: model manifest={recorded_path}, selected={selected}"
        )
    recorded_size = manifest.get(size_key)
    if recorded_size is None or int(recorded_size) != selected.stat().st_size:
        raise ValueError(
            f"{label} size mismatch: model manifest={recorded_size}, current={selected.stat().st_size}"
        )
    recorded_hash = manifest.get(hash_key)
    current_hash = sha256_file(selected)
    if not recorded_hash or recorded_hash != current_hash:
        raise ValueError(
            f"{label} SHA256 mismatch: model manifest={recorded_hash}, current={current_hash}"
        )


def _architecture_from_config(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "architecture_variant", "num_species", "hidden_channels", "n_layers",
        "num_rbf", "cutoff",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Runtime config is missing architecture fields: {missing}")
    return {
        "variant": str(config["architecture_variant"]),
        "num_species": int(config["num_species"]),
        "hidden_channels": int(config["hidden_channels"]),
        "n_layers": int(config["n_layers"]),
        "num_rbf": int(config["num_rbf"]),
        "cutoff": float(config["cutoff"]),
        "toxvaerd_alpha": float(config.get("toxvaerd_alpha", 0.1)),
    }


def _check_architecture(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    expected = _architecture_from_config(config)
    recorded = manifest.get("architecture")
    if not isinstance(recorded, dict):
        raise ValueError("Model manifest has no architecture map")
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual = recorded.get(key)
        if isinstance(expected_value, float):
            if actual is None or not math.isclose(
                float(actual), expected_value, rel_tol=1e-6, abs_tol=1e-8
            ):
                mismatches.append(f"{key}: manifest={actual}, config={expected_value}")
        elif isinstance(expected_value, int):
            if actual is None or int(actual) != expected_value:
                mismatches.append(f"{key}: manifest={actual}, config={expected_value}")
        elif str(actual) != expected_value:
            mismatches.append(f"{key}: manifest={actual!r}, config={expected_value!r}")
    if mismatches:
        raise ValueError("Model architecture/config mismatch:\n  - " + "\n  - ".join(mismatches))


def check_runtime_preflight(
    *,
    model: str | Path,
    config: str | Path,
    dataset: str | Path,
    priors: str | Path,
    rb_info: str | Path,
    residual_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    model = _require_artifact("model", model)
    config = _require_artifact("config", config)
    dataset = _require_artifact("dataset", dataset)
    priors = _require_artifact("priors", priors)
    rb_info = _require_artifact("rigid-body metadata", rb_info)
    residual_manifest = _require_artifact("residual provenance manifest", residual_manifest)
    model_manifest_path = _require_artifact(
        "model manifest", Path(f"{model}.manifest.json")
    )

    model_manifest = json.loads(model_manifest_path.read_text())
    if int(model_manifest.get("schema_version", -1)) != MODEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model manifest schema: {model_manifest_path}")
    if model_manifest.get("framework") != FRAMEWORK:
        raise ValueError(f"Unexpected framework in model manifest: {model_manifest_path}")
    if model_manifest.get("energy_gauge") != ENERGY_GAUGE:
        raise ValueError(
            f"Unexpected energy gauge in model manifest: {model_manifest.get('energy_gauge')!r}"
        )

    runtime_config = json.loads(config.read_text())
    _check_architecture(model_manifest, runtime_config)
    _check_manifest_artifact(
        "model", model_manifest, model_manifest_path, model,
        path_key="model_path", size_key="model_file_size_bytes", hash_key="model_sha256",
    )
    _check_manifest_artifact(
        "dataset", model_manifest, model_manifest_path, dataset,
        path_key="dataset_path", size_key="dataset_file_size_bytes", hash_key="dataset_sha256",
    )
    _check_manifest_artifact(
        "config", model_manifest, model_manifest_path, config,
        path_key="config_path", size_key="config_file_size_bytes", hash_key="config_sha256",
    )

    residual = residual_provenance.check_manifest(
        manifest_path=residual_manifest,
        dataset=dataset,
        rb_info=rb_info,
        priors=priors,
    )
    residual_dataset_hash = residual["outputs"]["dataset"]["sha256"]
    if model_manifest.get("dataset_sha256") != residual_dataset_hash:
        raise ValueError(
            "Model was not trained on the residual dataset certified by the selected residual provenance"
        )

    prior_hashes = residual_provenance.referenced_prior_artifacts(priors)
    report = {
        "schema_version": SCHEMA_VERSION,
        "framework": FRAMEWORK,
        "kind": KIND,
        "artifacts": {
            "model": {"path": str(model), "sha256": sha256_file(model)},
            "model_manifest": {
                "path": str(model_manifest_path),
                "sha256": sha256_file(model_manifest_path),
            },
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
            "priors": {"path": str(priors), "sha256": sha256_file(priors)},
            "rb_info": {"path": str(rb_info), "sha256": sha256_file(rb_info)},
            "residual_manifest": {
                "path": str(residual_manifest),
                "sha256": sha256_file(residual_manifest),
            },
        },
        "prior_artifact_sha256": prior_hashes,
        "model_best_validation_loss": float(model_manifest["best_validation_loss"]),
        "architecture": _architecture_from_config(runtime_config),
        "residual_validation_mean_l1": float(
            residual["build_inputs"]["ibi_validation_mean_l1"]
        ),
        "residual_validation_max_l1": float(
            residual["build_inputs"]["ibi_validation_max_l1"]
        ),
        "pass": True,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[POST-IBI RUNTIME PREFLIGHT]")
    print(f"model    : {model}")
    print(f"dataset  : {dataset}")
    print(f"config   : {config}")
    print(f"priors   : {priors}")
    print(f"rb_info  : {rb_info}")
    print(f"model_sha256   = {report['artifacts']['model']['sha256']}")
    print(f"dataset_sha256 = {report['artifacts']['dataset']['sha256']}")
    print(f"priors_sha256  = {report['artifacts']['priors']['sha256']}")
    print("[PASS] Model, residual dataset, config, validated priors/tables and rigid-body metadata are provenance-consistent.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--rb-info", required=True)
    parser.add_argument("--residual-manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    check_runtime_preflight(
        model=args.model,
        config=args.config,
        dataset=args.dataset,
        priors=args.priors,
        rb_info=args.rb_info,
        residual_manifest=args.residual_manifest,
        output=args.output,
    )


if __name__ == "__main__":
    main()
