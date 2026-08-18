#!/usr/bin/env python3
"""Run an iso-configurational TEL22 NVE temperature/amplitude sweep."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from analyze_temperature_sweep import build_summary, collect, print_summary
from make_scaled_checkpoint import rescale_checkpoint, sha256_file


def temperature_label(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-12):
        return str(int(rounded))
    return f"{value:.8g}".replace(".", "p")


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Executable not found: {value}")


def required_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def main() -> int:
    here = Path(__file__).resolve().parent
    root_default = here.parents[1]
    tutorial_default = root_default / "tutorials" / "tel22"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", type=Path, default=root_default)
    parser.add_argument("--tutorial-dir", type=Path, default=tutorial_default)
    parser.add_argument("--pypresso", default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--source-temperature-k", type=float, default=300.0)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[300.0, 100.0, 30.0])
    parser.add_argument("--precisions", nargs="+", choices=("float32", "float64"), default=["float32"])
    parser.add_argument("--dts", type=float, nargs="+", default=[0.001, 0.0015, 0.002, 0.003, 0.004, 0.005])
    parser.add_argument("--duration-ps", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--slope-min", type=float, default=1.7)
    parser.add_argument("--slope-max", type=float, default=2.3)
    parser.add_argument("--min-r2", type=float, default=0.97)
    parser.add_argument("--max-relative-drift", type=float, default=1e-4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.source_temperature_k <= 0.0:
        raise ValueError("--source-temperature-k must be positive")
    if len(args.temperatures) < 2 or any(t <= 0.0 for t in args.temperatures):
        raise ValueError("--temperatures needs at least two positive values")
    if len(set(args.temperatures)) != len(args.temperatures):
        raise ValueError("--temperatures must be unique")
    if len(args.dts) < 3 or any(dt <= 0.0 for dt in args.dts):
        raise ValueError("--dts needs at least three positive values")
    if args.duration_ps <= 0.0:
        raise ValueError("--duration-ps must be positive")

    root = args.framework_root.expanduser().resolve()
    tutorial = args.tutorial_dir.expanduser().resolve()
    certifier = required_file(root / "simulation" / "certify_nve.py", "NVE certifier")
    model = required_file(tutorial / "tel22_model.pt", "TEL22 model")
    config = required_file(tutorial / "tel22_training_config.json", "TEL22 config")
    priors = required_file(tutorial / "cg_priors.json", "TEL22 priors")
    rb_info = required_file(tutorial / "rigid_bodies_info.json", "rigid-body info")
    dataset = required_file(tutorial / "tel22_dataset.bin", "TEL22 dataset")
    source_checkpoint = required_file(
        args.source_checkpoint or (tutorial / "equilibrated.npz"), "source checkpoint"
    )
    manifest = required_file(Path(str(model) + ".manifest.json"), "TEL22 model manifest")

    default_pypresso = root / "espresso" / "build" / "pypresso"
    pypresso = resolve_executable(args.pypresso or (str(default_pypresso) if default_pypresso.exists() else "pypresso"))
    output_dir = args.output_dir.expanduser().resolve()

    print("[TEL22 ISO-CONFIGURATIONAL TEMPERATURE NVE]")
    print(f"Hamiltonian        : {priors.name} + {model.name} (PaiNN active)")
    print(f"source checkpoint  : {source_checkpoint}")
    print(f"source T label     : {args.source_temperature_k:g} K")
    print("changed variable   : translational v and body-frame omega scaled by sqrt(T/T_source)")
    print("positions/quats    : identical at every temperature")
    print(f"temperatures [K]   : {' '.join(f'{x:g}' for x in args.temperatures)}")
    print(f"precisions         : {' '.join(args.precisions)}")
    print(f"dt grid [ps]       : {' '.join(f'{x:g}' for x in args.dts)}")
    print(f"duration / dt      : {args.duration_ps:g} ps")
    print("device/search      : cpu / link-cell")
    print(f"output             : {output_dir}")
    print("scope              : amplitude diagnostic; NOT separately equilibrated canonical states")

    checkpoints_dir = output_dir / "checkpoints"
    generated: dict[float, Path] = {}
    checkpoint_summaries: list[dict[str, object]] = []
    source_hash = sha256_file(source_checkpoint)

    for temperature in args.temperatures:
        if math.isclose(temperature, args.source_temperature_k, rel_tol=0.0, abs_tol=1e-12):
            generated[temperature] = source_checkpoint
            checkpoint_summaries.append({
                "target_temperature_K": temperature,
                "checkpoint": str(source_checkpoint),
                "checkpoint_sha256": source_hash,
                "velocity_scale": 1.0,
                "source_checkpoint_reused_exactly": True,
            })
            continue
        checkpoint = checkpoints_dir / f"equilibrated_T{temperature_label(temperature)}K.npz"
        generated[temperature] = checkpoint
        if args.dry_run:
            checkpoint_summaries.append({
                "target_temperature_K": temperature,
                "checkpoint": str(checkpoint),
                "velocity_scale": math.sqrt(temperature / args.source_temperature_k),
                "source_checkpoint_reused_exactly": False,
                "planned_only": True,
            })
            continue
        if checkpoint.exists() and not args.overwrite:
            with __import__("numpy").load(checkpoint, allow_pickle=False) as chk:
                meta = json.loads(str(__import__("numpy").asarray(chk["metadata_json"]).item()))
            diag = meta.get("temperature_sweep_diagnostic", {})
            if (
                diag.get("source_checkpoint_sha256") != source_hash
                or not math.isclose(float(diag.get("source_temperature_K", math.nan)), args.source_temperature_k)
                or not math.isclose(float(diag.get("target_temperature_K", math.nan)), temperature)
            ):
                raise RuntimeError(
                    f"Existing derived checkpoint does not match this sweep: {checkpoint}. "
                    "Use --overwrite to regenerate it."
                )
            checkpoint_summaries.append({
                "target_temperature_K": temperature,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "velocity_scale": float(diag["velocity_scale_sqrt_T_ratio"]),
                "source_checkpoint_reused_exactly": False,
                "reused_existing": True,
            })
        else:
            checkpoint_summaries.append(
                rescale_checkpoint(
                    source_checkpoint,
                    checkpoint,
                    source_temperature_k=args.source_temperature_k,
                    target_temperature_k=temperature,
                )
            )

    plans: list[dict[str, object]] = []
    completed_with_strict_fail: list[str] = []
    for precision in args.precisions:
        for temperature in args.temperatures:
            label = f"T{temperature_label(temperature)}K_{precision}"
            run_output = output_dir / label
            command = [
                sys.executable,
                str(certifier),
                "--pypresso", pypresso,
                "--model", str(model),
                "--config", str(config),
                "--priors", str(priors),
                "--rb-info", str(rb_info),
                "--dataset", str(dataset),
                "--checkpoint", str(generated[temperature]),
                "--dts", *[f"{dt:.17g}" for dt in args.dts],
                "--duration-ps", f"{args.duration_ps:.17g}",
                "--device", "cpu",
                "--ml-precision", precision,
                "--neighbor-search", "link-cell",
                "--output-dir", str(run_output),
                "--slope-min", f"{args.slope_min:.17g}",
                "--slope-max", f"{args.slope_max:.17g}",
                "--min-r2", f"{args.min_r2:.17g}",
                "--max-relative-drift", f"{args.max_relative_drift:.17g}",
            ]
            if args.dry_run:
                command.append("--dry-run")
            elif args.overwrite:
                command.append("--overwrite")
            elif args.resume:
                command.append("--reuse-existing")
            plans.append({"temperature_K": temperature, "precision": precision, "command": command})
            print(f"\n[RUN] {label}")
            if args.dry_run:
                print("      " + " ".join(command))
                continue
            completed = subprocess.run(command, cwd=tutorial, check=False)
            if completed.returncode == 2:
                completed_with_strict_fail.append(label)
                print(f"[NOTE] {label}: trajectories completed but strict certification returned FAIL; continuing diagnostic sweep.")
            elif completed.returncode != 0:
                raise RuntimeError(f"NVE certifier failed operationally for {label} with exit code {completed.returncode}")

    manifest_payload = {
        "kind": "tel22_iso_configurational_temperature_nve_plan",
        "scope": "same positions/orientations; v and omega scaled by sqrt(T/T_source)",
        "source_temperature_K": args.source_temperature_k,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_hash,
        "model_manifest_sha256": sha256_file(manifest),
        "temperatures_K": args.temperatures,
        "precisions": args.precisions,
        "dts_ps": args.dts,
        "duration_ps": args.duration_ps,
        "checkpoint_preparation": checkpoint_summaries,
        "runs": plans,
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "temperature_sweep_manifest.json"
    if args.dry_run:
        print(f"\n[DRY-RUN] manifest would be written to {manifest_path}")
        return 0
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    rows = collect(output_dir)
    summary = build_summary(rows)
    summary["manifest"] = str(manifest_path)
    summary["strict_fail_runs"] = completed_with_strict_fail
    summary_path = output_dir / "temperature_nve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"[MANIFEST] {manifest_path}")
    print(f"[SUMMARY]  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
