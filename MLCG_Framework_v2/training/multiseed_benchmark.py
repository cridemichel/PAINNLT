#!/usr/bin/env python3
"""Run reproducible multi-seed PaiNN training benchmarks and summarize them.

The tool is intentionally dataset-agnostic.  Each ``--case`` supplies a label
and a dataset; every case is trained with the same base configuration and the
same list of split seeds.  Per-run configs/models/logs/manifests are isolated
under the output directory and a paired comparison is emitted when exactly two
cases are supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EPOCH_RE = re.compile(r"(?:Epoca|Epoch)\s*\[(\d+)/(\d+)\]", re.IGNORECASE)
VAL_RE = re.compile(r"\[VAL\]\s+Loss:\s*([0-9eE+\-.]+)")


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    dataset: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training_log(lines: Iterable[str]) -> tuple[int | None, float | None]:
    """Return (best_epoch, best_validation_loss) from trainer console output."""
    current_epoch: int | None = None
    best_epoch: int | None = None
    best_loss: float | None = None
    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        val_match = VAL_RE.search(line)
        if not val_match:
            continue
        value = float(val_match.group(1))
        if not math.isfinite(value):
            raise ValueError(f"Non-finite validation loss in training log: {value}")
        if current_epoch is None:
            raise ValueError("Validation loss encountered before an epoch marker")
        if best_loss is None or value < best_loss:
            best_loss = value
            best_epoch = current_epoch
    return best_epoch, best_loss


def manifest_loss_matches_console(log_loss: float, manifest_loss: float) -> bool:
    """Compare a full-precision manifest loss with the trainer's console value.

    The C++ trainer prints floating-point losses with the default stream
    precision (six significant digits), whereas the model manifest stores the
    full float value.  Compare against the value the manifest loss would have
    produced at that console precision instead of imposing a tighter numerical
    tolerance than the log can represent.
    """
    if not math.isfinite(log_loss) or not math.isfinite(manifest_loss):
        return False
    manifest_as_printed = float(format(manifest_loss, ".6g"))
    return log_loss == manifest_as_printed


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize_records(records: Sequence[dict], case_order: Sequence[str]) -> dict:
    if not records:
        raise ValueError("No benchmark records to summarize")
    by_case: dict[str, list[dict]] = {label: [] for label in case_order}
    seen: set[tuple[str, int]] = set()
    for record in records:
        label = str(record["case"])
        seed = int(record["seed"])
        if label not in by_case:
            raise ValueError(f"Unexpected benchmark case {label!r}")
        key = (label, seed)
        if key in seen:
            raise ValueError(f"Duplicate benchmark record for case={label!r}, seed={seed}")
        seen.add(key)
        by_case[label].append(record)

    aggregates: dict[str, dict] = {}
    seed_sets: dict[str, set[int]] = {}
    for label in case_order:
        rows = sorted(by_case[label], key=lambda r: int(r["seed"]))
        if not rows:
            raise ValueError(f"No records for case {label!r}")
        losses = [float(row["best_validation_loss"]) for row in rows]
        mean, std = _mean_std(losses)
        aggregates[label] = {
            "n": len(rows),
            "mean_best_validation_loss": mean,
            "sample_std_best_validation_loss": std,
            "min_best_validation_loss": min(losses),
            "max_best_validation_loss": max(losses),
        }
        seed_sets[label] = {int(row["seed"]) for row in rows}

    paired = None
    if len(case_order) == 2:
        reference, candidate = case_order
        if seed_sets[reference] != seed_sets[candidate]:
            raise ValueError("Two-case benchmark must use identical paired seed sets")
        ref_by_seed = {int(r["seed"]): r for r in by_case[reference]}
        cand_by_seed = {int(r["seed"]): r for r in by_case[candidate]}
        deltas = []
        pairs = []
        for seed in sorted(seed_sets[reference]):
            ref_loss = float(ref_by_seed[seed]["best_validation_loss"])
            cand_loss = float(cand_by_seed[seed]["best_validation_loss"])
            delta = cand_loss - ref_loss
            deltas.append(delta)
            pairs.append({
                "seed": seed,
                "reference_loss": ref_loss,
                "candidate_loss": cand_loss,
                "candidate_minus_reference": delta,
            })
        delta_mean, delta_std = _mean_std(deltas)
        paired = {
            "reference_case": reference,
            "candidate_case": candidate,
            "pairs": pairs,
            "mean_candidate_minus_reference": delta_mean,
            "sample_std_candidate_minus_reference": delta_std,
            "candidate_wins": sum(delta < 0.0 for delta in deltas),
            "reference_wins": sum(delta > 0.0 for delta in deltas),
            "ties": sum(delta == 0.0 for delta in deltas),
        }

    return {
        "schema_version": 1,
        "kind": "painn_multiseed_benchmark",
        "case_order": list(case_order),
        "records": sorted(records, key=lambda r: (case_order.index(str(r["case"])), int(r["seed"]))),
        "aggregates": aggregates,
        "paired_comparison": paired,
    }


def _validate_label(label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(
            f"Invalid case label {label!r}; use only letters, digits, '.', '_' or '-'"
        )


def _run_streaming(command: Sequence[str], log_path: Path) -> list[str]:
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return lines


def _load_completed_record(
    *,
    case: BenchmarkCase,
    seed: int,
    expected_config: dict,
    config_path: Path,
    model_path: Path,
    log_path: Path,
    manifest_path: Path,
) -> dict:
    """Validate and materialize one already-completed benchmark run.

    Resume is deliberately fail-closed: an existing complete run is reused
    only when config, dataset, model, log and manifest are mutually consistent.
    """
    required = (config_path, model_path, log_path, manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete benchmark run; missing: " + ", ".join(missing))

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if saved_config != expected_config:
        raise ValueError(
            f"Existing config mismatch for case={case.label}, seed={seed}; "
            "refusing to reuse a run produced with different settings"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_loss = float(manifest["best_validation_loss"])
    if int(manifest.get("split_seed", -1)) != seed:
        raise ValueError(
            f"Manifest split_seed mismatch for case={case.label}: "
            f"expected {seed}, got {manifest.get('split_seed')}"
        )

    expected_dataset_sha = sha256_file(case.dataset)
    if str(manifest.get("dataset_sha256", "")) != expected_dataset_sha:
        raise ValueError(
            f"Manifest dataset hash mismatch for case={case.label}, seed={seed}"
        )
    expected_config_sha = sha256_file(config_path)
    if str(manifest.get("config_sha256", "")) != expected_config_sha:
        raise ValueError(
            f"Manifest config hash mismatch for case={case.label}, seed={seed}"
        )
    expected_model_sha = sha256_file(model_path)
    if str(manifest.get("model_sha256", "")) != expected_model_sha:
        raise ValueError(
            f"Manifest model hash mismatch for case={case.label}, seed={seed}"
        )

    lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
    best_epoch, parsed_loss = parse_training_log(lines)
    if parsed_loss is None or best_epoch is None:
        raise ValueError(
            f"Could not parse validation history for case={case.label}, seed={seed}"
        )
    if not manifest_loss_matches_console(parsed_loss, manifest_loss):
        raise ValueError(
            f"Manifest/log best validation mismatch for case={case.label}, seed={seed}: "
            f"manifest={manifest_loss}, log={parsed_loss}"
        )

    return {
        "case": case.label,
        "seed": seed,
        "best_validation_loss": manifest_loss,
        "best_epoch": best_epoch,
        "dataset_path": str(case.dataset),
        "dataset_sha256": expected_dataset_sha,
        "config_path": str(config_path),
        "config_sha256": expected_config_sha,
        "model_path": str(model_path),
        "model_sha256": expected_model_sha,
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
    }


def _write_csv(path: Path, records: Sequence[dict]) -> None:
    fields = [
        "case", "seed", "best_validation_loss", "best_epoch",
        "dataset_path", "dataset_sha256", "config_path", "config_sha256",
        "model_path", "model_sha256", "manifest_path", "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})


def _print_summary(summary: dict) -> None:
    print("\n[MULTI-SEED TRAINING SUMMARY]")
    for label in summary["case_order"]:
        agg = summary["aggregates"][label]
        print(
            f"{label:>12s}: n={agg['n']} "
            f"mean={agg['mean_best_validation_loss']:.6f} "
            f"std={agg['sample_std_best_validation_loss']:.6f} "
            f"min={agg['min_best_validation_loss']:.6f} "
            f"max={agg['max_best_validation_loss']:.6f}"
        )
    paired = summary.get("paired_comparison")
    if paired is not None:
        print("\n[PAIRED SEED COMPARISON]")
        ref = paired["reference_case"]
        cand = paired["candidate_case"]
        for row in paired["pairs"]:
            print(
                f"seed={row['seed']:>6d} {ref}={row['reference_loss']:.6f} "
                f"{cand}={row['candidate_loss']:.6f} "
                f"delta({cand}-{ref})={row['candidate_minus_reference']:+.6f}"
            )
        print(
            f"paired mean delta={paired['mean_candidate_minus_reference']:+.6f} "
            f"std={paired['sample_std_candidate_minus_reference']:.6f} | "
            f"{cand} wins={paired['candidate_wins']} "
            f"{ref} wins={paired['reference_wins']} ties={paired['ties']}"
        )
        print("[NOTE] Lower validation loss is better; negative paired delta favors the candidate case.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, type=Path)
    parser.add_argument("--manifest-tool", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--case", action="append", nargs=2, metavar=("LABEL", "DATASET"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    if not args.seeds:
        raise ValueError("At least one seed is required")

    trainer = args.trainer.resolve()
    manifest_tool = args.manifest_tool.resolve()
    base_config_path = args.base_config.resolve()
    for path, description in (
        (trainer, "trainer"),
        (manifest_tool, "manifest tool"),
        (base_config_path, "base config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if not trainer.stat().st_mode & 0o111:
        raise PermissionError(f"Trainer is not executable: {trainer}")

    cases: list[BenchmarkCase] = []
    labels: set[str] = set()
    for raw_label, raw_dataset in args.case:
        _validate_label(raw_label)
        if raw_label in labels:
            raise ValueError(f"Duplicate case label: {raw_label}")
        labels.add(raw_label)
        dataset = Path(raw_dataset).resolve()
        if not dataset.is_file():
            raise FileNotFoundError(f"Missing dataset for case {raw_label!r}: {dataset}")
        cases.append(BenchmarkCase(raw_label, dataset))

    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("Base training config must be a JSON object")

    provenance = {
        "schema_version": 1,
        "kind": "painn_multiseed_benchmark_inputs",
        "base_config_path": str(base_config_path),
        "base_config_sha256": sha256_file(base_config_path),
        "seeds": list(args.seeds),
        "cases": [
            {
                "label": case.label,
                "dataset_path": str(case.dataset),
                "dataset_sha256": sha256_file(case.dataset),
                "dataset_file_size_bytes": case.dataset.stat().st_size,
            }
            for case in cases
        ],
    }
    output_dir = args.output_dir.resolve()
    inputs_path = output_dir / "benchmark_inputs.json"
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            inputs_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        elif args.resume:
            if not inputs_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume benchmark without provenance file: {inputs_path}"
                )
            previous = json.loads(inputs_path.read_text(encoding="utf-8"))
            if previous != provenance:
                raise ValueError(
                    "Benchmark resume provenance mismatch: requested seeds/config/datasets "
                    "do not match the existing benchmark_inputs.json"
                )
            print(f"[RESUME] Verified benchmark provenance: {inputs_path}")
        else:
            raise FileExistsError(
                f"Benchmark output already exists: {output_dir}. "
                "Use --resume to reuse verified completed runs or --overwrite to restart."
            )
    else:
        output_dir.mkdir(parents=True)
        inputs_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    records: list[dict] = []
    for case in cases:
        for seed in args.seeds:
            run_dir = output_dir / case.label / f"seed_{seed}"
            config_path = run_dir / "training_config.json"
            model_path = run_dir / "model.pt"
            log_path = run_dir / "training.log"
            manifest_path = Path(f"{model_path}.manifest.json")

            config = dict(base_config)
            config["split_seed"] = int(seed)

            if args.resume and run_dir.exists():
                required = (config_path, model_path, log_path, manifest_path)
                if all(path.is_file() for path in required):
                    record = _load_completed_record(
                        case=case,
                        seed=seed,
                        expected_config=config,
                        config_path=config_path,
                        model_path=model_path,
                        log_path=log_path,
                        manifest_path=manifest_path,
                    )
                    records.append(record)
                    print(
                        f"[RESUME] Reusing verified completed run case={case.label} "
                        f"seed={seed} best={record['best_validation_loss']:.9g}"
                    )
                    continue
                present = [path.name for path in required if path.exists()]
                print(
                    f"[RESUME] Incomplete run case={case.label} seed={seed}; "
                    f"retraining this run (present: {', '.join(present) or 'none'})."
                )
                shutil.rmtree(run_dir)

            run_dir.mkdir(parents=True, exist_ok=False)
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            print(f"\n[RUN] case={case.label} seed={seed}")
            lines = _run_streaming(
                [str(trainer), str(case.dataset), str(model_path), str(config_path)],
                log_path,
            )
            subprocess.run(
                [
                    args.python,
                    str(manifest_tool),
                    "--model", str(model_path),
                    "--dataset", str(case.dataset),
                    "--config", str(config_path),
                ],
                check=True,
            )

            record = _load_completed_record(
                case=case,
                seed=seed,
                expected_config=config,
                config_path=config_path,
                model_path=model_path,
                log_path=log_path,
                manifest_path=manifest_path,
            )
            records.append(record)

    summary = summarize_records(records, [case.label for case in cases])
    summary["inputs"] = provenance
    json_path = output_dir / "benchmark_summary.json"
    csv_path = output_dir / "benchmark_runs.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, summary["records"])
    _print_summary(summary)
    print(f"\n[DONE] Benchmark JSON: {json_path}")
    print(f"[DONE] Benchmark CSV:  {csv_path}")


if __name__ == "__main__":
    main()
