#!/usr/bin/env python3
"""Run one TEL22 command while sampling process-tree memory.

This is an external diagnostic: it does not change the simulated Hamiltonian,
the PaiNN bridge files, or the integration schedule. Allocator-related
environment variables are recorded so controlled A/B policies remain auditable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Sample:
    elapsed_seconds: float
    integration_step: int | None
    root_pid: int
    process_count: int
    rss_mib: float
    vsz_mib: float
    physical_footprint_mib: float | None
    swap_used_mib: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and measure TEL22/MPS process-memory growth."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--energy-file", type=Path, required=True)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("--print-every-samples", type=int, default=15)
    parser.add_argument("--footprint-every-samples", type=int, default=15)
    parser.add_argument("--warmup-step", type=int, default=500)
    parser.add_argument("--growth-threshold-mib", type=float, default=1024.0)
    parser.add_argument("--slope-threshold-mib-per-1000-steps", type=float, default=256.0)
    parser.add_argument(
        "--abort-memory-mib", type=float, default=0.0,
        help=(
            "Send SIGINT when process-tree RSS or macOS physical footprint reaches "
            "this value; 0 disables the guard."
        ),
    )
    parser.add_argument(
        "--input", action="append", default=[], metavar="ROLE=PATH",
        help="Hash a provenance input into the final report; repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    if args.expected_final_step <= 0:
        parser.error("--expected-final-step must be positive")
    if args.sample_interval_seconds <= 0:
        parser.error("--sample-interval-seconds must be positive")
    if args.print_every_samples <= 0 or args.footprint_every_samples <= 0:
        parser.error("sample cadences must be positive")
    if args.warmup_step < 0:
        parser.error("--warmup-step must be non-negative")
    if args.growth_threshold_mib < 0 or args.slope_threshold_mib_per_1000_steps < 0:
        parser.error("growth thresholds must be non-negative")
    if args.abort_memory_mib < 0:
        parser.error("--abort-memory-mib must be non-negative")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_provenance_inputs(items: Iterable[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --input {item!r}; expected ROLE=PATH")
        role, raw_path = item.split("=", 1)
        if not role or role in result:
            raise ValueError(f"empty or duplicate provenance role: {role!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing provenance input {role}: {path}")
        result[role] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def read_last_energy_step(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end == 0:
                return None
            pos = end - 1
            while pos > 0:
                handle.seek(pos)
                if handle.read(1) == b"\n" and pos < end - 1:
                    break
                pos -= 1
            handle.seek(pos + (1 if pos > 0 else 0))
            last = handle.readline().decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        return None
    if not last or last.lower().startswith("step"):
        return None
    try:
        return int(last.split(",", 1)[0])
    except (ValueError, IndexError):
        return None


def parse_ps_rows(text: str) -> dict[int, tuple[int, int, int]]:
    rows: dict[int, tuple[int, int, int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            pid, ppid, rss_kib, vsz_kib = map(int, fields[:4])
        except ValueError:
            continue
        rows[pid] = (ppid, rss_kib, vsz_kib)
    return rows


def process_tree_memory(root_pid: int) -> tuple[int, float, float]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,vsz="],
        check=False, capture_output=True, text=True,
    )
    rows = parse_ps_rows(completed.stdout)
    if root_pid not in rows:
        return 0, 0.0, 0.0
    children: dict[int, list[int]] = {}
    for pid, (ppid, _rss, _vsz) in rows.items():
        children.setdefault(ppid, []).append(pid)
    members: list[int] = []
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen or pid not in rows:
            continue
        seen.add(pid)
        members.append(pid)
        pending.extend(children.get(pid, []))
    rss_kib = sum(rows[pid][1] for pid in members)
    vsz_kib = sum(rows[pid][2] for pid in members)
    return len(members), rss_kib / 1024.0, vsz_kib / 1024.0


_SIZE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT])(?:i?B)?", re.IGNORECASE)


def parse_size_to_mib(value: str) -> float:
    match = _SIZE_RE.search(value)
    if not match:
        raise ValueError(f"cannot parse memory size: {value!r}")
    number = float(match.group(1))
    factor = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0, "T": 1024.0**2}
    return number * factor[match.group(2).upper()]


def macos_physical_footprint_mib(pid: int) -> float | None:
    if sys.platform != "darwin":
        return None
    try:
        completed = subprocess.run(
            ["vmmap", "-summary", str(pid)], check=False,
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"^Physical footprint:\s*(.+)$", completed.stdout, re.MULTILINE)
    if not match:
        return None
    try:
        return parse_size_to_mib(match.group(1))
    except ValueError:
        return None


def macos_swap_used_mib() -> float | None:
    if sys.platform != "darwin":
        return None
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], check=False,
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"used\s*=\s*([^\s]+)", completed.stdout)
    if not match:
        return None
    try:
        return parse_size_to_mib(match.group(1))
    except ValueError:
        return None


def physical_memory_mib() -> float | None:
    if sys.platform == "darwin":
        try:
            value = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], check=True,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return int(value) / (1024.0**2)
        except (FileNotFoundError, ValueError, subprocess.SubprocessError):
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ValueError):
        pass
    return None


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator


def analyze_samples(
    samples: list[Sample], warmup_step: int,
    growth_threshold_mib: float, slope_threshold_mib_per_1000_steps: float,
) -> dict[str, object]:
    by_step: dict[int, Sample] = {}
    for sample in samples:
        if sample.integration_step is not None and sample.rss_mib > 0:
            by_step[sample.integration_step] = sample
    ordered = [by_step[key] for key in sorted(by_step)]
    post = [sample for sample in ordered if sample.integration_step >= warmup_step]
    if len(post) < 3:
        return {
            "classification": "inconclusive_too_few_post_warmup_samples",
            "distinct_steps": len(ordered), "post_warmup_distinct_steps": len(post),
        }
    def metrics(name: str, values: list[tuple[float, float]]) -> dict[str, object] | None:
        if len(values) < 3:
            return None
        steps = [item[0] for item in values]
        memory = [item[1] for item in values]
        slope = linear_slope(steps, memory)
        if slope is None:
            return None
        end_minus_start = memory[-1] - memory[0]
        peak_minus_start = max(memory) - memory[0]
        return {
            "metric": name,
            "first_step": int(steps[0]),
            "last_step": int(steps[-1]),
            "at_warmup_mib": memory[0],
            "at_last_step_mib": memory[-1],
            "peak_mib": max(memory),
            "end_minus_warmup_mib": end_minus_start,
            "peak_minus_warmup_mib": peak_minus_start,
            "slope_mib_per_1000_steps": slope * 1000.0,
        }

    rss_metrics = metrics(
        "process_tree_rss",
        [(float(x.integration_step), x.rss_mib) for x in post],
    )
    assert rss_metrics is not None
    footprint_metrics = metrics(
        "macos_physical_footprint",
        [
            (float(x.integration_step), float(x.physical_footprint_mib))
            for x in post if x.physical_footprint_mib is not None
        ],
    )
    available = [rss_metrics] + ([footprint_metrics] if footprint_metrics else [])
    sustained = [
        item["metric"] for item in available
        if max(item["end_minus_warmup_mib"], item["peak_minus_warmup_mib"])
        >= growth_threshold_mib
        and item["slope_mib_per_1000_steps"] >= slope_threshold_mib_per_1000_steps
    ]
    signals = [
        item["metric"] for item in available
        if max(item["end_minus_warmup_mib"], item["peak_minus_warmup_mib"])
        >= growth_threshold_mib
        or item["slope_mib_per_1000_steps"] >= slope_threshold_mib_per_1000_steps
    ]
    if sustained:
        classification = "sustained_process_memory_growth_observed"
        classification_basis = sustained
    elif signals:
        classification = "process_memory_growth_signal_observed"
        classification_basis = signals
    else:
        classification = "bounded_over_observed_window"
        classification_basis = [item["metric"] for item in available]
    swaps = [x.swap_used_mib for x in samples if x.swap_used_mib is not None]
    return {
        "classification": classification,
        "classification_basis": classification_basis,
        "distinct_steps": len(ordered),
        "post_warmup_distinct_steps": len(post),
        "first_post_warmup_step": rss_metrics["first_step"],
        "last_step": rss_metrics["last_step"],
        "rss_at_warmup_mib": rss_metrics["at_warmup_mib"],
        "rss_at_last_step_mib": rss_metrics["at_last_step_mib"],
        "peak_rss_mib": rss_metrics["peak_mib"],
        "rss_end_minus_warmup_mib": rss_metrics["end_minus_warmup_mib"],
        "rss_peak_minus_warmup_mib": rss_metrics["peak_minus_warmup_mib"],
        "rss_slope_mib_per_1000_steps": rss_metrics["slope_mib_per_1000_steps"],
        "physical_footprint": footprint_metrics,
        "peak_swap_used_mib": max(swaps) if swaps else None,
        "thresholds": {
            "warmup_step": warmup_step,
            "growth_mib": growth_threshold_mib,
            "slope_mib_per_1000_steps": slope_threshold_mib_per_1000_steps,
        },
        "caution": (
            "External RSS/physical-footprint growth cannot by itself distinguish live tensors, "
            "MPS allocator cache, Objective-C autorelease accumulation, or allocator fragmentation."
        ),
    }


def write_summary(
    path: Path, *, args: argparse.Namespace, samples: list[Sample],
    provenance: dict[str, dict[str, str]], returncode: int | None,
    interrupted: bool, safety_abort: bool,
) -> dict[str, object]:
    completed_step = read_last_energy_step(args.energy_file)
    analysis = analyze_samples(
        samples, args.warmup_step, args.growth_threshold_mib,
        args.slope_threshold_mib_per_1000_steps,
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "kind": "tel22_painn_mps_process_memory_diagnostic",
        "scope": (
            "External process-memory observation only; no Hamiltonian, model, precision, "
            "bridge file, or integration behavior is modified. Allocator-related "
            "environment variables are recorded explicitly."
        ),
        "platform": {"sys_platform": sys.platform, "physical_memory_mib": physical_memory_mib()},
        "command": args.command,
        "command_shell": shlex.join(args.command),
        "environment": {
            key: os.environ[key] for key in sorted(os.environ)
            if key.startswith(("PYTORCH_MPS_", "MLCG_MPS_"))
        },
        "provenance_inputs": provenance,
        "sampling": {
            "interval_seconds": args.sample_interval_seconds,
            "footprint_every_samples": args.footprint_every_samples,
            "physical_footprint_csv_semantics": (
                "A value is present only when vmmap was sampled on that row; blanks are not "
                "forward-filled."
            ),
            "samples": len(samples),
            "elapsed_seconds": samples[-1].elapsed_seconds if samples else None,
            "csv": str((args.output_dir / "process_memory_samples.csv").resolve()),
            "run_log": str((args.output_dir / "run.log").resolve()),
        },
        "run": {
            "returncode": returncode,
            "interrupted": interrupted,
            "safety_abort": safety_abort,
            "expected_final_step": args.expected_final_step,
            "completed_step": completed_step,
            "complete": returncode == 0 and completed_step == args.expected_final_step,
        },
        "memory_analysis": analysis,
        "diagnostic_only": True,
        "production_change_allowed": False,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.energy_file = args.energy_file.expanduser().resolve()
    provenance = parse_provenance_inputs(args.input)
    if args.dry_run:
        print("[DRY-RUN] command:", shlex.join(args.command))
        print("[DRY-RUN] output:", args.output_dir)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "process_memory_samples.csv"
    log_path = args.output_dir / "run.log"
    summary_path = args.output_dir / "mps_memory_summary.json"
    samples: list[Sample] = []
    interrupted = False
    safety_abort = False
    returncode: int | None = None

    print("[MPS MEMORY DIAGNOSTIC]")
    print("command :", shlex.join(args.command))
    print("samples :", csv_path)
    print("run log :", log_path)
    with log_path.open("w", encoding="utf-8") as log_handle, csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_handle:
        fields = list(Sample.__dataclass_fields__)
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        csv_handle.flush()
        process = subprocess.Popen(
            args.command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
        )
        start = time.monotonic()
        index = 0
        last_footprint: float | None = None
        last_swap: float | None = None
        try:
            while process.poll() is None:
                process_count, rss_mib, vsz_mib = process_tree_memory(process.pid)
                sampled_footprint: float | None = None
                if index % args.footprint_every_samples == 0:
                    last_footprint = macos_physical_footprint_mib(process.pid)
                    sampled_footprint = last_footprint
                    last_swap = macos_swap_used_mib()
                sample = Sample(
                    elapsed_seconds=time.monotonic() - start,
                    integration_step=read_last_energy_step(args.energy_file),
                    root_pid=process.pid,
                    process_count=process_count,
                    rss_mib=rss_mib,
                    vsz_mib=vsz_mib,
                    # Do not forward-fill vmmap measurements.  Forward-filling
                    # the pre-initialization value previously made it appear to
                    # be a real warmup-step sample and inflated the fitted MPS
                    # growth slope.
                    physical_footprint_mib=sampled_footprint,
                    swap_used_mib=last_swap,
                )
                samples.append(sample)
                writer.writerow(asdict(sample))
                csv_handle.flush()
                if index % args.print_every_samples == 0:
                    step = "?" if sample.integration_step is None else str(sample.integration_step)
                    footprint = "n/a" if last_footprint is None else f"{last_footprint:.1f} MiB"
                    print(
                        f"[MEM] t={sample.elapsed_seconds:.0f}s step={step} "
                        f"rss={sample.rss_mib:.1f} MiB footprint={footprint}",
                        flush=True,
                    )
                observed_memory = max(
                    rss_mib,
                    last_footprint if last_footprint is not None else 0.0,
                )
                if args.abort_memory_mib > 0 and observed_memory >= args.abort_memory_mib:
                    safety_abort = True
                    print(
                        f"[SAFETY] observed memory {observed_memory:.1f} MiB reached guard "
                        f"{args.abort_memory_mib:.1f} MiB; sending SIGINT.",
                        file=sys.stderr, flush=True,
                    )
                    process.send_signal(signal.SIGINT)
                    break
                index += 1
                time.sleep(args.sample_interval_seconds)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                returncode = process.wait(timeout=30)
        except KeyboardInterrupt:
            interrupted = True
            print("[INTERRUPT] forwarding SIGINT to monitored process", file=sys.stderr)
            process.send_signal(signal.SIGINT)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                returncode = process.wait(timeout=30)

    report = write_summary(
        summary_path, args=args, samples=samples, provenance=provenance,
        returncode=returncode, interrupted=interrupted, safety_abort=safety_abort,
    )
    memory = report["memory_analysis"]
    run = report["run"]
    print("[RESULT] run_complete:", run["complete"])
    print("[RESULT] memory:", memory["classification"])
    if "rss_slope_mib_per_1000_steps" in memory:
        print(
            "[RESULT] RSS slope: "
            f"{memory['rss_slope_mib_per_1000_steps']:.3f} MiB/1000 steps; "
            f"peak={memory['peak_rss_mib']:.1f} MiB"
        )
    print("[REPORT]", summary_path)
    if interrupted or safety_abort:
        return 130
    return 0 if returncode == 0 else int(returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
