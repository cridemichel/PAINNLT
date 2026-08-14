"""Generic convergence summaries for bonded IBI reports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _resolve_source_priors(raw: str, report_path: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = report_path.parent / path
    return path.resolve()


def _rows_from_report(report_path: Path) -> list[dict]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    metrics = report.get("metrics", [])
    if not metrics:
        raise ValueError(f"IBI report contains no evaluated iterations: {report_path}")

    rows = []
    for item in metrics:
        groups = item.get("groups", {})
        values = []
        by_kind: dict[str, list[float]] = {}
        for group in groups.values():
            if group.get("mode") != "ibi":
                continue
            value = float(group["distribution_l1"])
            values.append(value)
            by_kind.setdefault(str(group["kind"]), []).append(value)
        if not values:
            raise ValueError(f"Iteration {item.get('iteration')} contains no IBI L1 metrics")

        source_priors = _resolve_source_priors(str(item["source_priors"]), report_path)
        rows.append({
            "sampling_iteration": int(item["iteration"]),
            "source_priors": str(source_priors),
            "source_report": str(report_path),
            "mean_l1": sum(values) / len(values),
            "max_l1": max(values),
            "groups": len(values),
            "mean_l1_by_kind": {
                kind: sum(kind_values) / len(kind_values)
                for kind, kind_values in sorted(by_kind.items())
            },
        })
    return rows


def summarize_convergence(
    report_path: str | Path,
    output_path: str | Path,
    best_dir: str | Path,
    *,
    previous_reports=(),
    overwrite: bool = False,
) -> dict:
    """Combine one or more IBI reports and materialize the best evaluated priors."""
    report_path = Path(report_path).resolve()
    output_path = Path(output_path)
    best_dir = Path(best_dir)
    report_paths = [Path(path).resolve() for path in previous_reports] + [report_path]
    rows = []
    seen_iterations = set()
    for path in report_paths:
        for row in _rows_from_report(path):
            iteration = row["sampling_iteration"]
            if iteration in seen_iterations:
                raise ValueError(
                    f"Duplicate sampling iteration {iteration} while combining reports; "
                    "continuation runs must use a non-overlapping iteration offset"
                )
            seen_iterations.add(iteration)
            rows.append(row)
    rows.sort(key=lambda row: row["sampling_iteration"])

    best = min(rows, key=lambda row: (row["mean_l1"], row["max_l1"], row["sampling_iteration"]))
    source_priors = Path(best["source_priors"])
    if not source_priors.is_file():
        raise FileNotFoundError(f"Best evaluated source priors do not exist: {source_priors}")
    source_dir = source_priors.parent

    if best_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Best-priors directory already exists: {best_dir}; use overwrite=True")
        shutil.rmtree(best_dir)
    shutil.copytree(source_dir, best_dir)
    best_priors = best_dir / source_priors.name
    if not best_priors.is_file():
        raise RuntimeError(f"Best-priors copy is incomplete: {best_priors}")

    summary = {
        "schema_version": 2,
        "metric": "unweighted mean distribution_l1 across pooled type=ibi groups",
        "important_semantics": (
            "sampling_iteration i evaluates source_priors before update i; therefore the best "
            "evaluated priors are taken from source_priors, not from iteration_i written after the update"
        ),
        "reports": [str(path) for path in report_paths],
        "iterations": rows,
        "best_sampling_iteration": best["sampling_iteration"],
        "best_mean_l1": best["mean_l1"],
        "best_max_l1": best["max_l1"],
        "best_source_report": best["source_report"],
        "best_source_priors": str(source_priors),
        "best_priors": str(best_priors.resolve()),
        "final_priors_are_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def print_convergence_summary(summary: dict, output_path: str | Path | None = None) -> None:
    print("[IBI CONVERGENCE SUMMARY]")
    for row in summary["iterations"]:
        kind_text = " ".join(
            f"{kind}={value:.6f}" for kind, value in row["mean_l1_by_kind"].items()
        )
        print(
            f"sampling {row['sampling_iteration']:>2d}: mean={row['mean_l1']:.6f} "
            f"max={row['max_l1']:.6f} {kind_text}"
        )
    print(
        f"[BEST] sampling iteration {summary['best_sampling_iteration']} evaluated "
        f"{summary['best_source_priors']} with mean L1={summary['best_mean_l1']:.6f}"
    )
    print(f"[BEST] Self-contained priors copied to: {summary['best_priors']}")
    print("[NOTE] The final post-update priors have not yet been sampled/evaluated in the newest run.")
    if output_path is not None:
        print(f"[DONE] Summary: {output_path}")
