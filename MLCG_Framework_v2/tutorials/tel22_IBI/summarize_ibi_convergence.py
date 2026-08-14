#!/usr/bin/env python3
"""Summarize TEL22 bonded-IBI convergence and materialize the best evaluated priors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ibi"))

from convergence import print_convergence_summary, summarize_convergence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--previous-report",
        type=Path,
        action="append",
        default=[],
        help="Earlier IBI report to include in the same convergence history; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = summarize_convergence(
        args.report,
        args.output,
        args.best_dir,
        previous_reports=args.previous_report,
        overwrite=args.overwrite,
    )
    print_convergence_summary(summary, args.output)


if __name__ == "__main__":
    main()
