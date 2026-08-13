#!/usr/bin/env python3
"""Repair inline sparse NVE energy sampling in simulation/certify_nve.py.

Handles the actual source layout where the sampling interval is computed inline,
e.g. max(1, steps // 5), rather than assigned to a variable named log_every.

The script:
  * removes only the misplaced marker line introduced by the faulty v2 patcher;
  * parses the restored source;
  * locates sparse-sampling expressions based on `steps` (including multiline);
  * replaces those expressions with literal 1 without reformatting the file;
  * additionally patches values passed immediately after "--log_every"/"--log_interval";
  * validates Python syntax and the sigma_E protocol;
  * creates a one-time backup.

It is idempotent.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path.cwd()
TARGET = ROOT / "simulation" / "certify_nve.py"
BACKUP = TARGET.with_name(TARGET.name + ".pre_inline_sampling_repair")

BAD_MARKERS = (
    "NVE certification: sample energy every integration step",
    "NVE certification: sample total energy every integration step",
    "NVE certification: sample energy every step",
)


def fail(msg: str) -> "NoReturn":
    raise SystemExit(f"[ERROR] {msg}")


def get_source_segment(lines, node: ast.AST) -> tuple[int, int, str]:
    """Return absolute character offsets and original source segment for node."""
    starts = [0]
    total = 0
    for line in lines:
        total += len(line)
        starts.append(total)

    sline = node.lineno - 1
    eline = node.end_lineno - 1
    start = starts[sline] + node.col_offset
    end = starts[eline] + node.end_col_offset
    source = "".join(lines)
    return start, end, source[start:end]


def is_steps_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "steps"


def contains_steps(node: ast.AST) -> bool:
    return any(is_steps_name(child) for child in ast.walk(node))


def is_sparse_sampling_expr(node: ast.AST) -> bool:
    """Recognize the historical approximately-five-samples-per-run expression."""
    # max(1, <expression involving steps>)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "max":
            has_one = any(
                isinstance(arg, ast.Constant) and arg.value == 1
                for arg in node.args
            )
            if has_one and contains_steps(node):
                # Restrict to expressions that downsample by about five intervals.
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.BinOp)
                        and isinstance(child.op, ast.FloorDiv)
                        and contains_steps(child.left)
                        and isinstance(child.right, ast.Constant)
                        and child.right.value == 5
                    ):
                        return True
    # Direct steps // 5 fallback.
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.FloorDiv)
        and contains_steps(node.left)
        and isinstance(node.right, ast.Constant)
        and node.right.value == 5
    ):
        return True
    return False


def collect_sparse_exprs(tree: ast.AST) -> list[ast.AST]:
    """Return outermost matching expressions only."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    matches = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        if not is_sparse_sampling_expr(node):
            continue
        parent = parents.get(id(node))
        if parent is not None and is_sparse_sampling_expr(parent):
            continue
        matches.append(node)
    return matches


def patch_flag_values(source: str) -> tuple[str, int]:
    """Patch list/tuple values following --log_every or --log_interval to '1'.

    Uses AST spans so multiline list construction is handled safely.
    """
    try:
        tree = ast.parse(source, filename=str(TARGET))
    except SyntaxError as exc:
        fail(f"cannot parse before flag repair: line {exc.lineno}: {exc.msg}")

    lines = source.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        elts = node.elts
        for i, elt in enumerate(elts[:-1]):
            if (
                isinstance(elt, ast.Constant)
                and isinstance(elt.value, str)
                and elt.value in ("--log_every", "--log-every", "--log_interval", "--log-interval")
            ):
                value = elts[i + 1]
                # Preserve existing literal 1/"1".
                if isinstance(value, ast.Constant) and value.value in (1, "1"):
                    continue
                start, end, _ = get_source_segment(lines, value)
                # CLI list entries are strings. Use "1" for maximum compatibility.
                replacements.append((start, end, '"1"'))

    for start, end, repl in sorted(replacements, reverse=True):
        source = source[:start] + repl + source[end:]
    return source, len(replacements)


def main() -> int:
    if not TARGET.is_file():
        fail(f"missing {TARGET}")

    original = TARGET.read_text(encoding="utf-8")

    # 1. Remove only the line inserted by the faulty v2 patcher.
    cleaned_lines = []
    removed_bad = 0
    for line in original.splitlines(keepends=True):
        if "log_every" in line and any(marker in line for marker in BAD_MARKERS):
            removed_bad += 1
            continue
        cleaned_lines.append(line)
    source = "".join(cleaned_lines)

    try:
        tree = ast.parse(source, filename=str(TARGET))
    except SyntaxError as exc:
        fail(
            f"source remains invalid after removing faulty marker: "
            f"line {exc.lineno}: {exc.msg}\n"
            "Inspect with:\n"
            "  nl -ba simulation/certify_nve.py | sed -n '185,220p'"
        )

    # 2. Replace inline max(1, steps // 5) / steps // 5 expressions by 1.
    lines = source.splitlines(keepends=True)
    matches = collect_sparse_exprs(tree)
    replacements: list[tuple[int, int, str, str]] = []
    for node in matches:
        start, end, segment = get_source_segment(lines, node)
        replacements.append((start, end, "1", segment))

    for start, end, repl, _segment in sorted(replacements, reverse=True):
        source = source[:start] + repl + source[end:]

    # 3. Patch explicit CLI flag values, if the command list still supplies
    #    a sparse expression through a different source construct.
    source, flag_replacements = patch_flag_values(source)

    # 4. If the PLAN line uses a variable called log_every but there is no
    #    assignment, insert a safe assignment immediately after `steps = ...`.
    tree = ast.parse(source, filename=str(TARGET))
    load_log_every = any(
        isinstance(n, ast.Name)
        and n.id == "log_every"
        and isinstance(n.ctx, ast.Load)
        for n in ast.walk(tree)
    )
    store_log_every = any(
        isinstance(n, ast.Name)
        and n.id == "log_every"
        and isinstance(n.ctx, ast.Store)
        for n in ast.walk(tree)
    )

    inserted_assignment = False
    if load_log_every and not store_log_every:
        step_assigns = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) and t.id == "steps" for t in targets):
                    step_assigns.append(n)
        if not step_assigns:
            fail("log_every is referenced but no assignment to steps was found")
        # Prefer the last steps assignment, normally the per-dt run planner.
        n = sorted(step_assigns, key=lambda x: (x.lineno, x.end_lineno))[-1]
        src_lines = source.splitlines(keepends=True)
        indent = src_lines[n.lineno - 1][: len(src_lines[n.lineno - 1]) - len(src_lines[n.lineno - 1].lstrip())]
        src_lines.insert(
            n.end_lineno,
            f"{indent}log_every = 1  # NVE certification: sample total energy every integration step\n",
        )
        source = "".join(src_lines)
        inserted_assignment = True

    source = source.rstrip() + "\n"

    # 5. Final guardrails.
    try:
        final_tree = ast.parse(source, filename=str(TARGET))
    except SyntaxError as exc:
        fail(f"internal repair produced invalid Python at line {exc.lineno}: {exc.msg}")

    if "sigma_E" not in source:
        fail(
            "sampling syntax can be repaired, but sigma_E is absent; "
            "the sigma(E) patch is incomplete"
        )

    # Reject any surviving historical steps//5 sampler.
    for n in ast.walk(final_tree):
        if (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.FloorDiv)
            and contains_steps(n.left)
            and isinstance(n.right, ast.Constant)
            and n.right.value == 5
        ):
            fail(
                f"historical steps//5 sampling still survives near line {n.lineno}; "
                "no file was written"
            )

    if source == original:
        print(f"[SKIP] {TARGET}: already repaired")
        return 0

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"[BACKUP] {BACKUP}")

    TARGET.write_text(source, encoding="utf-8")
    print(f"[PATCH] {TARGET}")
    print(f"        removed faulty v2 lines: {removed_bad}")
    print(f"        inline sparse expressions -> 1: {len(replacements)}")
    for _start, _end, _repl, segment in replacements:
        print(f"          - {segment.strip()} -> 1")
    print(f"        CLI log flag values -> \"1\": {flag_replacements}")
    print(f"        inserted log_every assignment: {inserted_assignment}")
    print("[PASS] syntax valid; sigma_E present; steps//5 sampler absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
