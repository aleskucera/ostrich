#!/usr/bin/env python3
"""Convert `tree`-style text (UTF-8 box drawing) into LaTeX \\dirtree{...} syntax."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Maximum dirtree level to emit (editable hyperparameter).
# Examples:
# - 5 => include .1 to .5, skip .6 and deeper
# - None => no depth limit
MAX_NESTING_LEVEL = 5

# File extensions to omit from the dirtree (case-insensitive).
EXCLUDE_EXTENSIONS = {".yaml", ".yml"}

# Directory names whose entire subtree is omitted from the dirtree.
EXCLUDE_BRANCH_NAMES = {"conf"}

# Branch marker: ├ or └, two horizontal rules, optional space, then the entry name.
_BRANCH = re.compile(r"(├──|└──)\s*(.*)$")


def _parse_depth_name(line: str) -> tuple[int, str] | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None
    if line.strip() == ".":
        return 0, "."
    m = _BRANCH.search(line)
    if not m:
        return None
    prefix = line[: m.start()]
    name = m.group(2).rstrip()
    depth = 1 + len(prefix) // 4
    return depth, name


def _tex_escape_name(s: str) -> str:
    """Escape for dirtree cell text (outside \\detokenize)."""
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("#", r"\#")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("_", r"\_")
        .replace("$", r"\$")
        .replace("^", r"\textasciicircum{}")
        .replace("~", r"\textasciitilde{}")
    )


def _tex_cell(s: str) -> str:
    """Return dirtree-safe text without using \\detokenize delimiters."""
    return _tex_escape_name(s)


def _should_include(name: str) -> bool:
    return Path(name).suffix.lower() not in EXCLUDE_EXTENSIONS


def _filter_excluded_branches(parsed: list[tuple[int, str]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    skip_depth: int | None = None
    for depth, name in parsed:
        if skip_depth is not None and depth > skip_depth:
            continue
        skip_depth = None
        if name in EXCLUDE_BRANCH_NAMES:
            skip_depth = depth
            continue
        out.append((depth, name))
    return out


def tree_lines_to_dirtree(lines: list[str]) -> str:
    parsed: list[tuple[int, str]] = []
    for raw in lines:
        item = _parse_depth_name(raw)
        if item is not None:
            parsed.append(item)

    parsed = _filter_excluded_branches(parsed)

    if not parsed:
        return ""

    # dirtree uses .1 for the root row; each `tree` level under "." is one deeper
    # (see appendix.tex: .1 project, .2 src, .3 main.py).
    out_lines = [r"\dirtree{"]
    last_dt = 0
    for depth, name in parsed:
        if not _should_include(name):
            continue
        if depth == 0:
            if MAX_NESTING_LEVEL is not None and 1 > MAX_NESTING_LEVEL:
                continue
            out_lines.append(f".1 {_tex_cell(name)}.")
            last_dt = 1
            continue
        dt = depth + 1
        if MAX_NESTING_LEVEL is not None and dt > MAX_NESTING_LEVEL:
            continue
        if dt > last_dt + 1:
            raise ValueError(
                f"dirtree depth jump {last_dt} -> {dt} for {name!r} "
                "(tree prefix may use an unexpected width; check input)"
            )
        out_lines.append(f".{dt} {_tex_cell(name)}.")
        last_dt = dt
    out_lines.append("}")
    return "\n".join(out_lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=here / "tree_in_text.txt",
        help="tree output file (default: tree_in_text.txt next to this script)",
    )
    args = p.parse_args()
    text = args.input.read_text(encoding="utf-8")
    try:
        latex = tree_lines_to_dirtree(text.splitlines(keepends=False))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(latex)


if __name__ == "__main__":
    main()
