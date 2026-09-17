"""Fail when `#` comment lines exceed a share of a file's non-blank lines.

Docstrings do not count: the budget is for commentary, not for the API text that
tools render. Files under a `--report-only` root are printed but never fail.
"""

from __future__ import annotations

import argparse
import io
import sys
import tokenize
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple


class Ratio(NamedTuple):
    """Non-blank lines and how many of them carry a `#` comment."""

    non_blank: int
    commented: int

    @property
    def value(self) -> float:
        return self.commented / self.non_blank if self.non_blank else 0.0


def measure(path: Path) -> Ratio:
    source = path.read_text(encoding="utf-8")
    non_blank = sum(1 for line in source.splitlines() if line.strip())
    comment_lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comment_lines.add(token.start[0])
    return Ratio(non_blank, len(comment_lines))


def collect(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.py") if ".venv" not in p.parts)
        elif path.suffix == ".py":
            yield path


def _is_under(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    return any(
        resolved == r.resolve() or r.resolve() in resolved.parents for r in roots
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max", type=float, default=0.15, dest="maximum")
    parser.add_argument("--report-only", action="append", default=[], type=Path)
    parser.add_argument(
        "--all", action="store_true", help="print every file, not only offenders"
    )
    args = parser.parse_args(argv)

    failures = 0
    for path in collect(args.paths):
        ratio = measure(path)
        over = ratio.value > args.maximum
        reported_only = _is_under(path, args.report_only)
        if over or args.all:
            state = "REPORT" if reported_only else ("FAIL" if over else "OK")
            print(
                f"{path}  {ratio.value:.1%}  ({ratio.commented}/{ratio.non_blank})  {state}"
            )
        if over and not reported_only:
            failures += 1
    print(f"comment-ratio: {failures} file(s) above {args.maximum:.0%}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
