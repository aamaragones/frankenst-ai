from pathlib import Path

import pytest
from comment_ratio import Ratio, collect, main, measure

pytestmark = pytest.mark.unit


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_docstrings_do_not_count(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.py",
        '"""Module.\n\nMore text.\n"""\n\n\ndef f() -> None:\n    """Func."""\n',
    )

    assert measure(path) == Ratio(non_blank=5, commented=0)


def test_hash_inside_a_string_is_not_a_comment(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "s.py", 'x = "# not a comment"\ny = 1  # inline\n# full line\n'
    )

    assert measure(path) == Ratio(non_blank=3, commented=2)


def test_blank_lines_are_excluded_from_the_denominator(tmp_path: Path) -> None:
    path = _write(tmp_path, "b.py", "\n\nx = 1\n\n# c\n\n")

    assert measure(path) == Ratio(non_blank=2, commented=1)


def test_empty_file_has_ratio_zero(tmp_path: Path) -> None:
    assert measure(_write(tmp_path, "e.py", "")).value == 0.0


def test_exactly_the_threshold_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "edge.py", "# c\n" + "x = 1\n" * 19)

    assert main([str(tmp_path), "--max", "0.05"]) == 0
    assert "0 file(s) above" in capsys.readouterr().out


def test_offender_fails_and_is_listed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "bad.py", "# a\n# b\nx = 1\n")
    _write(tmp_path, "good.py", "x = 1\n")

    assert main([str(tmp_path), "--max", "0.15"]) == 1
    out = capsys.readouterr().out
    assert "bad.py  66.7%  (2/3)  FAIL" in out
    assert "good.py" not in out


def test_report_only_root_never_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reported = tmp_path / "reported"
    reported.mkdir()
    _write(reported, "bad.py", "# a\nx = 1\n")

    assert main([str(tmp_path), "--report-only", str(reported)]) == 0
    assert "REPORT" in capsys.readouterr().out


def test_all_prints_compliant_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "good.py", "x = 1\n")

    assert main([str(tmp_path), "--all"]) == 0
    assert "good.py  0.0%  (0/1)  OK" in capsys.readouterr().out


def test_collect_walks_directories_and_accepts_files(tmp_path: Path) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    a = _write(nested, "a.py", "")
    b = _write(tmp_path, "b.py", "")
    _write(tmp_path, "c.txt", "")

    assert list(collect([tmp_path])) == [b, a]
    assert list(collect([a, tmp_path / "c.txt"])) == [a]
