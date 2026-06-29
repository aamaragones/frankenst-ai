"""Freeze the published wheel surface.

The base wheel must contain only the `frankstate` package (PEP 561 marker
included) and must never leak the mono-repo's reference packages.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _top_level_packages(names: list[str]) -> set[str]:
    return {
        name.split("/", 1)[0]
        for name in names
        if "/" in name and not name.split("/", 1)[0].endswith(".dist-info")
    }


@pytest.mark.integration
def test_wheel_contains_only_frankstate(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is required to build the wheel")

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    wheels = list(tmp_path.glob("frankstate-*.whl"))
    assert len(wheels) == 1, f"expected a single wheel, found {wheels}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()

    assert _top_level_packages(names) == {"frankstate"}
    assert "frankstate/py.typed" in names
    assert not any(name.startswith("core_examples/") for name in names)
    assert not any(name.startswith("services/") for name in names)
