"""The dependency rule, as an AST scan: a violation fails here, not in production."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[4] / "src"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _top(names: set[str]) -> set[str]:
    return {name.split(".")[0] for name in names}


def _files(*parts: str) -> list[Path]:
    return sorted(SRC.joinpath(*parts).rglob("*.py"))


def test_frankstate_imports_nothing_from_the_repo() -> None:
    for path in _files("frankstate"):
        assert not _top(_imports(path)) & {
            "config",
            "utils",
            "core_ai_examples",
            "services",
        }, path


def test_core_ai_examples_never_import_services() -> None:
    for path in _files("core_ai_examples"):
        assert "services" not in _top(_imports(path)), path


def test_utils_never_import_services_or_cores() -> None:
    for path in _files("utils"):
        assert not _top(_imports(path)) & {"services", "core_ai_examples"}, path


def test_secrets_backend_imports_nothing_from_the_repo() -> None:
    names = _top(_imports(SRC / "utils" / "secrets.py"))
    assert not names & {"config", "utils", "services", "core_ai_examples"}


def test_settings_imports_only_the_secrets_backend_from_the_repo() -> None:
    repo_imports = {
        n
        for n in _imports(SRC / "config" / "settings.py")
        if _top({n}) & {"config", "utils", "services", "core_ai_examples"}
    }
    assert repo_imports == {"utils.secrets"}


def test_only_settings_imports_the_secrets_backend() -> None:
    importers = [
        p
        for p in SRC.rglob("*.py")
        if "utils.secrets" in _imports(p) and p.name != "secrets.py"
    ]
    assert importers == [SRC / "config" / "settings.py"]


def test_only_graph_layouts_import_services_from_config() -> None:
    for path in _files("config"):
        if "services" in _top(_imports(path)):
            assert path.parent == SRC / "config" / "graph_layout", path
