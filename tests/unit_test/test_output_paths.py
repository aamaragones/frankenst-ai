import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from langgraph.types import Command

import core_examples.utils.common as common_module
import core_examples.utils.config_loader as config_loader_module
import core_examples.utils.logger as logger_module
import core_examples.utils.rag.local_chroma as local_chroma_module


@dataclass
class _FakeSettings:
    """Minimal settings stub for logging tests — avoids pydantic-settings env resolution."""
    log_level: str | None = None
    log_to_file: bool = False
    default_log_file_path: Path | None = None
    config_logging_file_path: Path | None = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "src" / "core_examples" / "config" / "config_logging.yml"
    )


def test_resolve_configured_path_uses_base_dir_for_relative_paths(tmp_path: Path) -> None:
    base_dir = tmp_path / "workspace"
    base_dir.mkdir()

    assert common_module.resolve_configured_path("logs", base_dir) == base_dir / "logs"


def test_resolve_configured_path_preserves_absolute_paths(tmp_path: Path) -> None:
    base_dir = tmp_path / "workspace"
    base_dir.mkdir()
    absolute_path = tmp_path / "shared" / "logs"

    assert common_module.resolve_configured_path(absolute_path, base_dir) == absolute_path


def test_save_text_to_artifact_uses_default_directory_without_constants_or_cwd(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "template-root"
    other_dir = tmp_path / "elsewhere"
    project_root.mkdir()
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    monkeypatch.setattr(common_module, "get_project_root_path", lambda: project_root)
    monkeypatch.setattr(common_module, "get_default_artifacts_directory", lambda: project_root / "artifacts")

    artifact_path = common_module.save_text_to_artifact("hello")

    assert artifact_path == project_root / "artifacts" / "artifact_template-root.txt"
    assert artifact_path.read_text(encoding="utf-8") == "hello"


def test_configure_logging_does_not_create_a_log_file_by_default(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "logs" / "application.log"
    fake_settings = _FakeSettings(log_level="INFO", log_to_file=False, default_log_file_path=log_path)
    monkeypatch.setattr(logger_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(logger_module, "_LOGGING_CONFIGURED", False)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        h.close()
        root_logger.removeHandler(h)

    logger_module.configure_logging()
    logging.getLogger("frank-tests").info("console only")
    logging.shutdown()

    assert not log_path.exists()


def test_configure_logging_creates_a_log_file_when_enabled(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "logs" / "application.log"
    fake_settings = _FakeSettings(log_level="INFO", log_to_file=True, default_log_file_path=log_path)
    monkeypatch.setattr(logger_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(logger_module, "_LOGGING_CONFIGURED", False)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        h.close()
        root_logger.removeHandler(h)

    logger_module.configure_logging()
    logging.getLogger("frank-tests").info("configured log path")
    logging.shutdown()

    assert log_path.exists()
    assert "configured log path" in log_path.read_text(encoding="utf-8")


def test_default_output_directories_resolve_from_settings() -> None:
    project_root = common_module.get_project_root_path()

    assert common_module.get_default_logs_directory() == project_root / "logs"
    assert common_module.get_default_artifacts_directory() == project_root / "artifacts"


def test_local_chroma_uses_default_directories_without_constants_or_cwd(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "template-root"
    other_dir = tmp_path / "elsewhere"
    project_root.mkdir()
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    monkeypatch.setattr(local_chroma_module, "get_project_root_path", lambda: project_root)
    monkeypatch.setattr(
        local_chroma_module,
        "get_default_artifacts_directory",
        lambda: project_root / "artifacts",
    )

    assert local_chroma_module.get_default_local_chroma_directory() == project_root / "artifacts" / ".chromadb"
    assert local_chroma_module.get_default_local_docstore_directory() == project_root / "artifacts" / ".docstore"
    assert local_chroma_module.get_default_local_rag_docs_directory() == project_root / "artifacts" / "rag_docs"
    assert local_chroma_module._resolve_local_storage_path(None, local_chroma_module.get_default_local_chroma_directory) == (
        project_root / "artifacts" / ".chromadb"
    )


def test_local_chroma_resolves_relative_paths_against_project_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "template-root"
    project_root.mkdir()
    monkeypatch.setattr(local_chroma_module, "get_project_root_path", lambda: project_root)

    resolved = local_chroma_module._resolve_local_storage_path(
        Path("custom/chroma"),
        local_chroma_module.get_default_local_chroma_directory,
    )

    assert resolved == project_root / "custom" / "chroma"


def test_print_process_astream_accepts_command_input() -> None:
    class FakeCompiledGraph:
        def __init__(self) -> None:
            self.seen_input = None
            self.seen_config = None
            self.seen_stream_mode = None

        async def astream(self, message_input, runnable_config, stream_mode="updates"):
            self.seen_input = message_input
            self.seen_config = runnable_config
            self.seen_stream_mode = stream_mode
            yield {"messages": ["ok"]}

    graph = FakeCompiledGraph()
    command_input = Command(goto=())

    result = asyncio.run(common_module.print_process_astream(graph, command_input))

    assert graph.seen_input is command_input
    assert graph.seen_stream_mode == "updates"
    assert result == {"messages": ["ok"]}


def test_read_yaml_raises_for_empty_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "empty.yml"
    yaml_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        config_loader_module.read_yaml(yaml_path)


def test_read_yaml_raises_for_non_mapping_root(tmp_path: Path) -> None:
    yaml_path = tmp_path / "list-root.yml"
    yaml_path.write_text("- item\n- another\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping at the root"):
        config_loader_module.read_yaml(yaml_path)


def test_read_yaml_raises_for_empty_mapping(tmp_path: Path) -> None:
    yaml_path = tmp_path / "empty-mapping.yml"
    yaml_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be an empty mapping"):
        config_loader_module.read_yaml(yaml_path)


def test_load_node_registry_validates_required_node_fields(tmp_path: Path) -> None:
    yaml_path = tmp_path / "nodes.yml"
    yaml_path.write_text(
        "nodes:\n  - id: OAKLANG_NODE\n    type: enhancer\n    description: missing name\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required node fields"):
        config_loader_module.load_node_registry(yaml_path)