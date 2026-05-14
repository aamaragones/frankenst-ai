from importlib import import_module, resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from core_examples.utils.config_loader import _read_text_resource


def resolve_package_resource(package: str, *relative_parts: str) -> Traversable:
    """Resolve a package resource without relying on filesystem-relative module paths."""

    resource = resources.files(package)
    for part in relative_parts:
        resource = resource.joinpath(part)
    return resource


def _get_core_constants_module():
    try:
        return import_module("core_examples.constants")
    except ImportError:
        return None


def get_project_root_path() -> Path:
    """Return the configured project root or fall back to the repository root."""

    fallback = Path(__file__).resolve().parents[3]
    constants_module = _get_core_constants_module()
    configured_path = getattr(constants_module, "PROJECT_ROOT_PATH", None) if constants_module else None

    return Path(configured_path).expanduser().resolve() if configured_path else fallback


def get_default_artifacts_directory() -> Path:
    """Return the default artifacts directory, using constants only when available."""

    project_root_path = get_project_root_path()
    constants_module = _get_core_constants_module()
    configured_path = getattr(constants_module, "ARTIFACTS_DIRECTORY_PATH", None) if constants_module else None

    return Path(configured_path).expanduser().resolve() if configured_path else project_root_path / "artifacts"


def get_default_logs_directory() -> Path:
    """Return the default logs directory, using constants only when available."""

    project_root_path = get_project_root_path()
    constants_module = _get_core_constants_module()
    configured_path = getattr(constants_module, "LOGS_DIRECTORY_PATH", None) if constants_module else None

    return Path(configured_path).expanduser().resolve() if configured_path else project_root_path / "logs"


def resolve_configured_path(path_value: str | Path, base_dir: str | Path) -> Path:
    """Resolve an absolute or base-dir-relative path from configuration."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    return Path(base_dir).expanduser().resolve() / path


def load_and_clean_text_file(file_path: str | Path | Traversable, remove_empty_lines: bool = False) -> str:
    try:
        content = _read_text_resource(file_path)
        if remove_empty_lines:
            content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
        else:
            content = content.strip()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Not found the file <{file_path}>.")
    
def save_text_to_artifact(
    content: str,
    filename: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> Path:
    project_root_path = get_project_root_path()
    target_dir = resolve_configured_path(
        artifacts_dir if artifacts_dir is not None else get_default_artifacts_directory(),
        project_root_path,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"artifact_{project_root_path.name}.txt"
    elif not filename.endswith('.txt'):
        filename += '.txt'

    file_path = target_dir / filename

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return file_path

async def print_process_astream(
    graph: CompiledStateGraph,
    message_input: dict[str, Any] | Command[Any] | None,
    runnable_config: RunnableConfig | None = None,
):
    """Print `astream()` updates for any input accepted by the compiled graph.

    In Frankenst-AI notebooks this is typically a state dictionary keyed by
    graph field names, but LangGraph
    also allows resuming execution with a `Command` or passing `None`.
    """
    events = []
    async for event in graph.astream(message_input, runnable_config, stream_mode="updates"):
        events.append(event)
        print(event)
        print("\n")

    if not events:
        raise ValueError("Graph stream produced no events.")

    return events[-1]