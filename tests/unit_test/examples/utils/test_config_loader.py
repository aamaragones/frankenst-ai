from pathlib import Path

import pytest

from config.settings import get_settings
from utils.config_loader import load_node_registry

pytestmark = pytest.mark.unit

NODE = """
nodes:
  - id: N
    name: n
    type: {type}
    metadata:
      {metadata}
"""


def _load(
    tmp_path: Path, *, type: str = "tools", metadata: str = "description: d"
) -> dict[str, dict[str, object]]:
    path = tmp_path / "nodes.yaml"
    path.write_text(NODE.format(type=type, metadata=metadata))
    return load_node_registry(path)


def test_shipped_registry_keeps_sensitive_and_simple_tools_consistent() -> None:
    registry = load_node_registry(get_settings().config_nodes_file_path)
    oak = set(registry["OAKTOOLS_NODE"]["metadata"]["tools"])
    simple = set(registry["SIMPLE_OAKTOOLS_NODE"]["metadata"]["tools"])
    sensitive = set(registry["HUMAN_REVIEW_NODE"]["metadata"]["sensitive_tools"])

    assert sensitive <= oak
    assert simple <= oak
    assert not sensitive & simple


def test_tool_name_lists_are_returned_as_declared(tmp_path: Path) -> None:
    registry = _load(tmp_path, metadata="tools: [a, b]")

    assert registry["N"]["metadata"] == {"tools": ["a", "b"]}


@pytest.mark.parametrize("metadata", ["tools: a", "tools: []", "sensitive_tools: [1]"])
def test_tool_name_lists_must_be_non_empty_lists_of_names(
    tmp_path: Path, metadata: str
) -> None:
    with pytest.raises(ValueError, match="non-empty list of tool names"):
        _load(tmp_path, metadata=metadata)
