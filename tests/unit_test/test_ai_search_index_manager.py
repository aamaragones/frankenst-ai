from typing import Any, cast

import pytest
from azure.search.documents.indexes import SearchIndexClient

from core_examples.utils.rag.ai_search_schemas import (
    get_ai_search_schema_name,
    list_ai_search_index_names,
    list_ai_search_schema_names,
    load_ai_search_index_definition,
    load_registered_ai_search_index_definition,
)
from core_examples.utils.rag.ai_search_unstructured_indexer import AISearchIndexManager

pytestmark = pytest.mark.unit


class FakeMissingIndexClient:
    def __init__(self) -> None:
        self.created_index: Any = None

    def get_index(self, name: str) -> Any:
        raise RuntimeError(f"Index {name} not found")

    def create_index(self, index: Any) -> None:
        self.created_index = index


def test_loader_lists_multiple_shared_index_schemas() -> None:
    schemas = list_ai_search_schema_names()

    assert "default-multivector" in schemas
    assert "pokeseriex-index" in schemas


def test_loader_lists_registered_index_names() -> None:
    index_names = list_ai_search_index_names()

    assert "demo-rag-multimodal-index" in index_names
    assert "pokeseriex-index" in index_names


def test_loader_resolves_registered_index_schema_name() -> None:
    assert get_ai_search_schema_name("pokeseriex-index") == "pokeseriex-index"


def test_loader_binds_runtime_index_name_to_shared_schema() -> None:
    index_definition = load_ai_search_index_definition(
        schema_name=get_ai_search_schema_name("pokeseriex-index"),
        index_name="runtime-index",
    )

    assert index_definition.name == "runtime-index"
    assert index_definition.fields[0].name == "id"


def test_loader_binds_runtime_index_name_from_registered_index_name() -> None:
    index_definition = load_registered_ai_search_index_definition(
        index_name="pokeseriex-index",
        runtime_index_name="runtime-index",
    )

    assert index_definition.name == "runtime-index"
    assert index_definition.fields[0].name == "id"


def test_create_index_loads_search_schema_from_shared_loader() -> None:
    index_client = FakeMissingIndexClient()
    manager = AISearchIndexManager(
        index_client=cast(SearchIndexClient, index_client), index_name="runtime-index"
    )

    manager.create_index("pokeseriex-index")

    assert index_client.created_index is not None
    assert index_client.created_index.name == "runtime-index"
    assert index_client.created_index.fields[0].name == "id"