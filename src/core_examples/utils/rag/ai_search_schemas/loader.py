import json
from pathlib import Path

from azure.search.documents.indexes.models import SearchIndex

SCHEMAS_DIRECTORY = Path(__file__).resolve().parent

AI_SEARCH_INDEX_SCHEMA_MAP: dict[str, str] = {
    "pokeseriex-index": "pokeseriex-index",
    "demo-rag-multimodal-index": "default-multivector",
}


def list_ai_search_schema_names() -> list[str]:
    """Return the available JSON schema names under the shared schema directory."""

    return sorted(schema_path.stem for schema_path in SCHEMAS_DIRECTORY.glob("*.json"))


def list_ai_search_index_names() -> list[str]:
    """Return the registered Azure AI Search index names."""

    return sorted(AI_SEARCH_INDEX_SCHEMA_MAP)


def get_ai_search_schema_path(schema_name: str) -> Path:
    """Resolve a named Azure AI Search schema file from the shared schema directory."""

    schema_path = SCHEMAS_DIRECTORY / f"{schema_name}.json"
    if schema_path.exists():
        return schema_path

    available_schemas = ", ".join(list_ai_search_schema_names()) or "none"
    raise FileNotFoundError(
        f"Azure AI Search schema '{schema_name}' was not found in '{SCHEMAS_DIRECTORY}'. "
        f"Available schemas: {available_schemas}."
    )


def get_ai_search_schema_name(index_name: str) -> str:
    """Return the schema name associated with a registered Azure AI Search index name."""

    if index_name in AI_SEARCH_INDEX_SCHEMA_MAP:
        return AI_SEARCH_INDEX_SCHEMA_MAP[index_name]

    available_index_names = ", ".join(list_ai_search_index_names()) or "none"
    raise KeyError(
        f"Azure AI Search index name '{index_name}' is not registered. "
        f"Available index names: {available_index_names}."
    )


def load_ai_search_index_definition(schema_name: str, index_name: str) -> SearchIndex:
    """Load a SearchIndex definition from JSON and override its runtime index name."""

    schema_path = get_ai_search_schema_path(schema_name)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    if not isinstance(schema, dict):
        raise ValueError(f"The index schema '{schema_path}' must contain a JSON object at the root.")

    schema["name"] = index_name
    index_definition = SearchIndex.from_dict(schema)
    if index_definition is None:
        raise ValueError(f"The index schema '{schema_path}' could not be deserialized into a SearchIndex.")

    return index_definition


def load_registered_ai_search_index_definition(
    index_name: str,
    runtime_index_name: str | None = None,
) -> SearchIndex:
    """Load a registered SearchIndex definition from an index name.

    `runtime_index_name` allows binding the schema to a different output name
    when needed, while keeping the registry itself keyed only by index name.
    """

    schema_name = get_ai_search_schema_name(index_name)
    return load_ai_search_index_definition(
        schema_name=schema_name,
        index_name=runtime_index_name or index_name,
    )