"""Fixtures for the examples tree: clean env, cleared process caches, published fakes.

Scoped here so the published slice under `unit_test/frankstate` needs none of this.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import pytest

from config.settings import get_settings
from services.llm.llm_services import LLMRuntime, LLMServices

_MANAGED_ENV = (
    "FRANK_CORE_PACKAGE_PATH",
    "LOG_LEVEL",
    "FRANK_LOG_LEVEL",
    "LOG_TO_FILE",
    "FRANK_LOG_TO_FILE",
    "AZURE_KEY_VAULT_NAME",
    "AZURE_BLOB_STORAGE_NAME",
    "AZURE_SEARCH_SERVICE_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "APPLICATION_INSIGHTS_CONNECTION_STRING",
    "AZURE_FOUNDRY_PROJECT_ENDPOINT",
    "AZURE_INFERENCE_MODEL_NAME",
    "AZURE_EMBEDDINGS_ENDPOINT",
    "AZURE_EMBEDDINGS_MODEL_NAME",
    "DATABRICKS_LLM_ENDPOINT",
    "DATABRICKS_EMBEDDINGS_ENDPOINT",
    "OLLAMA_WINDOWS_BASE_URL",
)


def _reset_process_caches() -> None:
    LLMServices.reset()
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    for name in _MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    _reset_process_caches()
    yield
    _reset_process_caches()


@pytest.fixture
def raises_on_call() -> Callable[[BaseException], Callable[..., NoReturn]]:
    def _factory(exc: BaseException) -> Callable[..., NoReturn]:
        def _raise(*_args: Any, **_kwargs: Any) -> NoReturn:
            raise exc

        return _raise

    return _factory


@pytest.fixture
def published_runtime(monkeypatch: pytest.MonkeyPatch) -> Callable[..., LLMRuntime]:
    """Publish fakes under the given kind names as if `launch()` had built them."""

    def _publish(**clients: Any) -> LLMRuntime:
        runtime = LLMRuntime(MappingProxyType(dict(clients)))
        monkeypatch.setattr(LLMServices, "_runtime", runtime)
        return runtime

    return _publish
