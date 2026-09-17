from typing import Any

import pytest

import config.settings as settings_module
from config.settings import get_settings
from services.llm import llm_services as llms_module
from services.llm.llm_services import LLMServices
from utils.config_loader import read_yaml

pytestmark = pytest.mark.unit

PROVIDERS = ("ollama", "azure_ai", "databricks")


class _Factory:
    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


def _stub_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    classes = {"chat": _Factory(), "embeddings": _Factory()}
    for name in ("_ollama_classes", "_azure_ai_classes", "_databricks_classes"):
        monkeypatch.setattr(LLMServices, name, lambda classes=classes: classes)
    monkeypatch.setattr(
        llms_module,
        "resolve_ollama_base_url",
        lambda config_host=None: "http://ollama.local",
    )
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda name, *, required=True, key_vault_name=None: f"<{name}>",
    )


def test_shipped_config_llms_declares_a_section_for_every_launch_key() -> None:
    config = read_yaml(get_settings().config_llms_file_path)

    assert config["launch"]
    for kind, provider in config["launch"].items():
        assert isinstance(config[provider][kind], dict), f"{provider}.{kind} missing"


def test_shipped_config_llms_builds_as_shipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_providers(monkeypatch)
    config = read_yaml(get_settings().config_llms_file_path)

    runtime = LLMServices.build_runtime(config)

    assert set(runtime.kinds) == set(config["launch"])


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_declared_section_of_each_provider_builds(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    _stub_providers(monkeypatch)
    config = read_yaml(get_settings().config_llms_file_path)
    declared = config.get(provider) or {}
    config["launch"] = {kind: provider for kind in declared}

    runtime = LLMServices.build_runtime(config)

    assert set(runtime.kinds) == set(declared)
