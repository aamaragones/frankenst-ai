from typing import Any, cast

import pytest
from azure.identity import DefaultAzureCredential

from services.llm import llm_services as llms_module

pytestmark = pytest.mark.unit


class CaptureFactory:
	def __init__(self) -> None:
		self.calls: list[dict[str, Any]] = []

	def __call__(self, **kwargs: Any) -> dict[str, Any]:
		self.calls.append(kwargs)
		return {"kwargs": kwargs}


def _patch_provider_classes(
	monkeypatch: pytest.MonkeyPatch,
	importer_name: str,
	chat_factory: CaptureFactory | None = None,
	embeddings_factory: CaptureFactory | None = None,
	turbo_factory: CaptureFactory | None = None,
) -> None:
	"""Stub a provider's importer — the only seam that touches its package."""
	classes = {
		"model": chat_factory or CaptureFactory(),
		"embeddings": embeddings_factory or CaptureFactory(),
		"turbo_model": turbo_factory or CaptureFactory(),
	}
	monkeypatch.setattr(llms_module.LLMServices, importer_name, lambda: classes)


def test_llmservices_build_runtime_uses_nested_ollama_sections(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	embeddings_factory = CaptureFactory()

	_patch_provider_classes(monkeypatch, "_ollama_classes", chat_factory, embeddings_factory)
	monkeypatch.setattr(llms_module, "resolve_ollama_base_url", lambda config_host=None: "http://ollama.local")

	config = {
		"launch": {"model": "ollama", "embeddings": "ollama"},
		"ollama": {
			"model": {"model": "gemma4:e4b", "temperature": 0},
			"embeddings": {"model": "embeddinggemma"},
		},
		"azure_ai": {},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	model_kwargs = cast(dict[str, Any], runtime.model)["kwargs"]
	embeddings_kwargs = cast(dict[str, Any], runtime.embeddings)["kwargs"]
	assert model_kwargs["model"] == "gemma4:e4b"
	assert model_kwargs["temperature"] == 0
	assert model_kwargs["base_url"] == "http://ollama.local"
	assert embeddings_kwargs["model"] == "embeddinggemma"
	assert embeddings_kwargs["base_url"] == "http://ollama.local"


def test_llmservices_build_runtime_resolves_nested_azure_ai_sections(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	embeddings_factory = CaptureFactory()
	secret_values = {
		"CHAT_ENDPOINT": "https://chat.example/openai/v1",
		"CHAT_CREDENTIAL": "chat-key",
		"CHAT_MODEL": "gpt-4o-mini",
		"CHAT_API_VERSION": "2025-05-01-preview",
		"EMBED_ENDPOINT": "https://embed.example/openai/v1",
		"EMBED_CREDENTIAL": "embed-key",
		"EMBED_MODEL": "text-embedding-3-small",
		"EMBED_API_VERSION": "2025-05-01-preview",
	}

	_patch_provider_classes(monkeypatch, "_azure_ai_classes", chat_factory, embeddings_factory)
	monkeypatch.setattr(llms_module, "get_secret", lambda secret_name: secret_values[secret_name])

	config = {
		"launch": {"model": "azure_ai", "embeddings": "azure_ai"},
		"ollama": {},
		"azure_ai": {
			"model": {
				"endpoint": {"secret": "CHAT_ENDPOINT"},
				"credential": {"secret": "CHAT_CREDENTIAL"},
				"model": {"secret": "CHAT_MODEL"},
				"api_version": {"secret": "CHAT_API_VERSION"},
				"temperature": 0,
			},
			"embeddings": {
				"endpoint": {"secret": "EMBED_ENDPOINT"},
				"credential": {"secret": "EMBED_CREDENTIAL"},
				"model": {"secret": "EMBED_MODEL"},
				"api_version": {"secret": "EMBED_API_VERSION"},
			},
		},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	model_kwargs = cast(dict[str, Any], runtime.model)["kwargs"]
	embeddings_kwargs = cast(dict[str, Any], runtime.embeddings)["kwargs"]
	assert model_kwargs["endpoint"] == "https://chat.example/openai/v1"
	assert model_kwargs["credential"] == "chat-key"
	assert model_kwargs["model"] == "gpt-4o-mini"
	assert embeddings_kwargs["model"] == "text-embedding-3-small"
	assert embeddings_kwargs["credential"] == "embed-key"


def test_llmservices_azure_ai_uses_default_credential_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	embeddings_factory = CaptureFactory()

	_patch_provider_classes(monkeypatch, "_azure_ai_classes", chat_factory, embeddings_factory)

	config = {
		"launch": {"model": "azure_ai", "embeddings": "azure_ai"},
		"ollama": {},
		"azure_ai": {
			"model": {
				"endpoint": "https://chat.example/openai/v1",
				"model": "gpt-4o-mini",
				"api_version": "2025-05-01-preview",
			},
			"embeddings": {
				"endpoint": "https://embed.example/openai/v1",
				"model": "text-embedding-3-small",
				"api_version": "2025-05-01-preview",
			},
		},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	model_kwargs = cast(dict[str, Any], runtime.model)["kwargs"]
	embeddings_kwargs = cast(dict[str, Any], runtime.embeddings)["kwargs"]
	assert isinstance(model_kwargs["credential"], DefaultAzureCredential)
	assert isinstance(embeddings_kwargs["credential"], DefaultAzureCredential)


def test_llmservices_build_runtime_resolves_nested_databricks_sections(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	embeddings_factory = CaptureFactory()
	secret_values = {
		"DATABRICKS_LLM_ENDPOINT": "chat-serving-endpoint",
		"DATABRICKS_EMBEDDINGS_ENDPOINT": "embeddings-serving-endpoint",
	}

	_patch_provider_classes(monkeypatch, "_databricks_classes", chat_factory, embeddings_factory)
	monkeypatch.setattr(llms_module, "get_secret", lambda secret_name: secret_values[secret_name])

	config = {
		"launch": {"model": "databricks", "embeddings": "databricks"},
		"databricks": {
			"model": {
				"endpoint": {"secret": "DATABRICKS_LLM_ENDPOINT"},
				"use_ai_gateway": True,
				"max_tokens": 4096,
			},
			"embeddings": {
				"endpoint": {"secret": "DATABRICKS_EMBEDDINGS_ENDPOINT"},
			},
		},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	model_kwargs = cast(dict[str, Any], runtime.model)["kwargs"]
	embeddings_kwargs = cast(dict[str, Any], runtime.embeddings)["kwargs"]
	assert model_kwargs["endpoint"] == "chat-serving-endpoint"
	assert model_kwargs["use_ai_gateway"] is True
	assert model_kwargs["max_tokens"] == 4096
	assert embeddings_kwargs["endpoint"] == "embeddings-serving-endpoint"
	# Auth is delegated to the Databricks SDK default chain — never injected.
	assert "credential" not in model_kwargs
	assert "credential" not in embeddings_kwargs


def test_llmservices_databricks_requires_endpoint() -> None:
	config = {
		"launch": {"model": "databricks"},
		"databricks": {"model": {"max_tokens": 1024}},
	}

	with pytest.raises(RuntimeError, match="Config section databricks.model must define model"):
		llms_module.LLMServices.build_runtime(config)


def test_llmservices_databricks_rejects_endpoint_and_model_alias() -> None:
	config = {
		"launch": {"model": "databricks"},
		"databricks": {
			"model": {"endpoint": "chat-serving-endpoint", "model": "chat-serving-endpoint"},
		},
	}

	with pytest.raises(RuntimeError, match="cannot define both endpoint and model"):
		llms_module.LLMServices.build_runtime(config)


def test_llmservices_skips_runtime_kind_not_declared_in_launch(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	_patch_provider_classes(monkeypatch, "_databricks_classes", chat_factory)

	config = {
		"launch": {"model": "databricks"},
		"databricks": {"model": {"endpoint": "chat-serving-endpoint"}},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	assert cast(dict[str, Any], runtime.model)["kwargs"]["endpoint"] == "chat-serving-endpoint"
	assert runtime.embeddings is None
	assert runtime.turbo_model is None


def test_llmservices_skips_runtime_when_provider_section_is_commented_out(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	_patch_provider_classes(monkeypatch, "_databricks_classes", chat_factory)

	# launch selects embeddings, but the databricks.embeddings section is
	# absent (e.g. commented out in config_llms.yaml) — skip, do not fail.
	config = {
		"launch": {"model": "databricks", "embeddings": "databricks"},
		"databricks": {"model": {"endpoint": "chat-serving-endpoint"}},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	assert runtime.model is not None
	assert runtime.embeddings is None


def test_llmservices_builds_turbo_model_with_the_chat_class(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	turbo_factory = CaptureFactory()

	_patch_provider_classes(monkeypatch, "_ollama_classes", chat_factory, turbo_factory=turbo_factory)
	monkeypatch.setattr(llms_module, "resolve_ollama_base_url", lambda config_host=None: "http://ollama.local")

	config = {
		"launch": {"model": "ollama", "turbo_model": "ollama"},
		"ollama": {
			"model": {"model": "gemma4:e4b"},
			"turbo_model": {"model": "gemma4:e2b"},
		},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	assert cast(dict[str, Any], runtime.model)["kwargs"]["model"] == "gemma4:e4b"
	assert cast(dict[str, Any], runtime.turbo_model)["kwargs"]["model"] == "gemma4:e2b"
	assert runtime.embeddings is None


def test_llmservices_raises_when_no_runtime_is_declared() -> None:
	config = {
		"launch": {"model": "azure_ai", "embeddings": "azure_ai"},
		"ollama": {},
		"azure_ai": {},
	}

	with pytest.raises(RuntimeError, match="No runtime declared"):
		llms_module.LLMServices.build_runtime(config)


def test_llmservices_never_imports_undeclared_provider_packages(monkeypatch: pytest.MonkeyPatch) -> None:
	_patch_provider_classes(monkeypatch, "_ollama_classes")
	monkeypatch.setattr(llms_module, "resolve_ollama_base_url", lambda config_host=None: "http://ollama.local")

	def _fail_databricks_import() -> dict[str, Any]:
		raise AssertionError("databricks-langchain must not be imported for an ollama-only config")

	def _fail_azure_import() -> dict[str, Any]:
		raise AssertionError("langchain-azure-ai must not be imported for an ollama-only config")

	monkeypatch.setattr(llms_module.LLMServices, "_databricks_classes", _fail_databricks_import)
	monkeypatch.setattr(llms_module.LLMServices, "_azure_ai_classes", _fail_azure_import)

	config = {
		"launch": {"model": "ollama"},
		"ollama": {"model": {"model": "gemma4:e4b"}},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	assert runtime.model is not None


def test_llmservices_reuses_provider_registry_for_model_and_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	embeddings_factory = CaptureFactory()

	_patch_provider_classes(monkeypatch, "_ollama_classes", chat_factory, embeddings_factory)
	monkeypatch.setattr(llms_module, "resolve_ollama_base_url", lambda config_host=None: "http://ollama.local")

	config = {
		"launch": {"model": "ollama", "embeddings": "ollama"},
		"ollama": {
			"model": {"model": "gemma4:e4b"},
			"embeddings": {"model": "embeddinggemma"},
		},
		"azure_ai": {},
	}

	runtime = llms_module.LLMServices.build_runtime(config)

	model_kwargs = cast(dict[str, Any], runtime.model)["kwargs"]
	embeddings_kwargs = cast(dict[str, Any], runtime.embeddings)["kwargs"]
	assert model_kwargs["base_url"] == "http://ollama.local"
	assert embeddings_kwargs["base_url"] == "http://ollama.local"


def test_llmservices_rejects_unsupported_provider_from_central_registry() -> None:
	config = {
		"launch": {"model": "azureopenai", "embeddings": "azureopenai"},
		"ollama": {},
		"azure_ai": {},
	}

	with pytest.raises(ValueError, match="Unsupported provider type: azureopenai"):
		llms_module.LLMServices.build_runtime(config)
