from typing import Any, cast

import pytest
from azure.identity import DefaultAzureCredential

from services.foundry import llms as llms_module

pytestmark = pytest.mark.unit


class CaptureFactory:
	def __init__(self) -> None:
		self.calls: list[dict[str, Any]] = []

	def __call__(self, **kwargs: Any) -> dict[str, Any]:
		self.calls.append(kwargs)
		return {"kwargs": kwargs}


def test_llmservices_build_runtime_uses_nested_ollama_sections(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	embeddings_factory = CaptureFactory()

	monkeypatch.setattr(llms_module, "ChatOllama", chat_factory)
	monkeypatch.setattr(llms_module, "OllamaEmbeddings", embeddings_factory)
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

	monkeypatch.setattr(llms_module, "AzureAIOpenAIApiChatModel", chat_factory)
	monkeypatch.setattr(llms_module, "AzureAIOpenAIApiEmbeddingsModel", embeddings_factory)
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

	monkeypatch.setattr(llms_module, "AzureAIOpenAIApiChatModel", chat_factory)
	monkeypatch.setattr(llms_module, "AzureAIOpenAIApiEmbeddingsModel", embeddings_factory)

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


def test_llmservices_azure_ai_requires_explicit_runtime_sections() -> None:
	config = {
		"launch": {"model": "azure_ai", "embeddings": "azure_ai"},
		"ollama": {},
		"azure_ai": {},
	}

	with pytest.raises(RuntimeError, match="Missing config section for: azure_ai.model"):
		llms_module.LLMServices.build_runtime(config)


def test_llmservices_reuses_provider_registry_for_model_and_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
	chat_factory = CaptureFactory()
	embeddings_factory = CaptureFactory()

	monkeypatch.setattr(llms_module, "ChatOllama", chat_factory)
	monkeypatch.setattr(llms_module, "OllamaEmbeddings", embeddings_factory)
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