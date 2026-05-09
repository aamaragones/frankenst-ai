from azure.identity import DefaultAzureCredential
import pytest

from services.foundry import llms as llms_module


class CaptureFactory:
	def __init__(self) -> None:
		self.calls: list[dict] = []

	def __call__(self, **kwargs):
		self.calls.append(kwargs)
		return {"kwargs": kwargs}


def test_llmservices_build_runtime_uses_nested_ollama_sections(monkeypatch) -> None:
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

	assert runtime.model["kwargs"]["model"] == "gemma4:e4b"
	assert runtime.model["kwargs"]["temperature"] == 0
	assert runtime.model["kwargs"]["base_url"] == "http://ollama.local"
	assert runtime.embeddings["kwargs"]["model"] == "embeddinggemma"
	assert runtime.embeddings["kwargs"]["base_url"] == "http://ollama.local"


def test_llmservices_build_runtime_resolves_nested_azure_ai_sections(monkeypatch) -> None:
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

	assert runtime.model["kwargs"]["endpoint"] == "https://chat.example/openai/v1"
	assert runtime.model["kwargs"]["credential"] == "chat-key"
	assert runtime.model["kwargs"]["model"] == "gpt-4o-mini"
	assert runtime.embeddings["kwargs"]["model"] == "text-embedding-3-small"
	assert runtime.embeddings["kwargs"]["credential"] == "embed-key"


def test_llmservices_azure_ai_uses_default_credential_without_api_key(monkeypatch) -> None:
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

	assert isinstance(runtime.model["kwargs"]["credential"], DefaultAzureCredential)
	assert isinstance(runtime.embeddings["kwargs"]["credential"], DefaultAzureCredential)


def test_llmservices_azure_ai_requires_explicit_runtime_sections() -> None:
	config = {
		"launch": {"model": "azure_ai", "embeddings": "azure_ai"},
		"ollama": {},
		"azure_ai": {},
	}

	with pytest.raises(RuntimeError, match="Missing config section for: azure_ai.model"):
		llms_module.LLMServices.build_runtime(config)


def test_llmservices_reuses_provider_registry_for_model_and_embeddings(monkeypatch) -> None:
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

	assert runtime.model["kwargs"]["base_url"] == "http://ollama.local"
	assert runtime.embeddings["kwargs"]["base_url"] == "http://ollama.local"


def test_llmservices_rejects_unsupported_provider_from_central_registry() -> None:
	config = {
		"launch": {"model": "azureopenai", "embeddings": "azureopenai"},
		"ollama": {},
		"azure_ai": {},
	}

	with pytest.raises(ValueError, match="Unsupported provider type: azureopenai"):
		llms_module.LLMServices.build_runtime(config)