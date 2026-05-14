import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from core_examples.constants import CONFIG_FILE_PATH
from core_examples.utils.config_loader import read_yaml
from core_examples.utils.key_vault import get_secret
from core_examples.utils.ollama.ollama_wsl_proxy import resolve_ollama_base_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMRuntime:
	"""Resolved runtime objects exposed to the rest of the application."""

	model: BaseChatModel
	embeddings: Embeddings
	turbo_model: BaseChatModel | None = None


class LLMServices:
	"""Centralized runtime builder for chat models and embeddings providers.

	The class keeps a small provider registry with direct callables while
	preserving provider-specific preparation logic in dedicated helpers.
	Consumers should continue to call `launch()` and then read
	`LLMServices.model` and `LLMServices.embeddings`.
	"""

	model: BaseChatModel | None = None
	embeddings: Embeddings | None = None
	turbo_model: BaseChatModel | None = None
	_launch_lock = Lock()

	@classmethod
	def _model_providers(cls) -> dict[str, Callable[[dict[str, Any]], BaseChatModel]]:
		"""Return the provider registry used by the model dispatcher."""

		return {
			"ollama": cls._load_ollama_model,
			"azure_ai": cls._load_azure_ai_model,
		}

	@classmethod
	def _embeddings_providers(cls) -> dict[str, Callable[[dict[str, Any]], Embeddings]]:
		"""Return the provider registry used by the embeddings dispatcher."""

		return {
			"ollama": cls._load_ollama_embeddings,
			"azure_ai": cls._load_azure_ai_embeddings,
		}

	@classmethod
	def _load_config(cls, config: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Load the central config and validate the launch selector section."""

		logger.info(
			"Loading LLM runtime configuration from %s.",
			"provided config" if config is not None else CONFIG_FILE_PATH,
		)
		resolved_config = config if config is not None else read_yaml(CONFIG_FILE_PATH)
		launch_config = resolved_config.get("launch")
		if not isinstance(launch_config, dict):
			raise RuntimeError("Missing config section for: launch")

		for key in ("model", "embeddings"):
			if key not in launch_config:
				raise RuntimeError(f"Missing config entry for: launch.{key}")

		return resolved_config

	@classmethod
	def _require(cls, config: dict[str, Any], path: str, *, as_section: bool = False) -> Any:
		"""Read a dotted config path and optionally require that it resolves to a section.

		When `as_section` is true, missing keys and non-mapping results are both
		reported as missing config sections for the requested path.
		"""

		value: Any = config
		for key in path.split("."):
			if not isinstance(value, dict) or key not in value:
				message_kind = "section" if as_section else "entry"
				raise RuntimeError(f"Missing config {message_kind} for: {path}")
			value = value[key]

		if as_section and not isinstance(value, dict):
			raise RuntimeError(f"Missing config section for: {path}")

		return value

	@classmethod
	def _resolve_config_value(cls, value: Any) -> Any:
		"""Resolve config literals and `{secret: ...}` references recursively."""

		if isinstance(value, dict):
			if set(value) == {"secret"}:
				secret_name = value["secret"]
				if not isinstance(secret_name, str) or not secret_name:
					raise RuntimeError("Config secret references must be non-empty strings.")
				return get_secret(secret_name)

			return {key: cls._resolve_config_value(item) for key, item in value.items()}

		if isinstance(value, list):
			return [cls._resolve_config_value(item) for item in value]

		return value

	@classmethod
	def _resolve_runtime_kwargs(cls, runtime_config: dict[str, Any]) -> dict[str, Any]:
		"""Resolve a runtime subsection into constructor kwargs."""

		resolved = cls._resolve_config_value(runtime_config)
		if not isinstance(resolved, dict):
			raise RuntimeError("Runtime configuration must resolve to a mapping.")
		return {key: value for key, value in resolved.items() if value is not None}

	@classmethod
	def _prepare_ollama_kwargs(cls, runtime_config: dict[str, Any], config_path: str) -> dict[str, Any]:
		"""Prepare Ollama kwargs and inject the resolved base URL when missing."""

		kwargs = cls._resolve_runtime_kwargs(runtime_config)
		host = kwargs.pop("host", None)
		if "base_url" not in kwargs:
			kwargs["base_url"] = resolve_ollama_base_url(config_host=host)

		if not kwargs.get("model"):
			raise RuntimeError(f"Missing config entry for: {config_path}.model")

		return kwargs

	@classmethod
	def _prepare_azure_ai_kwargs(cls, runtime_config: dict[str, Any], config_path: str) -> dict[str, Any]:
		"""Resolve and apply the Azure AI config validation owned by this project.

		This method intentionally validates only the local config contract and
		leaves deeper client validation to `langchain_azure_ai`.

		Azure AI Foundry's OpenAI-compatible wrapper enables Responses API by
		default, but not every Azure region supports it yet. Default to classic
		chat completions unless the repo config explicitly opts back in.
		"""

		kwargs = cls._resolve_runtime_kwargs(runtime_config)
		if kwargs.get("endpoint") and kwargs.get("project_endpoint"):
			raise RuntimeError(f"Config section {config_path} cannot define both endpoint and project_endpoint.")

		if not kwargs.get("endpoint") and not kwargs.get("project_endpoint"):
			raise RuntimeError(f"Missing config entry for: {config_path}.endpoint or {config_path}.project_endpoint")

		if not kwargs.get("model"):
			raise RuntimeError(f"Missing config entry for: {config_path}.model")

		# NOTE: use_responses_api false for Azure AI chat models since not all regions support it yet
		if config_path.endswith(".model"):
			kwargs.setdefault("use_responses_api", False)

		if not kwargs.get("credential"):
			kwargs["credential"] = DefaultAzureCredential()

		credential = kwargs.get("credential")
		credential_type = type(credential).__name__ if credential is not None and not isinstance(credential, str) else credential if isinstance(credential, str) else None
		logger.info(
			"Preparing Azure AI runtime for %s: model=%s project_endpoint=%s endpoint=%s api_version=%s use_responses_api=%s credential_type=%s",
			config_path,
			kwargs.get("model"),
			kwargs.get("project_endpoint"),
			kwargs.get("endpoint"),
			kwargs.get("api_version"),
			kwargs.get("use_responses_api"),
			credential_type,
		)

		return kwargs

	@classmethod
	def _load_ollama_model(cls, config: dict[str, Any]) -> BaseChatModel:
		runtime_config = cls._require(config, "ollama.model", as_section=True)
		kwargs = cls._prepare_ollama_kwargs(runtime_config, "ollama.model")
		logger.info("Creating Ollama chat runtime for model '%s'.", kwargs.get("model"))
		model = ChatOllama(**kwargs)
		logger.info("Loaded Ollama chat runtime '%s'.", type(model).__name__)
		return model

	@classmethod
	def _load_ollama_embeddings(cls, config: dict[str, Any]) -> Embeddings:
		runtime_config = cls._require(config, "ollama.embeddings", as_section=True)
		kwargs = cls._prepare_ollama_kwargs(runtime_config, "ollama.embeddings")
		logger.info("Creating Ollama embeddings runtime for model '%s'.", kwargs.get("model"))
		embeddings = OllamaEmbeddings(**kwargs)
		logger.info("Loaded Ollama embeddings runtime '%s'.", type(embeddings).__name__)
		return embeddings

	@classmethod
	def _load_azure_ai_model(cls, config: dict[str, Any]) -> BaseChatModel:
		runtime_config = cls._require(config, "azure_ai.model", as_section=True)
		kwargs = cls._prepare_azure_ai_kwargs(runtime_config, "azure_ai.model")
		model = AzureAIOpenAIApiChatModel(**kwargs)
		logger.info(
			"Loaded Azure AI runtime for %s: runtime_class=%s model=%s client_type=%s async_client_type=%s",
			"azure_ai.model",
			type(model).__name__,
			getattr(model, "model", kwargs.get("model")),
			type(getattr(model, "client", None)).__name__ if getattr(model, "client", None) is not None else None,
			type(getattr(model, "async_client", None)).__name__ if getattr(model, "async_client", None) is not None else None,
		)
		return model

	@classmethod
	def _load_azure_ai_embeddings(cls, config: dict[str, Any]) -> Embeddings:
		runtime_config = cls._require(config, "azure_ai.embeddings", as_section=True)
		kwargs = cls._prepare_azure_ai_kwargs(runtime_config, "azure_ai.embeddings")
		embeddings = AzureAIOpenAIApiEmbeddingsModel(**kwargs)
		logger.info(
			"Loaded Azure AI runtime for %s: runtime_class=%s model=%s client_type=%s async_client_type=%s has_embed_query=%s",
			"azure_ai.embeddings",
			type(embeddings).__name__,
			getattr(embeddings, "model", kwargs.get("model")),
			type(getattr(embeddings, "client", None)).__name__ if getattr(embeddings, "client", None) is not None else None,
			type(getattr(embeddings, "async_client", None)).__name__ if getattr(embeddings, "async_client", None) is not None else None,
			hasattr(embeddings, "embed_query"),
		)
		return embeddings

	@classmethod
	def _load_model(cls, config: dict[str, Any], provider_name: str) -> BaseChatModel:
		logger.info("Resolving chat model provider '%s'.", provider_name)
		loader = cls._model_providers().get(provider_name)
		if loader is None:
			raise ValueError(f"Unsupported provider type: {provider_name}")
		model = loader(config)
		logger.info("Chat model provider '%s' loaded runtime '%s'.", provider_name, type(model).__name__)
		return model

	@classmethod
	def _load_embeddings(cls, config: dict[str, Any], provider_name: str) -> Embeddings:
		logger.info("Resolving embeddings provider '%s'.", provider_name)
		loader = cls._embeddings_providers().get(provider_name)
		if loader is None:
			raise ValueError(f"Unsupported provider type: {provider_name}")
		embeddings = loader(config)
		logger.info("Embeddings provider '%s' loaded runtime '%s'.", provider_name, type(embeddings).__name__)
		return embeddings

	@classmethod
	def _current_runtime(cls) -> LLMRuntime | None:
		"""Return the current shared runtime from class-level state if initialized."""

		if cls.model is None or cls.embeddings is None:
			return None

		return LLMRuntime(cls.model, cls.embeddings, cls.turbo_model)

	@classmethod
	def build_runtime(cls, config: dict[str, Any] | None = None) -> LLMRuntime:
		"""Build a fresh runtime from config without mutating class attributes."""

		resolved_config = cls._load_config(config)
		model_provider = cls._require(resolved_config, "launch.model")
		embeddings_provider = cls._require(resolved_config, "launch.embeddings")
		logger.info(
			"Building LLM runtime with providers model=%s embeddings=%s.",
			model_provider,
			embeddings_provider,
		)
		model = cls._load_model(resolved_config, model_provider)
		embeddings = cls._load_embeddings(resolved_config, embeddings_provider)
		logger.info(
			"Built LLM runtime successfully: model_class=%s embeddings_class=%s turbo_model=%s.",
			type(model).__name__,
			type(embeddings).__name__,
			None,
		)
		return LLMRuntime(model, embeddings, None)

	@classmethod
	def launch(cls, config: dict[str, Any] | None = None, *, force_reload: bool = False) -> LLMRuntime:
		"""Publish one shared runtime per process unless an explicit reload is requested.

		When the shared runtime is already available, repeated calls reuse it.
		Pass `force_reload=True` to rebuild the published runtime.
		"""

		logger.info(
			"LLMServices.launch requested: force_reload=%s has_cached_model=%s has_cached_embeddings=%s.",
			force_reload,
			cls.model is not None,
			cls.embeddings is not None,
		)
		current_runtime = None if force_reload else cls._current_runtime()
		if current_runtime is not None:
			logger.info(
				"LLMServices.launch reusing cached runtime: model_class=%s embeddings_class=%s.",
				type(current_runtime.model).__name__,
				type(current_runtime.embeddings).__name__,
			)
			return current_runtime

		logger.info("LLMServices.launch acquiring runtime initialization lock.")
		with cls._launch_lock:
			current_runtime = None if force_reload else cls._current_runtime()
			if current_runtime is not None:
				logger.info(
					"LLMServices.launch found cached runtime after lock acquisition: model_class=%s embeddings_class=%s.",
					type(current_runtime.model).__name__,
					type(current_runtime.embeddings).__name__,
				)
				return current_runtime

			logger.info("LLMServices.launch initializing a new shared runtime.")
			try:
				runtime = cls.build_runtime(config)
			except Exception:
				logger.exception("LLMServices.launch failed while building the shared runtime.")
				raise
			cls.model = runtime.model
			cls.embeddings = runtime.embeddings
			cls.turbo_model = runtime.turbo_model
			logger.info(
				"LLMServices.launch published shared runtime: model_class=%s embeddings_class=%s turbo_model_class=%s.",
				type(runtime.model).__name__,
				type(runtime.embeddings).__name__,
				type(runtime.turbo_model).__name__ if runtime.turbo_model is not None else None,
			)
			return runtime