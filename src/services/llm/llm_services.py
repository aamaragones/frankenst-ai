import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from core_examples.config.settings import get_settings
from core_examples.utils.config_loader import read_yaml
from core_examples.utils.key_vault import get_secret
from core_examples.utils.ollama.ollama_wsl_proxy import resolve_ollama_base_url

logger = logging.getLogger(__name__)


def _class_name(instance: Any) -> str | None:
	"""Return the class name of an optional runtime for logging."""
	return type(instance).__name__ if instance is not None else None


@dataclass(frozen=True)
class _ProviderSpec:
	"""How a provider imports its runtime classes and prepares their kwargs."""

	import_classes: Callable[[], dict[str, Any]]
	prepare_kwargs: Callable[[dict[str, Any], str], dict[str, Any]]


@dataclass(frozen=True)
class LLMRuntime:
	"""Resolved runtime objects exposed to the rest of the application.

	Each runtime kind is optional: it is None when config_llms.yaml does not
	declare it (see LLMServices._load_declared_runtime).
	"""

	model: BaseChatModel | None = None
	embeddings: Embeddings | None = None
	turbo_model: BaseChatModel | None = None


class LLMServices:
	"""Centralized runtime builder for chat models and embeddings providers.

	config_llms.yaml drives everything: a runtime kind (`model` / `embeddings` /
	`turbo_model`) is built only when `launch.<kind>` selects a provider and
	`<provider>.<kind>` declares its kwargs. What is not declared is skipped,
	provider import included, so any combination installs and runs.

	Consumers call `launch()` and read `LLMServices.model`,
	`LLMServices.embeddings` and `LLMServices.turbo_model`.
	"""

	RUNTIME_KINDS = ("model", "embeddings", "turbo_model")

	# `_runtime` is the published runtime; the model / embeddings / turbo_model
	# class attributes are the read-only mirror consumers are documented to read.
	model: BaseChatModel | None = None
	embeddings: Embeddings | None = None
	turbo_model: BaseChatModel | None = None
	_runtime: LLMRuntime | None = None
	_launch_lock = Lock()

	@staticmethod
	def _ollama_classes() -> dict[str, Any]:
		"""Import langchain-ollama only when an ollama runtime is declared."""
		from langchain_ollama import ChatOllama, OllamaEmbeddings

		return {"model": ChatOllama, "embeddings": OllamaEmbeddings, "turbo_model": ChatOllama}

	@staticmethod
	def _azure_ai_classes() -> dict[str, Any]:
		"""Import langchain-azure-ai only when an azure_ai runtime is declared."""
		from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
		from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel

		return {
			"model": AzureAIOpenAIApiChatModel,
			"embeddings": AzureAIOpenAIApiEmbeddingsModel,
			"turbo_model": AzureAIOpenAIApiChatModel,
		}

	@staticmethod
	def _databricks_classes() -> dict[str, Any]:
		"""Import databricks-langchain only when a databricks runtime is declared."""
		from databricks_langchain import ChatDatabricks, DatabricksEmbeddings

		return {"model": ChatDatabricks, "embeddings": DatabricksEmbeddings, "turbo_model": ChatDatabricks}

	@classmethod
	def _providers(cls) -> dict[str, _ProviderSpec]:
		"""Return the provider registry driving all runtime loading."""
		return {
			"ollama": _ProviderSpec(cls._ollama_classes, cls._prepare_ollama_kwargs),
			"azure_ai": _ProviderSpec(cls._azure_ai_classes, cls._prepare_azure_ai_kwargs),
			"databricks": _ProviderSpec(cls._databricks_classes, cls._prepare_databricks_kwargs),
		}

	@classmethod
	def _load_config(cls, config: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Load the central config and validate the launch selector section."""
		if config is not None:
			config_source: str | object = "provided config"
			resolved_config = config
		else:
			settings = get_settings()
			config_source = settings.config_llms_file_path
			resolved_config = read_yaml(settings.config_llms_file_path)

		logger.info(
			"Loading LLM runtime configuration from %s.",
			config_source,
		)
		launch_config = resolved_config.get("launch")
		if not isinstance(launch_config, dict):
			raise RuntimeError("Missing config section for: launch")

		return resolved_config

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

		Runtime compatibility is delegated
		to `langchain_azure_ai`, which is required because embeddings are known to
		work here with `services.ai/openai/v1` and no explicit API version.

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
		if not config_path.endswith(".embeddings"):
			kwargs.setdefault("use_responses_api", False)

		if not kwargs.get("credential"):
			from azure.identity import DefaultAzureCredential

			kwargs["credential"] = DefaultAzureCredential()

		credential = kwargs.get("credential")
		if isinstance(credential, str):
			credential_type = credential
		elif credential is not None:
			credential_type = type(credential).__name__
		else:
			credential_type = None
		logger.info(
			"Preparing Azure AI runtime for %s: model=%s project_endpoint=%s endpoint=%s api_version=%s "
			"use_responses_api=%s credential_type=%s",
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
	def _prepare_databricks_kwargs(cls, runtime_config: dict[str, Any], config_path: str) -> dict[str, Any]:
		"""Resolve and apply the Databricks config validation owned by this project.

		This method intentionally validates only the local config contract and
		leaves deeper client validation to `databricks_langchain`.

		`databricks_langchain` accepts `model` as an alias of `endpoint`; this
		repository standardizes on `endpoint` (the serving endpoint / AI Gateway
		model name), so defining both is rejected.

		No credential is injected: authentication is delegated to the Databricks
		SDK default chain (env DATABRICKS_HOST/DATABRICKS_TOKEN, a
		`~/.databrickscfg` profile, or the ambient in-workspace identity).
		"""
		kwargs = cls._resolve_runtime_kwargs(runtime_config)
		if kwargs.get("endpoint") and kwargs.get("model"):
			raise RuntimeError(f"Config section {config_path} cannot define both endpoint and model.")

		if not kwargs.get("endpoint"):
			raise RuntimeError(f"Missing config entry for: {config_path}.endpoint")

		logger.info(
			"Preparing Databricks runtime for %s: endpoint=%s temperature=%s max_tokens=%s",
			config_path,
			kwargs.get("endpoint"),
			kwargs.get("temperature"),
			kwargs.get("max_tokens"),
		)

		return kwargs

	@classmethod
	def _load_declared_runtime(cls, config: dict[str, Any], kind: str) -> Any | None:
		"""Load one runtime kind if config_llms declares it; skip it otherwise.

		Generic key-driven loading: a runtime kind is built only when BOTH keys
		exist — `launch.<kind>` selects the provider and `<provider>.<kind>` is a
		mapping with its constructor kwargs. When either key is absent (e.g.
		commented out) the runtime is skipped and the provider package is never
		imported, so installs may carry any combination of providers and kinds.
		"""
		provider_name = config["launch"].get(kind)
		if not provider_name:
			logger.info("launch.%s is not declared — skipping the %s runtime.", kind, kind)
			return None

		spec = cls._providers().get(provider_name)
		if spec is None:
			raise ValueError(f"Unsupported provider type: {provider_name}")

		config_path = f"{provider_name}.{kind}"
		provider_section = config.get(provider_name)
		runtime_config = provider_section.get(kind) if isinstance(provider_section, dict) else None
		if not isinstance(runtime_config, dict):
			logger.warning("Config section %s is not declared — skipping the %s runtime.", config_path, kind)
			return None

		kwargs = spec.prepare_kwargs(runtime_config, config_path)
		runtime_class = spec.import_classes()[kind]
		runtime = runtime_class(**kwargs)
		logger.info(
			"Loaded %s runtime from %s: runtime_class=%s.",
			kind,
			config_path,
			type(runtime).__name__,
		)
		return runtime

	@classmethod
	def _current_runtime(cls) -> LLMRuntime | None:
		"""Return the published shared runtime, if a launch already produced one."""
		return cls._runtime

	@classmethod
	def build_runtime(cls, config: dict[str, Any] | None = None) -> LLMRuntime:
		"""Build a fresh runtime from config without mutating class attributes."""
		resolved_config = cls._load_config(config)
		launch_config = resolved_config["launch"]
		logger.info(
			"Building LLM runtime with providers model=%s embeddings=%s turbo_model=%s.",
			launch_config.get("model"),
			launch_config.get("embeddings"),
			launch_config.get("turbo_model"),
		)
		runtimes = {kind: cls._load_declared_runtime(resolved_config, kind) for kind in cls.RUNTIME_KINDS}
		if all(runtime is None for runtime in runtimes.values()):
			raise RuntimeError(
				"No runtime declared: config_llms.yaml must declare launch.model, launch.embeddings "
				"and/or launch.turbo_model with a matching provider section."
			)

		logger.info(
			"Built LLM runtime successfully: model_class=%s embeddings_class=%s turbo_model_class=%s.",
			_class_name(runtimes["model"]),
			_class_name(runtimes["embeddings"]),
			_class_name(runtimes["turbo_model"]),
		)
		return LLMRuntime(runtimes["model"], runtimes["embeddings"], runtimes["turbo_model"])

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
				_class_name(current_runtime.model),
				_class_name(current_runtime.embeddings),
			)
			return current_runtime

		logger.info("LLMServices.launch acquiring runtime initialization lock.")
		with cls._launch_lock:
			current_runtime = None if force_reload else cls._current_runtime()
			if current_runtime is not None:
				logger.info(
					"LLMServices.launch found cached runtime after lock acquisition: "
					"model_class=%s embeddings_class=%s.",
					_class_name(current_runtime.model),
					_class_name(current_runtime.embeddings),
				)
				return current_runtime

			logger.info("LLMServices.launch initializing a new shared runtime.")
			try:
				runtime = cls.build_runtime(config)
			except Exception:
				logger.exception("LLMServices.launch failed while building the shared runtime.")
				raise
			cls._runtime = runtime
			cls.model = runtime.model
			cls.embeddings = runtime.embeddings
			cls.turbo_model = runtime.turbo_model
			logger.info(
				"LLMServices.launch published shared runtime: model_class=%s embeddings_class=%s turbo_model_class=%s.",
				_class_name(runtime.model),
				_class_name(runtime.embeddings),
				_class_name(runtime.turbo_model),
			)
			return runtime
