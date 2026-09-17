"""Provider-agnostic chat and embeddings runtimes, built from config_llms.yaml."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any, Literal

from config.settings import get_settings
from utils.config_loader import read_yaml
from utils.ollama.ollama_wsl_proxy import resolve_ollama_base_url

logger = logging.getLogger(__name__)

_ClassKey = Literal["chat", "embeddings"]


def _class_key(kind: str) -> _ClassKey:
    """`embeddings` or `*_embeddings` builds the embeddings client; anything else chat."""
    if kind == "embeddings" or kind.endswith("_embeddings"):
        return "embeddings"
    return "chat"


@dataclass(frozen=True)
class _ProviderSpec:
    import_classes: Callable[[], dict[_ClassKey, type[Any]]]
    prepare_kwargs: Callable[[dict[str, Any], str], dict[str, Any]]


@dataclass(frozen=True)
class LLMRuntime:
    """One client per `launch` key, reachable as an attribute or through `require`."""

    _clients: Mapping[str, Any]

    @property
    def kinds(self) -> tuple[str, ...]:
        """The `launch` keys, in declaration order."""
        return tuple(self._clients)

    def __contains__(self, kind: object) -> bool:
        """Whether `launch` declared this kind."""
        return kind in self._clients

    def __getattr__(self, kind: str) -> Any:
        """`runtime.judge` is the client launched under `judge`."""
        if kind.startswith("_"):
            raise AttributeError(kind)
        try:
            return self._clients[kind]
        except KeyError:
            raise AttributeError(
                f"No runtime named {kind!r}; declared: {sorted(self._clients)}"
            ) from None

    def require(self, *kinds: str) -> tuple[Any, ...]:
        """Clients in the order asked; `RuntimeError` names what is missing."""
        missing = [kind for kind in kinds if kind not in self._clients]
        if missing:
            raise RuntimeError(
                f"config_llms.yaml declares no runtime for {missing}; "
                f"declared: {sorted(self._clients)}"
            )
        return tuple(self._clients[kind] for kind in kinds)


class LLMServices:
    """Build every runtime `launch:` names and publish them once per process.

    `launch.<kind>: <provider>` plus `<provider>.<kind>` (constructor kwargs) is one
    runtime. Any kind name works, so a new line in the yaml is a new model. A
    provider package is imported only when a launch key selects it.
    """

    _runtime: LLMRuntime | None = None
    _launch_lock = Lock()

    @staticmethod
    def _ollama_classes() -> dict[_ClassKey, type[Any]]:
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        return {"chat": ChatOllama, "embeddings": OllamaEmbeddings}

    @staticmethod
    def _azure_ai_classes() -> dict[_ClassKey, type[Any]]:
        from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
        from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel

        return {
            "chat": AzureAIOpenAIApiChatModel,
            "embeddings": AzureAIOpenAIApiEmbeddingsModel,
        }

    @staticmethod
    def _databricks_classes() -> dict[_ClassKey, type[Any]]:
        from databricks_langchain import ChatDatabricks, DatabricksEmbeddings

        return {"chat": ChatDatabricks, "embeddings": DatabricksEmbeddings}

    @classmethod
    def _providers(cls) -> dict[str, _ProviderSpec]:
        return {
            "ollama": _ProviderSpec(cls._ollama_classes, cls._prepare_ollama_kwargs),
            "azure_ai": _ProviderSpec(
                cls._azure_ai_classes, cls._prepare_azure_ai_kwargs
            ),
            "databricks": _ProviderSpec(
                cls._databricks_classes, cls._prepare_databricks_kwargs
            ),
        }

    @classmethod
    def _load_config(cls, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if config is None:
            path = get_settings().config_llms_file_path
            logger.info("Loading LLM runtime configuration from %s.", path)
            config = read_yaml(path)
        launch = config.get("launch")
        if not isinstance(launch, dict):
            raise RuntimeError("Missing config section for: launch")
        if not launch:
            raise RuntimeError(
                "No runtime declared: launch must map at least one kind to a provider."
            )
        return config

    @classmethod
    def _resolve_config_value(cls, value: Any) -> Any:
        """Resolve `{secret: NAME}` leaves through settings, recursively."""
        if isinstance(value, dict):
            if set(value) == {"secret"}:
                name = value["secret"]
                if not isinstance(name, str) or not name:
                    raise RuntimeError(
                        "Config secret references must be non-empty strings."
                    )
                return get_settings().resolve_secret(name)
            return {key: cls._resolve_config_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._resolve_config_value(item) for item in value]
        return value

    @classmethod
    def _resolve_runtime_kwargs(cls, runtime_config: dict[str, Any]) -> dict[str, Any]:
        resolved = cls._resolve_config_value(runtime_config)
        if not isinstance(resolved, dict):
            raise RuntimeError("Runtime configuration must resolve to a mapping.")
        return {key: value for key, value in resolved.items() if value is not None}

    @classmethod
    def _prepare_ollama_kwargs(
        cls, runtime_config: dict[str, Any], config_path: str
    ) -> dict[str, Any]:
        kwargs = cls._resolve_runtime_kwargs(runtime_config)
        host = kwargs.pop("host", None)
        if "base_url" not in kwargs:
            kwargs["base_url"] = resolve_ollama_base_url(config_host=host)
        if not kwargs.get("model"):
            raise RuntimeError(f"Missing config entry for: {config_path}.model")
        return kwargs

    @classmethod
    def _prepare_azure_ai_kwargs(
        cls, runtime_config: dict[str, Any], config_path: str
    ) -> dict[str, Any]:
        """Validate the local contract; `langchain_azure_ai` validates the rest.

        `use_responses_api` is not defaulted here: the yaml declares it, because
        whether a Foundry region serves /v1/responses is deployment data.
        """
        kwargs = cls._resolve_runtime_kwargs(runtime_config)
        if kwargs.get("endpoint") and kwargs.get("project_endpoint"):
            raise RuntimeError(
                f"Config section {config_path} cannot define both endpoint and project_endpoint."
            )
        if not kwargs.get("endpoint") and not kwargs.get("project_endpoint"):
            raise RuntimeError(
                f"Missing config entry for: {config_path}.endpoint or {config_path}.project_endpoint"
            )
        if not kwargs.get("model"):
            raise RuntimeError(f"Missing config entry for: {config_path}.model")
        if not kwargs.get("credential"):
            from azure.identity import DefaultAzureCredential

            kwargs["credential"] = DefaultAzureCredential()
        logger.info(
            "Preparing Azure AI runtime for %s: model=%s use_responses_api=%s",
            config_path,
            kwargs.get("model"),
            kwargs.get("use_responses_api"),
        )
        return kwargs

    @classmethod
    def _prepare_databricks_kwargs(
        cls, runtime_config: dict[str, Any], config_path: str
    ) -> dict[str, Any]:
        """`model` names the endpoint; `endpoint` is the deprecated alias. Not both.

        No credential is injected: the Databricks SDK default auth chain handles it.
        """
        kwargs = cls._resolve_runtime_kwargs(runtime_config)
        endpoint, model = kwargs.get("endpoint"), kwargs.get("model")
        if endpoint and model:
            raise RuntimeError(
                f"Config section {config_path} cannot define both endpoint and model."
            )
        if not (endpoint or model):
            raise RuntimeError(
                f"Config section {config_path} must define model "
                "(or the deprecated endpoint alias)."
            )
        logger.info(
            "Preparing Databricks runtime for %s: endpoint=%s use_responses_api=%s",
            config_path,
            endpoint or model,
            kwargs.get("use_responses_api"),
        )
        return kwargs

    @classmethod
    def _load_declared_runtime(cls, config: dict[str, Any], kind: str) -> Any:
        provider_name = config["launch"][kind]
        spec = cls._providers().get(provider_name)
        if spec is None:
            raise ValueError(f"Unsupported provider type: {provider_name}")
        config_path = f"{provider_name}.{kind}"
        provider_section = config.get(provider_name)
        runtime_config = (
            provider_section.get(kind) if isinstance(provider_section, dict) else None
        )
        if not isinstance(runtime_config, dict):
            raise RuntimeError(
                f"launch.{kind} selects {provider_name} but {config_path} is not declared."
            )
        kwargs = spec.prepare_kwargs(runtime_config, config_path)
        runtime = spec.import_classes()[_class_key(kind)](**kwargs)
        logger.info("Loaded %s from %s: %s.", kind, config_path, type(runtime).__name__)
        return runtime

    @classmethod
    def build_runtime(cls, config: dict[str, Any] | None = None) -> LLMRuntime:
        """A fresh runtime from config; publishes nothing."""
        resolved = cls._load_config(config)
        clients = {
            kind: cls._load_declared_runtime(resolved, kind)
            for kind in resolved["launch"]
        }
        return LLMRuntime(MappingProxyType(clients))

    @classmethod
    def launch(
        cls, config: dict[str, Any] | None = None, *, force_reload: bool = False
    ) -> LLMRuntime:
        """The shared runtime, built on first call; `force_reload` rebuilds it."""
        if not force_reload and cls._runtime is not None:
            return cls._runtime
        with cls._launch_lock:
            if not force_reload and cls._runtime is not None:
                logger.debug("LLMServices.launch reusing the cached runtime.")
                return cls._runtime
            runtime = cls.build_runtime(config)
            cls._runtime = runtime
            logger.info("LLMServices.launch published runtimes: %s.", runtime.kinds)
            return runtime

    @classmethod
    def get(cls, kind: str) -> Any:
        """The published client named `kind`; `RuntimeError` before `launch()`."""
        if cls._runtime is None:
            raise RuntimeError("LLMServices.launch() has not been called.")
        (client,) = cls._runtime.require(kind)
        return client

    @classmethod
    def reset(cls) -> None:
        """Forget the published runtime; the test seam."""
        cls._runtime = None
