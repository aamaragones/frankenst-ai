from collections.abc import Callable
from typing import Any, cast

import pytest
from azure.identity import DefaultAzureCredential

import config.settings as settings_module
from services.llm import llm_services as llms_module
from services.llm.llm_services import LLMRuntime, LLMServices, _class_key

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
) -> None:
    classes = {
        "chat": chat_factory or CaptureFactory(),
        "embeddings": embeddings_factory or CaptureFactory(),
    }
    monkeypatch.setattr(LLMServices, importer_name, lambda: classes)


def _patch_secrets(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda name, *, required=True, key_vault_name=None: values[name],
    )


def _kwargs(client: Any) -> dict[str, Any]:
    return cast(dict[str, Any], client)["kwargs"]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("embeddings", "embeddings"),
        ("fast_embeddings", "embeddings"),
        ("model", "chat"),
        ("turbo_model", "chat"),
        ("judge", "chat"),
        ("embeddings_judge", "chat"),
    ],
)
def test_class_key_picks_embeddings_only_by_suffix(kind: str, expected: str) -> None:
    assert _class_key(kind) == expected


def test_build_runtime_uses_nested_ollama_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_factory, embeddings_factory = CaptureFactory(), CaptureFactory()
    _patch_provider_classes(
        monkeypatch, "_ollama_classes", chat_factory, embeddings_factory
    )
    monkeypatch.setattr(
        llms_module,
        "resolve_ollama_base_url",
        lambda config_host=None: "http://ollama.local",
    )
    config = {
        "launch": {"model": "ollama", "embeddings": "ollama"},
        "ollama": {
            "model": {"model": "gemma4:e4b", "temperature": 0},
            "embeddings": {"model": "embeddinggemma"},
        },
    }

    runtime = LLMServices.build_runtime(config)

    assert runtime.kinds == ("model", "embeddings")
    assert _kwargs(runtime.model) == {
        "model": "gemma4:e4b",
        "temperature": 0,
        "base_url": "http://ollama.local",
    }
    assert _kwargs(runtime.embeddings) == {
        "model": "embeddinggemma",
        "base_url": "http://ollama.local",
    }


def test_build_runtime_creates_one_client_per_launch_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama_chat, ollama_embeddings = CaptureFactory(), CaptureFactory()
    databricks_chat, databricks_embeddings = CaptureFactory(), CaptureFactory()
    _patch_provider_classes(
        monkeypatch, "_ollama_classes", ollama_chat, ollama_embeddings
    )
    _patch_provider_classes(
        monkeypatch, "_databricks_classes", databricks_chat, databricks_embeddings
    )
    monkeypatch.setattr(
        llms_module,
        "resolve_ollama_base_url",
        lambda config_host=None: "http://ollama.local",
    )
    config = {
        "launch": {
            "model": "ollama",
            "embeddings": "ollama",
            "judge": "databricks",
            "fast_embeddings": "databricks",
        },
        "ollama": {
            "model": {"model": "gemma4"},
            "embeddings": {"model": "embeddinggemma"},
        },
        "databricks": {
            "judge": {"model": "judge-endpoint"},
            "fast_embeddings": {"model": "bge"},
        },
    }

    runtime = LLMServices.build_runtime(config)

    assert runtime.kinds == ("model", "embeddings", "judge", "fast_embeddings")
    assert "judge" in runtime and "nope" not in runtime
    assert _kwargs(runtime.judge)["model"] == "judge-endpoint"
    assert databricks_chat.calls == [{"model": "judge-endpoint"}]
    assert databricks_embeddings.calls == [{"model": "bge"}]
    assert ollama_chat.calls and ollama_embeddings.calls


def test_build_runtime_resolves_secrets_through_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_factory, embeddings_factory = CaptureFactory(), CaptureFactory()
    _patch_provider_classes(
        monkeypatch, "_azure_ai_classes", chat_factory, embeddings_factory
    )
    monkeypatch.setenv("AZURE_KEY_VAULT_NAME", "frankenst-kv")
    seen: list[tuple[str, str | None]] = []

    values = {
        "CHAT_ENDPOINT": "https://chat.example",
        "CHAT_MODEL": "gpt",
        "CHAT_CREDENTIAL": "k",
    }

    def fake_get_secret(
        name: str, *, required: bool = True, key_vault_name: str | None = None
    ) -> str | None:
        if name in values:
            seen.append((name, key_vault_name))
        return values.get(name)

    monkeypatch.setattr(settings_module, "get_secret", fake_get_secret)
    config = {
        "launch": {"model": "azure_ai"},
        "azure_ai": {
            "model": {
                "endpoint": {"secret": "CHAT_ENDPOINT"},
                "model": {"secret": "CHAT_MODEL"},
                "credential": {"secret": "CHAT_CREDENTIAL"},
                "temperature": 0,
            }
        },
    }

    runtime = LLMServices.build_runtime(config)

    assert _kwargs(runtime.model) == {
        "endpoint": "https://chat.example",
        "model": "gpt",
        "credential": "k",
        "temperature": 0,
    }
    assert seen == [
        ("CHAT_ENDPOINT", "frankenst-kv"),
        ("CHAT_MODEL", "frankenst-kv"),
        ("CHAT_CREDENTIAL", "frankenst-kv"),
    ]
    assert embeddings_factory.calls == []


def test_azure_ai_uses_default_credential_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_factory = CaptureFactory()
    _patch_provider_classes(monkeypatch, "_azure_ai_classes", chat_factory)
    config = {
        "launch": {"model": "azure_ai"},
        "azure_ai": {
            "model": {"project_endpoint": "https://p.example", "model": "gpt"}
        },
    }

    runtime = LLMServices.build_runtime(config)

    kwargs = _kwargs(runtime.model)
    assert isinstance(kwargs["credential"], DefaultAzureCredential)
    assert "use_responses_api" not in kwargs


def test_azure_ai_forwards_use_responses_api_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_factory = CaptureFactory()
    _patch_provider_classes(monkeypatch, "_azure_ai_classes", chat_factory)
    config = {
        "launch": {"model": "azure_ai"},
        "azure_ai": {
            "model": {
                "endpoint": "https://e",
                "model": "gpt",
                "credential": "k",
                "use_responses_api": True,
            }
        },
    }

    assert _kwargs(LLMServices.build_runtime(config).model)["use_responses_api"] is True


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (
            {"endpoint": "https://e", "project_endpoint": "https://p", "model": "m"},
            "cannot define both",
        ),
        ({"model": "m"}, "azure_ai.model.endpoint or azure_ai.model.project_endpoint"),
        ({"endpoint": "https://e"}, "azure_ai.model.model"),
    ],
)
def test_azure_ai_validates_the_local_contract(
    monkeypatch: pytest.MonkeyPatch, section: dict[str, Any], message: str
) -> None:
    _patch_provider_classes(monkeypatch, "_azure_ai_classes")

    with pytest.raises(RuntimeError, match=message):
        LLMServices.build_runtime(
            {"launch": {"model": "azure_ai"}, "azure_ai": {"model": section}}
        )


def test_build_runtime_resolves_nested_databricks_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_factory, embeddings_factory = CaptureFactory(), CaptureFactory()
    _patch_provider_classes(
        monkeypatch, "_databricks_classes", chat_factory, embeddings_factory
    )
    _patch_secrets(monkeypatch, {"LLM": "chat-endpoint", "EMB": "embed-endpoint"})
    config = {
        "launch": {"model": "databricks", "embeddings": "databricks"},
        "databricks": {
            "model": {
                "model": {"secret": "LLM"},
                "use_ai_gateway": True,
                "use_responses_api": True,
            },
            "embeddings": {"endpoint": {"secret": "EMB"}},
        },
    }

    runtime = LLMServices.build_runtime(config)

    assert _kwargs(runtime.model) == {
        "model": "chat-endpoint",
        "use_ai_gateway": True,
        "use_responses_api": True,
    }
    assert _kwargs(runtime.embeddings) == {"endpoint": "embed-endpoint"}


def test_databricks_requires_model_or_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider_classes(monkeypatch, "_databricks_classes")

    with pytest.raises(RuntimeError, match="must define model"):
        LLMServices.build_runtime(
            {
                "launch": {"model": "databricks"},
                "databricks": {"model": {"temperature": 0}},
            }
        )


def test_databricks_rejects_endpoint_and_model_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provider_classes(monkeypatch, "_databricks_classes")

    with pytest.raises(RuntimeError, match="cannot define both endpoint and model"):
        LLMServices.build_runtime(
            {
                "launch": {"model": "databricks"},
                "databricks": {"model": {"endpoint": "a", "model": "b"}},
            }
        )


def test_launch_key_without_provider_section_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provider_classes(monkeypatch, "_databricks_classes")

    with pytest.raises(
        RuntimeError,
        match=r"launch\.judge selects databricks but databricks\.judge is not declared",
    ):
        LLMServices.build_runtime(
            {"launch": {"judge": "databricks"}, "databricks": {"model": {"model": "m"}}}
        )


@pytest.mark.parametrize(
    "config", [{}, {"launch": None}, {"ollama": {"model": {"model": "m"}}}]
)
def test_missing_launch_section_raises(config: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="Missing config section for: launch"):
        LLMServices.build_runtime(config)


def test_empty_launch_section_raises() -> None:
    with pytest.raises(RuntimeError, match="No runtime declared"):
        LLMServices.build_runtime({"launch": {}})


def test_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported provider type: bedrock"):
        LLMServices.build_runtime(
            {"launch": {"model": "bedrock"}, "bedrock": {"model": {}}}
        )


def test_never_imports_undeclared_provider_packages(
    monkeypatch: pytest.MonkeyPatch,
    raises_on_call: Callable[[BaseException], Callable[..., Any]],
) -> None:
    _patch_provider_classes(monkeypatch, "_ollama_classes")
    monkeypatch.setattr(
        llms_module,
        "resolve_ollama_base_url",
        lambda config_host=None: "http://ollama.local",
    )
    monkeypatch.setattr(
        LLMServices,
        "_azure_ai_classes",
        raises_on_call(AssertionError("azure imported")),
    )
    monkeypatch.setattr(
        LLMServices,
        "_databricks_classes",
        raises_on_call(AssertionError("databricks imported")),
    )
    config = {
        "launch": {"model": "ollama"},
        "ollama": {"model": {"model": "gemma4"}},
        "azure_ai": {"model": {"endpoint": "https://e", "model": "gpt"}},
        "databricks": {"model": {"model": "m"}},
    }

    assert LLMServices.build_runtime(config).kinds == ("model",)


def test_runtime_require_returns_clients_in_order_and_names_missing() -> None:
    runtime = LLMRuntime(
        llms_module.MappingProxyType({"model": "m", "embeddings": "e"})
    )

    assert runtime.require("embeddings", "model") == ("e", "m")
    with pytest.raises(RuntimeError, match=r"\['judge'\].*\['embeddings', 'model'\]"):
        runtime.require("model", "judge")
    with pytest.raises(AttributeError, match="No runtime named 'judge'"):
        _ = runtime.judge


def test_launch_publishes_once_and_force_reload_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: list[dict[str, Any] | None] = []

    def fake_build(config: dict[str, Any] | None = None) -> LLMRuntime:
        builds.append(config)
        return LLMRuntime(llms_module.MappingProxyType({"model": object()}))

    monkeypatch.setattr(LLMServices, "build_runtime", fake_build)

    first = LLMServices.launch({"launch": {"model": "ollama"}})
    second = LLMServices.launch()
    third = LLMServices.launch(force_reload=True)

    assert first is second and third is not first
    assert builds == [{"launch": {"model": "ollama"}}, None]
    assert LLMServices.get("model") is third.model


def test_launch_publishes_nothing_when_the_build_fails(
    monkeypatch: pytest.MonkeyPatch,
    raises_on_call: Callable[[BaseException], Callable[..., Any]],
) -> None:
    monkeypatch.setattr(
        LLMServices, "build_runtime", raises_on_call(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        LLMServices.launch()
    with pytest.raises(RuntimeError, match="launch\\(\\) has not been called"):
        LLMServices.get("model")


def test_get_rejects_undeclared_kind_after_launch(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    published_runtime(model="m")

    assert LLMServices.get("model") == "m"
    with pytest.raises(RuntimeError, match="declares no runtime for \\['judge'\\]"):
        LLMServices.get("judge")


def test_launch_reads_config_llms_from_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _patch_provider_classes(monkeypatch, "_ollama_classes")
    monkeypatch.setattr(
        llms_module,
        "resolve_ollama_base_url",
        lambda config_host=None: "http://ollama.local",
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config_llms.yaml").write_text(
        "ollama:\n  model:\n    model: gemma4\nlaunch:\n  model: ollama\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FRANK_CORE_PACKAGE_PATH", str(tmp_path))

    assert LLMServices.launch().kinds == ("model",)
