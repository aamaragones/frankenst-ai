from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_settings import BaseSettings

import config.settings as settings_module
from config.settings import AzureSettings, CoreSettings, LoggingSettings, get_settings

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _never_called(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("Key Vault should not be called")


def test_core_settings_resolve_default_repository_paths() -> None:
    settings = get_settings()

    assert settings.core_package_path == REPO_ROOT / "src"
    assert settings.project_root_path == REPO_ROOT
    assert settings.config_directory_path == REPO_ROOT / "src" / "config"
    assert (
        settings.config_llms_file_path
        == settings.config_directory_path / "config_llms.yaml"
    )
    assert (
        settings.config_logging_file_path
        == settings.config_directory_path / "config_logging.yaml"
    )
    assert (
        settings.config_nodes_file_path
        == settings.config_directory_path / "config_nodes.yaml"
    )


def test_core_settings_allow_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = Path("/tmp/frankenst-core/src")
    monkeypatch.setenv("FRANK_CORE_PACKAGE_PATH", str(override))

    settings = get_settings()

    assert settings.core_package_path == override
    assert settings.config_llms_file_path == override / "config" / "config_llms.yaml"


def test_core_settings_allow_init_override_for_core_package_path() -> None:
    override = Path("/tmp/frankenst-core/src")

    settings = CoreSettings(core_package_path=override)

    assert settings.project_root_path == Path("/tmp/frankenst-core")
    assert settings.config_llms_file_path == override / "config" / "config_llms.yaml"


def test_core_settings_read_logging_from_standard_env_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_TO_FILE", "true")

    settings = get_settings()

    assert settings.logging.level == "DEBUG"
    assert settings.logging.to_file is True


def test_core_settings_can_load_standard_env_names_from_env_file(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LOG_LEVEL=WARNING\nLOG_TO_FILE=true\n", encoding="utf-8")

    settings = CoreSettings(logging=cast(Any, LoggingSettings)(_env_file=env_file))

    assert settings.logging.level == "WARNING"
    assert settings.logging.to_file is True


def test_core_settings_expose_nested_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("AZURE_KEY_VAULT_NAME", "frankenst-kv")
    monkeypatch.setenv("AZURE_BLOB_STORAGE_NAME", "blob-from-env")
    monkeypatch.setenv("APPLICATION_INSIGHTS_CONNECTION_STRING", "telemetry-from-env")
    monkeypatch.setenv("AZURE_SEARCH_SERVICE_ENDPOINT", "https://search.example")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "search-from-env")
    monkeypatch.setattr(settings_module, "get_secret", _never_called)

    settings = get_settings()

    assert settings.logging.level == "ERROR"
    assert settings.azure.key_vault_name == "frankenst-kv"
    assert settings.azure.search_api_key_value == "search-from-env"


def test_azure_settings_can_fall_back_to_key_vault_for_blob_storage_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_secret(
        secret_name: str, *, required: bool = False, key_vault_name: str | None = None
    ) -> str | None:
        if (
            secret_name == "AZURE_BLOB_STORAGE_NAME"
            and key_vault_name == "frankenst-kv"
        ):
            return "blob-from-kv"
        return None

    monkeypatch.setattr(settings_module, "get_secret", fake_get_secret)

    settings = AzureSettings(key_vault_name="frankenst-kv")

    assert settings.blob_storage_name == "blob-from-kv"


def test_azure_settings_prefer_env_before_key_vault_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_BLOB_STORAGE_NAME", "blob-from-env")
    monkeypatch.setenv("APPLICATION_INSIGHTS_CONNECTION_STRING", "telemetry-from-env")
    monkeypatch.setenv("AZURE_SEARCH_SERVICE_ENDPOINT", "https://search.example")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "search-from-env")
    monkeypatch.setattr(settings_module, "get_secret", _never_called)

    settings = AzureSettings(key_vault_name="frankenst-kv")

    assert settings.blob_storage_name == "blob-from-env"
    assert settings.telemetry_connection_string_value == "telemetry-from-env"


def test_azure_settings_can_fall_back_to_key_vault_for_telemetry_connection_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_secret(
        secret_name: str, *, required: bool = False, key_vault_name: str | None = None
    ) -> str | None:
        if (
            secret_name == "APPLICATION_INSIGHTS_CONNECTION_STRING"
            and key_vault_name == "frankenst-kv"
        ):
            return "telemetry-from-kv"
        return None

    monkeypatch.setattr(settings_module, "get_secret", fake_get_secret)

    settings = AzureSettings(key_vault_name="frankenst-kv")

    assert settings.telemetry_connection_string_value == "telemetry-from-kv"


def test_resolve_secret_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_KEY_VAULT_NAME", "frankenst-kv")
    monkeypatch.setenv("MY_SECRET", "from-env")
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda name, *, required, key_vault_name: "env-only",
    )

    assert get_settings().resolve_secret("MY_SECRET") == "env-only"


def test_resolve_secret_passes_the_settings_vault_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_KEY_VAULT_NAME", "frankenst-kv")
    seen: dict[str, Any] = {}

    def fake_get_secret(
        name: str, *, required: bool, key_vault_name: str | None
    ) -> str:
        seen.update(name=name, required=required, key_vault_name=key_vault_name)
        return "from-kv"

    monkeypatch.setattr(settings_module, "get_secret", fake_get_secret)

    assert get_settings().resolve_secret("MY_SECRET", required=False) == "from-kv"
    assert seen == {
        "name": "MY_SECRET",
        "required": False,
        "key_vault_name": "frankenst-kv",
    }


def test_settings_have_no_empty_string_defaults() -> None:
    for name in dir(settings_module):
        candidate = getattr(settings_module, name)
        if not (isinstance(candidate, type) and issubclass(candidate, BaseSettings)):
            continue
        for field_name, field in candidate.model_fields.items():
            assert field.default != "", (
                f"{candidate.__name__}.{field_name} defaults to ''"
            )
