from pathlib import Path
from typing import Any, cast

import core_examples.config.settings as settings_module
from core_examples.config.settings import (
    AzureSettings,
    CoreSettings,
    LoggingSettings,
    get_settings,
)


def test_core_settings_resolve_default_repository_paths() -> None:
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.core_package_path == Path(__file__).resolve().parents[2] / "src" / "core_examples"
    assert settings.src_directory_path == Path(__file__).resolve().parents[2] / "src"
    assert settings.project_root_path == Path(__file__).resolve().parents[2]
    assert settings.config_llms_file_path == settings.config_directory_path / "config_llms.yml"
    assert settings.config_logging_file_path == settings.config_directory_path / "config_logging.yml"
    assert settings.config_nodes_file_path == settings.config_directory_path / "config_nodes.yml"


def test_core_settings_allow_environment_override(monkeypatch) -> None:
    override_package_path = Path("/tmp/frankenst-core/src/core_examples")
    get_settings.cache_clear()
    monkeypatch.setenv("FRANK_CORE_PACKAGE_PATH", str(override_package_path))

    settings = get_settings()

    assert settings.core_package_path == override_package_path
    assert settings.config_llms_file_path == override_package_path / "config" / "config_llms.yml"

    get_settings.cache_clear()


def test_core_settings_allow_init_override_for_core_package_path() -> None:
    override_package_path = Path("/tmp/frankenst-core/src/core_examples")

    settings = CoreSettings(core_package_path=override_package_path)

    assert settings.core_package_path == override_package_path
    assert settings.config_llms_file_path == override_package_path / "config" / "config_llms.yml"

    get_settings.cache_clear()


def test_core_settings_read_logging_and_key_vault_from_standard_env_names(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_TO_FILE", "true")

    settings = get_settings()

    assert settings.logging.level == "DEBUG"
    assert settings.logging.to_file is True

    get_settings.cache_clear()


def test_core_settings_can_load_standard_env_names_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LOG_LEVEL=WARNING\nLOG_TO_FILE=true\n",
        encoding="utf-8",
    )

    settings = CoreSettings(logging=cast(Any, LoggingSettings)(_env_file=env_file))

    assert settings.logging.level == "WARNING"
    assert settings.logging.to_file is True


def test_core_settings_expose_nested_domains(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("AZURE_KEY_VAULT_NAME", "frankenst-kv")
    monkeypatch.setenv("AZURE_BLOB_STORAGE_NAME", "blob-from-env")
    monkeypatch.setenv("APPLICATION_INSIGHTS_CONNECTION_STRING", "telemetry-from-env")
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Key Vault should not be called")),
    )

    settings = get_settings()

    assert settings.logging.level == "ERROR"
    assert settings.azure.key_vault_name == "frankenst-kv"
    assert settings.config_llms_file_path == settings.config_directory_path / "config_llms.yml"
    assert settings.config_nodes_file_path == settings.config_directory_path / "config_nodes.yml"

    get_settings.cache_clear()


def test_azure_settings_can_fall_back_to_key_vault_for_blob_storage_name(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_BLOB_STORAGE_NAME", raising=False)
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda secret_name, *, required=False, key_vault_name=None: "blob-from-kv"
        if secret_name == "AZURE_BLOB_STORAGE_NAME" and key_vault_name == "frankenst-kv"
        else None,
    )

    settings = AzureSettings(key_vault_name="frankenst-kv")

    assert settings.blob_storage_name == "blob-from-kv"


def test_azure_settings_prefer_env_before_key_vault_fallback(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_BLOB_STORAGE_NAME", "blob-from-env")
    monkeypatch.setenv("APPLICATION_INSIGHTS_CONNECTION_STRING", "telemetry-from-env")
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Key Vault should not be called")),
    )

    settings = AzureSettings(key_vault_name="frankenst-kv")

    assert settings.blob_storage_name == "blob-from-env"


def test_azure_settings_can_fall_back_to_key_vault_for_telemetry_connection_string(monkeypatch) -> None:
    monkeypatch.delenv("APPLICATION_INSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda secret_name, *, required=False, key_vault_name=None: "telemetry-from-kv"
        if secret_name == "APPLICATION_INSIGHTS_CONNECTION_STRING"
        and key_vault_name == "frankenst-kv"
        else None,
    )

    settings = AzureSettings(key_vault_name="frankenst-kv")

    assert settings.telemetry_connection_string is not None
    assert settings.telemetry_connection_string.get_secret_value() == "telemetry-from-kv"
    assert settings.telemetry_connection_string_value == "telemetry-from-kv"


def test_azure_settings_prefer_env_for_telemetry_before_key_vault_fallback(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_BLOB_STORAGE_NAME", "blob-from-env")
    monkeypatch.setenv("APPLICATION_INSIGHTS_CONNECTION_STRING", "telemetry-from-env")
    monkeypatch.setattr(
        settings_module,
        "get_secret",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Key Vault should not be called")),
    )

    settings = AzureSettings(key_vault_name="frankenst-kv")

    assert settings.telemetry_connection_string_value == "telemetry-from-env"