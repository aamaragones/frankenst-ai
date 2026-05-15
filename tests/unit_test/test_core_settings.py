from pathlib import Path

from core_examples.config.settings import CoreSettings, get_settings


def test_core_settings_resolve_default_repository_paths() -> None:
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.core_package_path == Path(__file__).resolve().parents[2] / "src" / "core_examples"
    assert settings.src_directory_path == Path(__file__).resolve().parents[2] / "src"
    assert settings.project_root_path == Path(__file__).resolve().parents[2]
    assert settings.config_file_path == settings.config_directory_path / "config.yml"
    assert settings.config_logging_file_path == settings.config_directory_path / "config_logging.yml"
    assert settings.config_nodes_file_path == settings.config_directory_path / "config_nodes.yml"


def test_core_settings_allow_environment_override(monkeypatch) -> None:
    override_path = Path("/tmp/frankenst-config.yml")
    get_settings.cache_clear()
    monkeypatch.setenv("FRANK_CONFIG_FILE_PATH", str(override_path))

    settings = get_settings()

    assert settings.config_file_path == override_path

    get_settings.cache_clear()


def test_core_settings_read_logging_and_key_vault_from_standard_env_names(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_TO_FILE", "true")

    settings = get_settings()

    assert settings.log_level == "DEBUG"
    assert settings.log_to_file is True

    get_settings.cache_clear()


def test_core_settings_can_load_standard_env_names_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LOG_LEVEL=WARNING\nLOG_TO_FILE=true\n",
        encoding="utf-8",
    )

    settings = CoreSettings(_env_file=env_file)

    assert settings.log_level == "WARNING"
    assert settings.log_to_file is True