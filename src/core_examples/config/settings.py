from functools import lru_cache
from pathlib import Path

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
)
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from core_examples.utils.key_vault import get_secret


def _default_core_package_path() -> Path:
    return Path(__file__).resolve().parent.parent


class DomainSettings(BaseSettings):
    """Shared base model for nested repository settings domains."""

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        populate_by_name=True,
    )


class SecretBackedSettings(DomainSettings):
    key_vault_name: str | None = Field(
        default=None,
        validation_alias="AZURE_KEY_VAULT_NAME",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            KeyVaultFallbackSettingsSource(settings_cls),
            file_secret_settings,
        )


class LoggingSettings(DomainSettings):
    level: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LOG_LEVEL", "FRANK_LOG_LEVEL"),
    )
    to_file: bool = Field(
        default=False,
        validation_alias=AliasChoices("LOG_TO_FILE", "FRANK_LOG_TO_FILE"),
    )


class AzureSettings(SecretBackedSettings):
    blob_storage_name: str | None = Field(
        default=None,
        validation_alias="AZURE_BLOB_STORAGE_NAME",
        json_schema_extra={"key_vault_fallback": True},
    )
    search_service_endpoint: str | None = Field(
        default=None,
        validation_alias="AZURE_SEARCH_SERVICE_ENDPOINT",
        json_schema_extra={"key_vault_fallback": True},
    )
    search_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="AZURE_SEARCH_API_KEY",
        json_schema_extra={"key_vault_fallback": True},
    )
    # NOTE: The following settings are examples of how additional
    # Azure-related secrets could be configured with Key Vault fallback.
    # agents_function_app_api_key: SecretStr | None = Field(
    #     default=None,
    #     validation_alias="AZURE-FUNCTION-APP-API-KEY",
    #     json_schema_extra={"key_vault_fallback": True},
    # )
    # agents_function_app_base_url: AnyHttpUrl | None = Field(
    #     default=None,
    #     validation_alias="AZURE-FUNCTION-APP-BASE-URL",
    #     json_schema_extra={"key_vault_fallback": True},
    # )
    telemetry_connection_string: SecretStr | None = Field(
        default=None,
        validation_alias="APPLICATION_INSIGHTS_CONNECTION_STRING",
        json_schema_extra={"key_vault_fallback": True},
    )

    @property
    def telemetry_connection_string_value(self) -> str | None:
        return (
            value.get_secret_value()
            if (value := self.telemetry_connection_string) is not None
            else None
        )

    @property
    def search_api_key_value(self) -> str | None:
        return (
            value.get_secret_value()
            if (value := self.search_api_key) is not None
            else None
        )


class KeyVaultFallbackSettingsSource(PydanticBaseSettingsSource):
    """Resolve selected settings from Key Vault only after native settings sources."""

    KEY_VAULT_FALLBACK_FLAG = "key_vault_fallback"
    KEY_VAULT_NAME_FIELD = "key_vault_name"

    @staticmethod
    def _field_aliases(field: FieldInfo) -> tuple[str, ...]:
        alias = field.validation_alias
        if isinstance(alias, str):
            return (alias,)
        if isinstance(alias, AliasChoices):
            return tuple(choice for choice in alias.choices if isinstance(choice, str))
        return ()

    @classmethod
    def _state_keys(cls, field_name: str, field: FieldInfo) -> tuple[str, ...]:
        return (field_name, *cls._field_aliases(field))

    @classmethod
    def _uses_key_vault_fallback(cls, field: FieldInfo) -> bool:
        extra = field.json_schema_extra
        return isinstance(extra, dict) and extra.get(cls.KEY_VAULT_FALLBACK_FLAG) is True

    def _current_state_value(self, field_name: str, field: FieldInfo) -> object:
        for key in self._state_keys(field_name, field):
            if key in self.current_state:
                return self.current_state[key]

        return None

    def _key_vault_name(self) -> str | None:
        key_vault_name = self._current_state_value(
            self.KEY_VAULT_NAME_FIELD,
            self.settings_cls.model_fields[self.KEY_VAULT_NAME_FIELD],
        )
        if isinstance(key_vault_name, str) and key_vault_name:
            return key_vault_name
        return None

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[object, str, bool]:
        aliases = self._field_aliases(field)

        if not self._uses_key_vault_fallback(field) or not aliases:
            return None, field_name, False

        key_vault_name = self._key_vault_name()
        if key_vault_name is None:
            return None, field_name, False

        if self._current_state_value(field_name, field) is not None:
            return None, field_name, False

        secret_name = aliases[0]
        return get_secret(secret_name, required=False, key_vault_name=key_vault_name), secret_name, False

    def __call__(self) -> dict[str, object]:
        data: dict[str, object] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, resolved_key, _ = self.get_field_value(field, field_name)
            if field_value is not None:
                data[resolved_key] = field_value

        return data


class CoreSettings(BaseSettings):
    """Repository settings for the `core_examples` reference layer.

    These settings centralize paths and configuration file locations while
    keeping environment overrides available for repository runtimes.

    The YAML files remain the runtime source of truth for graph, node registry
    and logging templates; this model only centralizes their locations and
    environment-driven overrides.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
        nested_model_default_partial_update=True,
    )

    core_package_path: Path = Field(
        default_factory=_default_core_package_path,
        validation_alias="FRANK_CORE_PACKAGE_PATH",
    )
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)

    @property
    def src_directory_path(self) -> Path:
        return self.core_package_path.parent

    @property
    def project_root_path(self) -> Path:
        return self.src_directory_path.parent

    @property
    def logs_directory_path(self) -> Path:
        return self.project_root_path / "logs"

    @property
    def default_log_file_path(self) -> Path:
        return self.logs_directory_path / "application.log"

    @property
    def artifacts_directory_path(self) -> Path:
        return self.project_root_path / "artifacts"

    @property
    def config_directory_path(self) -> Path:
        return self.core_package_path / "config"

    @property
    def config_llms_file_path(self) -> Path:
        return self.config_directory_path / "config_llms.yml"

    @property
    def config_logging_file_path(self) -> Path:
        return self.config_directory_path / "config_logging.yml"

    @property
    def config_nodes_file_path(self) -> Path:
        return self.config_directory_path / "config_nodes.yml"


@lru_cache(maxsize=1)
def get_settings() -> CoreSettings:
    """Return cached repository settings for the reference layer."""
    return CoreSettings()