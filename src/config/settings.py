"""Typed settings: the single entry point for configuration paths and secrets."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, overload

from pydantic import AliasChoices, Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from utils.secrets import get_secret


def _default_core_package_path() -> Path:
    return Path(__file__).resolve().parent.parent


class DomainSettings(BaseSettings):
    """Base for the nested settings domains: env, then .env."""

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        populate_by_name=True,
    )


class KeyVaultFallbackSettingsSource(PydanticBaseSettingsSource):
    """Fill flagged fields from Key Vault only after env and .env said nothing."""

    KEY_VAULT_FALLBACK_FLAG = "key_vault_fallback"
    KEY_VAULT_NAME_FIELD = "key_vault_name"

    @staticmethod
    def _field_aliases(field: FieldInfo) -> tuple[str, ...]:
        alias = field.validation_alias
        if isinstance(alias, str):
            return (alias,)
        if isinstance(alias, AliasChoices):
            return tuple(c for c in alias.choices if isinstance(c, str))
        return ()

    @classmethod
    def _uses_key_vault_fallback(cls, field: FieldInfo) -> bool:
        extra = field.json_schema_extra
        return (
            isinstance(extra, dict) and extra.get(cls.KEY_VAULT_FALLBACK_FLAG) is True
        )

    def _current_state_value(self, field_name: str, field: FieldInfo) -> object:
        for key in (field_name, *self._field_aliases(field)):
            if key in self.current_state:
                return self.current_state[key]
        return None

    def _key_vault_name(self) -> str | None:
        name = self._current_state_value(
            self.KEY_VAULT_NAME_FIELD,
            self.settings_cls.model_fields[self.KEY_VAULT_NAME_FIELD],
        )
        return name if isinstance(name, str) and name else None

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[object, str, bool]:
        aliases = self._field_aliases(field)
        if not self._uses_key_vault_fallback(field) or not aliases:
            return None, field_name, False
        key_vault_name = self._key_vault_name()
        if key_vault_name is None:
            return None, field_name, False
        if self._current_state_value(field_name, field) is not None:
            return None, field_name, False
        value = get_secret(aliases[0], required=False, key_vault_name=key_vault_name)
        return value, aliases[0], False

    def __call__(self) -> dict[str, object]:
        data: dict[str, object] = {}
        for field_name, field in self.settings_cls.model_fields.items():
            value, key, _ = self.get_field_value(field, field_name)
            if value is not None:
                data[key] = value
        return data


class SecretBackedSettings(DomainSettings):
    """A domain whose flagged fields may come from the Key Vault it names."""

    key_vault_name: str | None = Field(
        default=None, validation_alias="AZURE_KEY_VAULT_NAME"
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
    """Root log level and whether a file handler is attached."""

    level: str | None = Field(
        default=None, validation_alias=AliasChoices("LOG_LEVEL", "FRANK_LOG_LEVEL")
    )
    to_file: bool = Field(
        default=False, validation_alias=AliasChoices("LOG_TO_FILE", "FRANK_LOG_TO_FILE")
    )


class AzureSettings(SecretBackedSettings):
    """Azure resource names and keys the examples reach for."""

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
    telemetry_connection_string: SecretStr | None = Field(
        default=None,
        validation_alias="APPLICATION_INSIGHTS_CONNECTION_STRING",
        json_schema_extra={"key_vault_fallback": True},
    )

    @property
    def telemetry_connection_string_value(self) -> str | None:
        value = self.telemetry_connection_string
        return value.get_secret_value() if value is not None else None

    @property
    def search_api_key_value(self) -> str | None:
        value = self.search_api_key
        return value.get_secret_value() if value is not None else None


class CoreSettings(BaseSettings):
    """Locates the YAML config files, which stay the runtime source of truth."""

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
    def project_root_path(self) -> Path:
        return self.core_package_path.parent

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
        return self.config_directory_path / "config_llms.yaml"

    @property
    def config_logging_file_path(self) -> Path:
        return self.config_directory_path / "config_logging.yaml"

    @property
    def config_nodes_file_path(self) -> Path:
        return self.config_directory_path / "config_nodes.yaml"

    @overload
    def resolve_secret(self, name: str, *, required: Literal[True] = True) -> str: ...

    @overload
    def resolve_secret(self, name: str, *, required: Literal[False]) -> str | None: ...

    def resolve_secret(self, name: str, *, required: bool = True) -> str | None:
        """Env first, then the Key Vault named by AZURE_KEY_VAULT_NAME."""
        vault = self.azure.key_vault_name
        if required:
            return get_secret(name, required=True, key_vault_name=vault)
        return get_secret(name, required=False, key_vault_name=vault)


@lru_cache(maxsize=1)
def get_settings() -> CoreSettings:
    return CoreSettings()
