from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_core_package_path() -> Path:
    return Path(__file__).resolve().parent.parent


class CoreSettings(BaseSettings):
    """Repository settings for the `core_examples` reference layer.

    These settings centralize paths and configuration file locations while
    keeping environment overrides available for repository runtimes.
    """

    model_config = SettingsConfigDict(
        env_prefix="FRANKENST_",
        extra="ignore",
    )

    core_package_path: Path = Field(default_factory=_default_core_package_path)
    src_directory_path: Path | None = None
    project_root_path: Path | None = None
    logs_directory_path: Path | None = None
    default_log_file_path: Path | None = None
    artifacts_directory_path: Path | None = None
    config_directory_path: Path | None = None
    config_file_path: Path | None = None
    config_logging_file_path: Path | None = None
    config_nodes_file_path: Path | None = None

    @model_validator(mode="after")
    def _populate_derived_paths(self) -> "CoreSettings":
        if self.src_directory_path is None:
            self.src_directory_path = self.core_package_path.parent

        if self.project_root_path is None:
            self.project_root_path = self.src_directory_path.parent

        if self.logs_directory_path is None:
            self.logs_directory_path = self.project_root_path / "logs"

        if self.default_log_file_path is None:
            self.default_log_file_path = self.logs_directory_path / "application.log"

        if self.artifacts_directory_path is None:
            self.artifacts_directory_path = self.project_root_path / "artifacts"

        if self.config_directory_path is None:
            self.config_directory_path = self.core_package_path / "config"

        if self.config_file_path is None:
            self.config_file_path = self.config_directory_path / "config.yml"

        if self.config_logging_file_path is None:
            self.config_logging_file_path = self.config_directory_path / "config_logging.yml"

        if self.config_nodes_file_path is None:
            self.config_nodes_file_path = self.config_directory_path / "config_nodes.yml"

        return self


@lru_cache(maxsize=1)
def get_settings() -> CoreSettings:
    """Return cached repository settings for the reference layer."""

    return CoreSettings()