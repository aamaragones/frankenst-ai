import logging
import logging.config
import os
from typing import Any

import yaml

from core_examples.constants import CONFIG_LOGGING_FILE_PATH, DEFAULT_LOG_FILE_PATH

_LOGGING_CONFIGURED = False


def configure_logging(default_level: str = "INFO") -> logging.Logger:
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return logging.getLogger(__name__)

    log_level = os.getenv("LOG_LEVEL", default_level).upper()
    log_to_file = _read_env_flag("LOG_TO_FILE")

    try:
        logging.config.dictConfig(_build_logging_config(log_level, log_to_file))
    except (OSError, ValueError, yaml.YAMLError):
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.getLogger(__name__).exception("Failed to load backend logging configuration from YAML.")

    _LOGGING_CONFIGURED = True
    return logging.getLogger(__name__)


def _build_logging_config(log_level: str, log_to_file: bool) -> dict[str, Any]:
    with CONFIG_LOGGING_FILE_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    root_config = config.setdefault("root", {})
    root_config["level"] = log_level
    root_config["handlers"] = ["console"]

    if log_to_file:
        DEFAULT_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        handlers = config.setdefault("handlers", {})
        file_handler = handlers.setdefault("file", {})
        file_handler["filename"] = str(DEFAULT_LOG_FILE_PATH)
        root_config["handlers"].append("file")

    return config


def _read_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}