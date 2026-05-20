import logging
import logging.config
from typing import Any

import yaml

from core_examples.config.settings import get_settings

_LOGGING_CONFIGURED = False


def configure_logging(default_level: str = "INFO") -> logging.Logger:
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return logging.getLogger(__name__)

    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()
    settings = get_settings()
    log_level = (settings.logging.level or default_level).upper()
    log_to_file = settings.logging.to_file

    try:
        logging.config.dictConfig(_build_logging_config(log_level, log_to_file))
    except (OSError, ValueError, yaml.YAMLError):
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.getLogger(__name__).exception("Failed to load logging configuration from YAML.")

    _LOGGING_CONFIGURED = True
    return logging.getLogger(__name__)


def _build_logging_config(log_level: str, log_to_file: bool) -> dict[str, Any]:
    settings = get_settings()
    config_logging_file_path = settings.config_logging_file_path
    default_log_file_path = settings.default_log_file_path

    with config_logging_file_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    root_config = config.setdefault("root", {})
    root_config["level"] = log_level
    root_config["handlers"] = ["console"]

    if log_to_file:
        default_log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers = config.setdefault("handlers", {})
        file_handler = handlers.setdefault(
            "file",
            {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "filename": str(default_log_file_path),
            },
        )
        file_handler.setdefault("class", "logging.FileHandler")
        file_handler.setdefault("formatter", "standard")
        file_handler["filename"] = str(default_log_file_path)
        root_config["handlers"].append("file")

    return config