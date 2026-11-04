"""
config.py

This module provides a utility for loading configuration settings from a YAML file.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def get_config(path=None):
    """
    Loads config/config.yaml and returns it as a dictionary. A `path` (main.py/evaluate.py --config) holds
    only the keys it changes, applied on top, so the backend settings are written once.
    """

    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        with Path(path).open(encoding="utf-8") as f:
            for key, value in (yaml.safe_load(f) or {}).items():
                # One level deep, so an override can change e.g. only OLLAMA_CONFIG.MODEL.
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value

    data_path = Path(os.path.expandvars(config["DATA_PATH"]))
    if config["DATA_PATH"] != "FILL_ME":  # left as is, so utils.validate_config can report it
        config["DATA_PATH"] = str(data_path if data_path.is_absolute() else PROJECT_ROOT / data_path)

    # Only vLLM needs MODEL_PATH (checked in utils.load_model); Ollama names its model in OLLAMA_CONFIG.
    model_path = os.path.expandvars(config.get("MODEL_PATH", ""))
    config["MODEL_PATH"] = "" if model_path == "${MODEL_PATH}" else model_path

    return config


def load_config(path) -> None:
    """
    Switch every module to another config file (main.py --config).

    Modules hold this same dict (`from config import config`), so it is updated in place
    instead of being replaced.
    """
    config.clear()
    config.update(get_config(path))


config = get_config()
