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
    Load one YAML config file and return it as a dictionary. Every config is complete on its own.

    Without `path`, config/config.yaml is used if it exists; otherwise the config is empty and main.py
    asks for --config. The import must never need a file: on a fresh clone without config.yaml it would
    otherwise crash before --config is even read.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if config.get("DATA_PATH") and config["DATA_PATH"] != "FILL_ME":  # FILL_ME is reported by validate_config
        data_path = Path(os.path.expandvars(config["DATA_PATH"]))
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
