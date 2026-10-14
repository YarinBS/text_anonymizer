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

def get_config():
    """
    Loads a YAML configuration file and returns it as a dictionary
    """

    with (PROJECT_ROOT / "config" / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = Path(os.path.expandvars(config["DATA_PATH"]))
    config["DATA_PATH"] = str(data_path if data_path.is_absolute() else PROJECT_ROOT / data_path)

    # Only vLLM needs MODEL_PATH (checked in utils.load_model); Ollama names its model in OLLAMA_CONFIG.
    model_path = os.path.expandvars(config.get("MODEL_PATH", ""))
    config["MODEL_PATH"] = "" if model_path == "${MODEL_PATH}" else model_path

    return config

config = get_config()
