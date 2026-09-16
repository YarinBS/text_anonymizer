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

    model_path = os.path.expandvars(config.get("MODEL_PATH", ""))
    if not model_path or model_path == "${MODEL_PATH}":
        raise RuntimeError("MODEL_PATH must be set in the environment or .env")
    config["MODEL_PATH"] = model_path

    return config

config = get_config()
