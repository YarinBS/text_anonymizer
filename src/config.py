"""
config.py

Loads the run's settings: config/config.yaml (settings shared by every hospital, and HOSPITAL_CONFIG)
plus config/<HOSPITAL_CONFIG>.yaml (that hospital's data, columns, privacy switches and backend).
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"

# Shared by every module (`from config import config`). Empty until load_config() runs, so importing
# never needs a config file (tests, or a fresh clone whose FILL_MEs are not filled in yet).
config = {}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config() -> dict:
    """
    Read config.yaml, then the hospital file it names in HOSPITAL_CONFIG, and fill `config` with both.

    A key may be in only one of the two files. If "hospital overrides global" were allowed, a privacy switch
    turned on in config.yaml would be silently undone by an older value in a hospital file.
    `config` is updated in place, not replaced, because other modules hold this same dict.
    """
    shared = _read_yaml(CONFIG_DIR / "config.yaml")
    hospital_name = shared.get("HOSPITAL_CONFIG")
    if not hospital_name:
        raise ValueError("config/config.yaml must set HOSPITAL_CONFIG, e.g. HOSPITAL_CONFIG: rambam")
    hospital_path = CONFIG_DIR / f"{hospital_name}.yaml"
    if not hospital_path.exists():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != "config")
        raise FileNotFoundError(f"HOSPITAL_CONFIG: {hospital_name} has no file {hospital_path}; available: {available}")
    hospital = _read_yaml(hospital_path)

    overlap = sorted(set(shared) & set(hospital))
    if overlap:
        raise ValueError(f"Keys set in both config.yaml and {hospital_path.name} (keep each in one file): {overlap}")
    merged = {**shared, **hospital}

    # Relative paths start at the repo root, so a run does not depend on the folder it is started from.
    for key in ("DATA_PATH", "REUSE_EXTRACTIONS"):
        if merged.get(key) and merged[key] != "FILL_ME":  # FILL_ME is reported by validate_config
            path = Path(os.path.expandvars(merged[key]))
            merged[key] = str(path if path.is_absolute() else PROJECT_ROOT / path)

    # Only vLLM needs MODEL_PATH (checked in utils.load_model); Ollama names its model in OLLAMA_CONFIG.
    model_path = os.path.expandvars(merged.get("MODEL_PATH", ""))
    merged["MODEL_PATH"] = "" if model_path == "${MODEL_PATH}" else model_path

    config.clear()
    config.update(merged)
    return config
