import importlib
import sys

import pandas as pd
import pytest


sys.path.insert(0, "src")


def load_utils(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/tmp/test-model")
    return importlib.import_module("utils")


def test_valid_text_records_preserve_original_indices(monkeypatch):
    utils = load_utils(monkeypatch)
    frame = pd.DataFrame({"text": ["first", 123, None, "second"]})

    assert utils.get_valid_text_records(frame, "text") == [
        (0, "first"),
        (3, "second"),
    ]


def test_main_fails_before_writing_when_model_result_count_differs(monkeypatch, tmp_path):
    main = importlib.import_module("main")
    monkeypatch.setattr(main, "load_config", lambda: None)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "input_deidentified_final.csv"
    pd.DataFrame({"text": ["first", "second", "third"]}).to_csv(input_path, index=False)

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["text"],
        "MODEL_PATH": "/tmp/test-model",
        "BACKEND": "vllm",
        "VLLM_CONFIG": {},
    })
    monkeypatch.setattr(main, "load_model", lambda model_path: object())
    monkeypatch.setattr(main, "get_system_prompt", lambda: "prompt")
    monkeypatch.setattr(main, "extract_pii_from_text", lambda **kwargs: [{}, {}])

    with pytest.raises(RuntimeError, match="returned 2 results for 3"):
        main.main()

    assert not output_path.exists()


def test_main_preserves_multiline_fields_and_original_columns(monkeypatch, tmp_path):
    main = importlib.import_module("main")
    monkeypatch.setattr(main, "load_config", lambda: None)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "input_deidentified_final.csv"
    pd.DataFrame({"id": [1], "text": ["first\nsecond"]}).to_csv(input_path, index=False)

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["text"],
        "MODEL_PATH": "/tmp/test-model",
        "BACKEND": "vllm",
        "VLLM_CONFIG": {},
    })
    monkeypatch.setattr(main, "load_model", lambda model_path: object())
    monkeypatch.setattr(main, "get_system_prompt", lambda: "prompt")
    monkeypatch.setattr(main, "extract_pii_from_text", lambda **kwargs: [{}])
    monkeypatch.setattr(main, "deidentify_text", lambda text, extracted, **kwargs: (f"redacted: {text}", []))

    main.main()
    result = pd.read_csv(output_path)

    assert len(result) == 1
    assert result.loc[0, "text"] == "first\nsecond"
    assert result.loc[0, "text_deidentified"] == "redacted: first\nsecond"
    assert result.loc[0, "id"] == 1


def test_main_converts_txt_input_to_csv(monkeypatch, tmp_path):
    main = importlib.import_module("main")
    monkeypatch.setattr(main, "load_config", lambda: None)
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "input_deidentified_final.csv"
    input_path.write_text("first note\nsecond note\n", encoding="utf-8")

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["note"],
        "MODEL_PATH": "/tmp/test-model",
        "BACKEND": "vllm",
        "VLLM_CONFIG": {},
    })
    monkeypatch.setattr(main, "load_model", lambda model_path: object())
    monkeypatch.setattr(main, "get_system_prompt", lambda: "prompt")
    monkeypatch.setattr(main, "extract_pii_from_text", lambda **kwargs: [{}, {}])
    monkeypatch.setattr(main, "deidentify_text", lambda text, extracted, **kwargs: (f"redacted: {text}", []))

    main.main()
    result = pd.read_csv(output_path)

    assert list(result.columns) == ["note", "note_deidentified"]
    assert result["note"].tolist() == ["first note", "second note"]


def test_txt_input_rejects_multiple_configured_columns(monkeypatch, tmp_path):
    main = importlib.import_module("main")
    monkeypatch.setattr(main, "load_config", lambda: None)
    input_path = tmp_path / "input.txt"
    input_path.write_text("note", encoding="utf-8")

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["first", "second"],
        "MODEL_PATH": "/tmp/test-model",
        "BACKEND": "vllm",
        "VLLM_CONFIG": {},
    })

    with pytest.raises(ValueError, match="exactly one configured column"):
        main.main()


def write_configs(folder, shared: str, hospitals: dict):
    """A config folder of its own: config.yaml plus one <name>.yaml per hospital."""
    (folder / "config.yaml").write_text(shared, encoding="utf-8")
    for name, text in hospitals.items():
        (folder / f"{name}.yaml").write_text(text, encoding="utf-8")


@pytest.fixture
def config_module(monkeypatch, tmp_path):
    """config.py reading from tmp_path, with its own `config` dict so no test leaks settings into another."""
    module = importlib.import_module("config")
    monkeypatch.setattr(module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(module, "config", {})
    return module


def test_hospital_config_is_added_to_the_shared_one(monkeypatch, tmp_path, config_module):
    monkeypatch.setenv("MODEL_PATH", "/tmp/test-model")
    write_configs(tmp_path, "HOSPITAL_CONFIG: a\nMODEL_PATH: ${MODEL_PATH}\n",
                  {"a": "DATA_PATH: data/notes.csv\nREGEX_BACKSTOP: true\n", "b": "REGEX_BACKSTOP: false\n"})

    import os
    original_directory = os.getcwd()
    try:
        os.chdir(tmp_path)  # relative paths start at the repo root, not at the working directory
        loaded = config_module.load_config()
    finally:
        os.chdir(original_directory)

    assert loaded is config_module.config  # filled in place: other modules hold this dict
    assert loaded == {"HOSPITAL_CONFIG": "a", "MODEL_PATH": "/tmp/test-model", "REGEX_BACKSTOP": True,
                      "DATA_PATH": str(config_module.PROJECT_ROOT / "data" / "notes.csv")}


def test_key_in_both_files_stops_the_run(tmp_path, config_module):
    # Otherwise a switch turned on in config.yaml would be silently undone by the hospital file.
    write_configs(tmp_path, "HOSPITAL_CONFIG: a\nREGEX_BACKSTOP: true\n", {"a": "REGEX_BACKSTOP: false\n"})
    with pytest.raises(ValueError, match="REGEX_BACKSTOP"):
        config_module.load_config()


def test_unknown_or_missing_hospital_stops_the_run(tmp_path, config_module):
    write_configs(tmp_path, "HOSPITAL_CONFIG: absent\n", {"a": "DATA_PATH: x.csv\n"})
    with pytest.raises(FileNotFoundError, match="available: \\['a'\\]"):
        config_module.load_config()
    write_configs(tmp_path, "MODEL_PATH: x\n", {})
    with pytest.raises(ValueError, match="HOSPITAL_CONFIG"):
        config_module.load_config()
