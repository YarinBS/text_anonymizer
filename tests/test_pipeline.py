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
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "input_deidentified_final.csv"
    pd.DataFrame({"text": ["first", "second", "third"]}).to_csv(input_path, index=False)

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["text"],
        "MODEL_PATH": "/tmp/test-model",
    })
    monkeypatch.setattr(main, "load_model", lambda model_path: object())
    monkeypatch.setattr(main, "get_system_prompt", lambda: "prompt")
    monkeypatch.setattr(main, "extract_pii_from_text", lambda **kwargs: [{}, {}])

    with pytest.raises(RuntimeError, match="returned 2 results for 3"):
        main.main()

    assert not output_path.exists()


def test_main_preserves_multiline_fields_and_original_columns(monkeypatch, tmp_path):
    main = importlib.import_module("main")
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "input_deidentified_final.csv"
    pd.DataFrame({"id": [1], "text": ["first\nsecond"]}).to_csv(input_path, index=False)

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["text"],
        "MODEL_PATH": "/tmp/test-model",
    })
    monkeypatch.setattr(main, "load_model", lambda model_path: object())
    monkeypatch.setattr(main, "get_system_prompt", lambda: "prompt")
    monkeypatch.setattr(main, "extract_pii_from_text", lambda **kwargs: [{}])
    monkeypatch.setattr(main, "remove_pii_from_text", lambda text, extracted: f"redacted: {text}")

    main.main()
    result = pd.read_csv(output_path)

    assert len(result) == 1
    assert result.loc[0, "text"] == "first\nsecond"
    assert result.loc[0, "text_deidentified"] == "redacted: first\nsecond"
    assert result.loc[0, "id"] == 1


def test_main_converts_txt_input_to_csv(monkeypatch, tmp_path):
    main = importlib.import_module("main")
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "input_deidentified_final.csv"
    input_path.write_text("first note\nsecond note\n", encoding="utf-8")

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["note"],
        "MODEL_PATH": "/tmp/test-model",
    })
    monkeypatch.setattr(main, "load_model", lambda model_path: object())
    monkeypatch.setattr(main, "get_system_prompt", lambda: "prompt")
    monkeypatch.setattr(main, "extract_pii_from_text", lambda **kwargs: [{}, {}])
    monkeypatch.setattr(main, "remove_pii_from_text", lambda text, extracted: f"redacted: {text}")

    main.main()
    result = pd.read_csv(output_path)

    assert list(result.columns) == ["note", "note_deidentified"]
    assert result["note"].tolist() == ["first note", "second note"]


def test_txt_input_rejects_multiple_configured_columns(monkeypatch, tmp_path):
    main = importlib.import_module("main")
    input_path = tmp_path / "input.txt"
    input_path.write_text("note", encoding="utf-8")

    monkeypatch.setattr(main, "config", {
        "DATA_PATH": str(input_path),
        "COLUMNS_TO_DEIDENTIFY": ["first", "second"],
        "MODEL_PATH": "/tmp/test-model",
    })

    with pytest.raises(ValueError, match="exactly one configured column"):
        main.main()


def test_config_paths_are_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", "/tmp/test-model")
    config_module = importlib.import_module("config")

    import os
    original_directory = os.getcwd()
    try:
        os.chdir(tmp_path)
        loaded = config_module.get_config()
    finally:
        os.chdir(original_directory)

    assert loaded["DATA_PATH"].endswith("data/only_injections_freetext.csv")
    assert loaded["MODEL_PATH"] == "/tmp/test-model"


def test_setup_scripts_use_the_single_requirements_file():
    for path in ("scripts/setup.sh", "scripts/install_cuda_env.sh"):
        with open(path, encoding="utf-8") as file:
            contents = file.read()
        assert "requirements.txt" in contents
        assert "requirements_cpu.txt" not in contents
        assert "requirements_gpu.txt" not in contents
