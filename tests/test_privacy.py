"""Tests for the privacy fixes and the opt-in switches. Synthetic text only, never real notes."""

import datetime as dt
import importlib
import json
import re
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, "src")

utils = importlib.import_module("utils")
main = importlib.import_module("main")
date_shift = importlib.import_module("date_shift")
redaction = importlib.import_module("redaction")

AGE, PHONE, ID, DOC = "[REDACTED_PATIENT_AGE]", "[REDACTED_PHONE]", "[REDACTED_PATIENT_ID]", "[REDACTED_DOCTOR_NAME]"


@pytest.fixture
def rambam(monkeypatch):
    """Turn every opt-in switch on, as config/rambam.yaml does."""
    for key, value in {"NAME_LISTS": True, "SAFE_TEXT_MATCHING": True, "REGEX_BACKSTOP": True,
                       "DATE_MODE": "visit_offset"}.items():
        monkeypatch.setitem(utils.config, key, value)


# --- always-on fixes ---------------------------------------------------------------------------

def test_schema_requires_every_key_for_both_backends(monkeypatch):
    schema = utils._response_schema()
    assert set(schema["required"]) == set(schema["properties"])
    monkeypatch.setitem(utils.config, "NAME_LISTS", True)
    assert utils._response_schema()["properties"]["doctor_name"]["type"] == "array"


def test_error_row_carries_no_original_text():
    assert utils.deidentify_text("secret note", {"raw_response": "{bad"}) == ("[ANONYMIZATION_ERROR]", ["anonymization_error"])


def test_ollama_prompt_that_fills_the_context_is_an_error(monkeypatch):
    monkeypatch.setitem(utils.config, "BACKEND", "ollama")
    monkeypatch.setitem(utils.config, "OLLAMA_CONFIG", {"MODEL": "fake", "OPTIONS": {"num_ctx": 100, "num_predict": 20}})

    class Client:
        def chat(self, **kwargs):
            count = 90 if "long" in kwargs["messages"][-1]["content"] else 10
            return SimpleNamespace(message=SimpleNamespace(content="{}"), prompt_eval_count=count, done_reason="stop")

    results = utils.extract_pii_from_text(["short", "long"], Client(), "prompt")
    assert results[0] == {}
    assert "raw_response" in results[1]


def test_vllm_skips_too_long_prompts_and_cut_answers(monkeypatch):
    monkeypatch.setitem(utils.config, "BACKEND", "vllm")
    monkeypatch.setitem(utils.config, "VLLM_CONFIG", {"MODEL_CONFIG": {"max_model_len": 210},
                                                      "GENERATION_CONFIG": {"max_tokens": 10}})

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            # Like transformers >= 5 by default: a dict, whose len() would be 2 (the bug this guards against).
            ids = [0] * len(conversation[-1]["content"])  # one token per character
            return ids if kwargs.get("return_dict") is False else {"input_ids": ids, "attention_mask": ids}

    class Model:
        sent = None

        def get_tokenizer(self):
            return Tokenizer()

        def chat(self, messages, sampling_params):
            Model.sent = len(messages)
            reasons = {"ok": "stop", "cut": "length"}
            return [SimpleNamespace(outputs=[SimpleNamespace(text="{}", finish_reason=reasons[m[-1]["content"].split("\n")[1]])])
                    for m in messages]

    results = utils.extract_pii_from_text(["ok", "x" * 300, "cut"], Model(), "p")
    assert Model.sent == 2  # the 300-character text was never sent
    assert results[0] == {}
    assert "raw_response" in results[1] and "raw_response" in results[2]


def test_age_whole_number_only():
    text = "בן 71. בן71, VA 0.71, 6/71, IOP 171, 71/80, date 3.4.71"
    assert utils.remove_pii_from_text(text, {"patient_age": 71}) == (
        f"בן {AGE}. בן{AGE}, VA 0.71, 6/71, IOP 171, 71/80, date 3.4.71")


def test_patient_id_keeps_plain_replacement():
    assert utils.remove_pii_from_text("ID 12345678917/01/22", {"patient_id": "123456789"}) == f"ID {ID}17/01/22"


def test_default_prompt_unchanged_and_name_lists_prompt_uses_lists(monkeypatch):
    assert '"doctor_name": "ד\\"ר פרידמן"' in utils.get_system_prompt()
    monkeypatch.setitem(utils.config, "NAME_LISTS", True)
    prompt = utils.get_system_prompt()
    assert '"doctor_name": ["ד\\"ר פרידמן"]' in prompt and '"patient_name": []' in prompt


# --- dates (DATE_MODE: visit_offset) ------------------------------------------------------------

def test_dates_become_offsets_from_the_visit(rambam):
    pii = {"intervention_dates": ["01/01/2024", "05/02/2024"]}
    text, _ = utils.deidentify_text("injection 01/01/2024, today 05/02/2024", pii, visit_date=dt.date(2024, 2, 5))
    assert text == "injection [DATE_DAY_-35], today [DATE_DAY_0]"


def test_merged_injection_serial_keeps_its_number(rambam):
    text, _ = utils.deidentify_text("זריקת אווסטין #114/02/22", {"intervention_dates": ["14/02/22"]}, visit_date=dt.date(2022, 3, 1))
    assert text == "זריקת אווסטין #1[DATE_DAY_-15]"


def test_visual_acuity_and_iop_are_not_dates(rambam):
    text = "VA 6/9, IOP 21/18, 0.5"
    assert utils.deidentify_text(text, {}, visit_date=dt.date(2024, 1, 1))[0] == text


def test_dates_the_llm_missed_are_caught_when_full(rambam):
    text, flags = utils.deidentify_text("seen 3.4.2023 and 3 באפריל 2023", {}, visit_date=dt.date(2023, 4, 10))
    assert text == "seen [DATE_DAY_-7] and [DATE_DAY_-7]"
    assert "backstop:date" in flags


@pytest.mark.parametrize("span, expected", [
    ("03/2022", "[DATE_MONTH_-2]"),
    ("2019", "[DATE_YEAR_-3]"),
    ("15/03", "[REDACTED_DATE]"),       # no year: cannot be placed exactly
    ("31/02/2022", "[REDACTED_DATE]"),  # invalid day
    ("1.1.95", "[DATE_DAY_-9991]"),    # two-digit year after the visit year -> 1995
])
def test_offset_tokens(span, expected):
    assert date_shift.to_offset_token(span, dt.date(2022, 5, 10)) == expected


def test_date_of_birth_is_removed_not_offset(rambam):
    text, _ = utils.deidentify_text("נולד 01/01/1930", {"patient_date_of_birth": "01/01/1930"}, visit_date=dt.date(2024, 3, 12))
    assert text == "נולד [REDACTED_DATE_OF_BIRTH]"


def test_without_visit_date_every_date_is_redacted(rambam):
    text, flags = utils.deidentify_text("on 1.2.20 and 3.4.2021", {"intervention_dates": ["1.2.20"]}, visit_date=None)
    assert text == "on [REDACTED_DATE] and [REDACTED_DATE]"
    assert "no_visit_date" in flags


# --- names (SAFE_TEXT_MATCHING) ------------------------------------------------------------------

def test_short_name_is_not_cut_out_of_words(rambam):
    assert utils.remove_pii_from_text("Dr Tal read Talmud", {"doctor_name": ["Tal"]}) == f"Dr {DOC} read Talmud"


def test_hebrew_prefix_is_kept_and_longer_words_are_spared(rambam):
    pii = {"other_mentioned_names": ["רון"]}
    assert utils.remove_pii_from_text("מרון ואהרון", pii) == "מ[REDACTED_OTHER_MENTIONED_NAMES] ואהרון"


def test_short_name_glued_to_a_word_still_falls_back(rambam):
    # Only match is inside a word (e.g. a lost line break): redact anyway rather than leak.
    assert utils.remove_pii_from_text("ד\"ר טלעין ימין", {"doctor_name": ["טל"]}) == f"ד\"ר {DOC}עין ימין"


def test_longest_name_first_and_never_inside_placeholders(rambam):
    pii = {"patient_name": ["כהן", "אבי כהן"], "patient_id": "123456789", "doctor_name": ["Red"]}
    text = utils.remove_pii_from_text("אבי כהן, Dr Red, ID 123456789", pii)
    assert text == f"[REDACTED_PATIENT_NAME], Dr {DOC}, ID {ID}"


def test_unmatched_value_is_flagged_by_field_name_only(rambam):
    _, flags = utils.deidentify_text("nothing here", {"doctor_name": ["Dr Secret"]}, visit_date=dt.date(2024, 1, 1))
    assert flags == ["unmatched:doctor_name"]


# --- regex backstop ------------------------------------------------------------------------------

def test_backstop_phones_emails_urls_ids_ages(rambam):
    text = "call 050-1234567 or 04-8541111, a.b@x.co.il, www.x.org, ID 123456782, not 123456789, בת 94, בתאריך 12"
    out, flags = utils.deidentify_text(text, {})
    assert out == (f"call {PHONE} or {PHONE}, [REDACTED_EMAIL], [REDACTED_URL], ID {ID}, not 123456789, "
                   f"בת {AGE}, בתאריך 12")
    assert {"backstop:phone", "backstop:email", "backstop:url", "backstop:id", "backstop:age"} <= set(flags)


# --- outputs -------------------------------------------------------------------------------------

def test_age_at_visit_is_top_coded():
    ages = utils.age_at_visit(pd.Series(["1930-01-01", "2000-06-15"]), pd.Series(["2024-03-12", "2024-06-14"]))
    assert ages.tolist() == [90, 23]


def test_shareable_file_holds_only_allowlisted_columns(monkeypatch, tmp_path):
    df = pd.DataFrame({"pid": ["a"], "visit": ["2024-01-01"], "birth": ["1930-01-01"], "age": [95],
                       "note": ["raw"], "note_deidentified": ["clean"], "secret": ["x"]})
    cfg = {"COLUMNS_TO_DEIDENTIFY": ["note"], "SHARE_COLUMNS": ["pid", "visit", "age", "note_deidentified"],
           "BIRTHDATE_COLUMN": "birth", "VISIT_DATE_COLUMN": "visit", "AGE_COLUMNS": ["age"]}
    utils.write_shareable(df, tmp_path / "out.csv", cfg)
    out = pd.read_csv(tmp_path / "out.csv")
    assert list(out.columns) == ["pid", "visit", "age", "note_deidentified", "age_at_visit"]
    assert out.loc[0, "age"] == 90 and out.loc[0, "age_at_visit"] == 90

    cfg["SHARE_COLUMNS"] = ["note"]
    with pytest.raises(ValueError, match="raw text"):
        utils.write_shareable(df, tmp_path / "out.csv", cfg)


def test_main_end_to_end_with_flags_extractions_and_reuse(monkeypatch, tmp_path, rambam):
    input_path = tmp_path / "notes.csv"
    pd.DataFrame({"pid": ["p1"], "visit": ["2024-02-05"], "note": ["ד\"ר לוי, 01/01/2024, 050-1234567"]}).to_csv(input_path, index=False)
    cfg = {"DATA_PATH": str(input_path), "COLUMNS_TO_DEIDENTIFY": ["note"], "MODEL_PATH": "/tmp/m",
           "BACKEND": "vllm", "VLLM_CONFIG": {},
           "VISIT_DATE_COLUMN": "visit", "FLAGS_COLUMN": True, "SAVE_EXTRACTIONS": True,
           "SHARE_COLUMNS": ["pid", "note_deidentified"]}
    monkeypatch.setattr(main, "config", cfg)
    monkeypatch.setattr(main, "load_model", lambda model_path: object())
    monkeypatch.setattr(main, "extract_pii_from_text",
                        lambda **kwargs: [{"doctor_name": ["ד\"ר לוי"], "intervention_dates": ["01/01/2024"]}])

    monkeypatch.setattr(main, "load_config", lambda: None)
    main.main()
    expected = f"{DOC}, [DATE_DAY_-35], {PHONE}"
    final = pd.read_csv(tmp_path / "notes_deidentified_final.csv")
    assert final.loc[0, "note_deidentified"] == expected
    assert final.loc[0, "note_deid_flags"] == "backstop:phone"
    assert list(pd.read_csv(tmp_path / "notes_shareable.csv").columns) == ["pid", "note_deidentified"]

    # Re-applying from the saved extractions must not need the model and must give the same text.
    monkeypatch.setattr(main, "load_model", lambda model_path: pytest.fail("model loaded"))
    (tmp_path / "notes_deidentified_final.csv").unlink()
    cfg["REUSE_EXTRACTIONS"] = str(tmp_path / "notes_extractions.jsonl")
    main.main()
    assert pd.read_csv(tmp_path / "notes_deidentified_final.csv").loc[0, "note_deidentified"] == expected
    saved = [json.loads(line) for line in (tmp_path / "notes_extractions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved[0]["column"] == "note" and saved[0]["row"] == 0


def test_refraction_ranges_and_at_shorthand_are_untouched(rambam):
    # Found on real notes: mixed separators and "@" in measurements were taken for dates and emails.
    text = "ref +1.25-0.75*90, range 12-10.50, VA 0.5@12.3"
    assert utils.deidentify_text(text, {}, visit_date=dt.date(2024, 1, 1))[0] == text


# --- findings of the code review (2026-11-04) -----------------------------------------------------

def test_day_first_visit_dates_and_unreadable_ones_fail_loudly():
    df = pd.DataFrame({"visit": ["05/02/2024", "20/02/2024", None, "2024-02-05", "2024-02-05 10:30:00"]})
    assert utils.date_column(df, "visit").tolist() == [
        dt.date(2024, 2, 5), dt.date(2024, 2, 20), None, dt.date(2024, 2, 5), dt.date(2024, 2, 5)]
    with pytest.raises(ValueError, match="could not be read"):
        utils.date_column(pd.DataFrame({"visit": ["not a date"]}), "visit")


def test_reused_extraction_is_ignored_when_the_text_changed():
    saved = {("note", "0"): (utils.text_hash("Dr Alpha saw"), {"doctor_name": ["Dr Alpha"]})}
    assert utils.reused_extraction(saved, "note", 0, "Dr Alpha saw") == {"doctor_name": ["Dr Alpha"]}
    assert utils.reused_extraction(saved, "note", 0, "Dr Beta saw") == {"raw_response": ""}


@pytest.mark.parametrize("text, pii", [
    ("נולד 01/01/1930", {}),                                # DOB the model missed
    ("נולד 01/01/1930", {"other_dates": ["01/01/1930"]}),   # DOB labelled as another date
])
def test_missed_date_of_birth_never_becomes_an_age(rambam, text, pii):
    out, _ = utils.deidentify_text(text, pii, visit_date=dt.date(2024, 3, 12))
    assert out == "נולד [REDACTED_DATE]"  # more than 89 years before the visit
    out, _ = utils.deidentify_text("נולד 05/06/1960", {}, visit_date=dt.date(2024, 3, 12), birthdate=dt.date(1960, 6, 5))
    assert out == "נולד [REDACTED_DATE_OF_BIRTH]"  # equal to the known birthdate


def test_follow_up_date_with_two_digit_year_stays_in_the_future():
    assert date_shift.to_offset_token("15/01/25", dt.date(2024, 12, 10)) == "[DATE_DAY_36]"


def test_partial_date_span_does_not_cut_into_other_numbers(rambam):
    out, _ = utils.deidentify_text("IOP 12/30, seen 05/06/2019", {"other_dates": ["2/3", "2019"]}, visit_date=dt.date(2024, 6, 5))
    assert out.startswith("IOP 12/30, seen [DATE_DAY_")


def test_iso_dates_are_caught(rambam):
    assert utils.deidentify_text("seen 2024-02-01", {}, visit_date=dt.date(2024, 2, 5))[0] == "seen [DATE_DAY_-4]"


def test_bracketed_text_in_the_note_is_not_protected(rambam):
    assert utils.remove_pii_from_text("seen by [SMITH]", {"doctor_name": ["SMITH"]}) == f"seen by [{DOC}]"


def test_birthdate_after_visit_gives_no_age():
    assert utils.age_at_visit(pd.Series(["2030-01-01"]), pd.Series(["2024-01-01"])).isna().all()


def test_audit_counts_error_rows_and_no_false_doctor_alarm(tmp_path, capsys):
    sys.path.insert(0, "scripts")
    audit = importlib.import_module("audit_output")
    path = tmp_path / "out.csv"
    pd.DataFrame({"n_deidentified": ["[ANONYMIZATION_ERROR]", "[ANONYMIZATION_ERROR]",
                                     f'ד"ר {DOC} בדק']}).to_csv(path, index=False)
    audit.audit(str(path))
    import re
    lines = dict(m.groups() for m in re.finditer(r"^\s+(.+?)\s+(\d+) cells$", capsys.readouterr().out, re.M))
    assert lines["error rows"] == "2" and lines["doctor title + word"] == "0"


def test_residual_counts_failed_rows_and_whole_number_ages():
    evaluate = importlib.import_module("evaluate")
    metrics, _ = evaluate.compute_residual(
        ["[ANONYMIZATION_ERROR]", "VA 0.71"], [{"patient_name": "Avi"}, {"patient_age": 71}], ["patient_name", "patient_age"])
    assert metrics["per_category"] == {"patient_name": {"hits": 1, "gt": 1}, "patient_age": {"hits": 0, "gt": 1}}


@pytest.mark.parametrize("override, message", [
    ({"COLUMNS_TO_DEIDENTIFY": ["FILL_ME"]}, "Fill in"),
    ({"VISIT_DATE_COLUMN": "FILL_ME"}, "Fill in"),
    ({"COLUMNS_TO_DEIDENTIFY": ["missing_col"]}, "not found in the data"),
    ({"VISIT_DATE_COLUMN": "missing_col"}, "not found in the data"),
    ({"SHARE_COLUMNS": ["note_raw_typo"]}, "SHARE_COLUMNS not found"),
    ({"OLLAMA_CONFIG": None, "BACKEND": "vllm"}, "misses these keys"),
])
def test_config_errors_stop_before_the_model_loads(override, message):
    df = pd.DataFrame({"pid": ["a"], "visit": ["2024-01-01"], "note": ["text"]})
    cfg = {"DATA_PATH": "notes.csv", "BACKEND": "ollama", "OLLAMA_CONFIG": {},
           "COLUMNS_TO_DEIDENTIFY": ["note"], "DATE_MODE": "visit_offset", "VISIT_DATE_COLUMN": "visit",
           "BIRTHDATE_COLUMN": None, "SHARE_COLUMNS": ["pid", "note_deidentified", "note_deid_flags"]}
    utils.validate_config(cfg, df)  # the valid config passes
    with pytest.raises(ValueError, match=message):
        utils.validate_config({**cfg, **override}, df)


@pytest.mark.parametrize("hospital", ["rambam", "hy", "hy_injections"])
def test_shipped_configs_load_and_refuse_to_run_until_filled(monkeypatch, tmp_path, hospital):
    # The real config files, with HOSPITAL_CONFIG switched: a key in both files would fail here.
    config_module = importlib.import_module("config")
    real = config_module.CONFIG_DIR
    if not (real / f"{hospital}.yaml").exists():
        pytest.skip(f"config/{hospital}.yaml is local only")
    shared = (real / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(re.sub(r"^HOSPITAL_CONFIG:.*$", f"HOSPITAL_CONFIG: {hospital}", shared, flags=re.M),
                                          encoding="utf-8")
    (tmp_path / f"{hospital}.yaml").write_text((real / f"{hospital}.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "config", {})

    cfg = config_module.load_config()
    assert cfg["BACKEND"] == "ollama" and "OLLAMA_CONFIG" in cfg and "VLLM_CONFIG" in cfg
    assert cfg["REUSE_EXTRACTIONS"] is None  # a shipped config must always run the model
    with pytest.raises(ValueError, match="Fill in"):
        utils.validate_config(cfg, pd.DataFrame({"x": ["text"]}))
