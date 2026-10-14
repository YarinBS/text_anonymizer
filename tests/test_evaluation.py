import importlib
import json
import sys
from types import SimpleNamespace

import ollama


sys.path.insert(0, "src")

utils = importlib.import_module("utils")
evaluate = importlib.import_module("evaluate")


class FakeOllamaClient:
    """Answers with a fixed reply per text, so tests need no Ollama server (synthetic data only)."""

    def __init__(self, replies):
        self.replies = replies

    def chat(self, model, messages, format, options, think):
        # All keys must be required, or the grammar allows answers that stop after one key.
        assert set(format["required"]) == set(format["properties"])
        text = messages[-1]["content"].split("\n", 1)[1]
        reply = self.replies[text]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(message=SimpleNamespace(content=reply))


def test_ollama_extraction_keeps_order_and_marks_failures(monkeypatch):
    monkeypatch.setitem(utils.config, "BACKEND", "ollama")
    monkeypatch.setitem(utils.config, "OLLAMA_CONFIG", {"MODEL": "fake", "NUM_PARALLEL": 3, "OPTIONS": {}})
    client = FakeOllamaClient({
        "a": json.dumps({"doctor_name": "Dr A"}),
        "b": "not json",
        "c": ollama.ResponseError("server error"),
        "d": json.dumps({"doctor_name": "Dr D"}),
    })

    results = utils.extract_pii_from_text(["a", "b", "c", "d"], client, "prompt")

    assert results[0] == {"doctor_name": "Dr A"}
    assert results[1] == {"raw_response": "not json"}
    assert "raw_response" in results[2]
    assert results[3] == {"doctor_name": "Dr D"}


def test_remove_pii_keeps_dates_by_default_and_redacts_them_when_asked():
    pii = {"doctor_name": "Dr Levi", "intervention_dates": ["1.2.20"]}
    text = "Dr Levi injected on 1.2.20"

    assert utils.remove_pii_from_text(text, pii) == "[REDACTED_DOCTOR_NAME] injected on 1.2.20"
    assert utils.remove_pii_from_text(text, pii, skip_fields=()) == (
        "[REDACTED_DOCTOR_NAME] injected on [REDACTED_INTERVENTION_DATES]"
    )


def test_recall_counts_per_category_and_redacted_fields():
    samples = [
        {"text": "x", "injections": [{"category": "doctor_name", "value": "Dr A"},
                                     {"category": "intervention_dates", "value": "1.2.20"}]},
        {"text": "y", "injections": []},
        {"text": "z", "injections": [{"category": "patient_age", "value": "70"}]},
    ]
    extracted = [{"doctor_name": "Dr A", "intervention_dates": []}, {}, {"raw_response": "bad"}]

    metrics, details = evaluate.compute_recall(samples, extracted)

    assert metrics["overall"] == {"recall": 1 / 3, "detected": 1, "total": 3}
    assert metrics["redacted_fields_only"] == {"recall": 0.5, "detected": 1, "total": 2}
    assert [d["sample_idx"] for d in details] == [0, 2]


def test_residual_and_retention_on_known_example():
    gt = {"doctor_name": "Dr A", "intervention_dates": ["1.2.20"], "patient_age": None}
    original = "Dr A gave avastin on 1.2.20 today"
    anonymized = "[REDACTED_DOCTOR_NAME] gave avastin on 1.2.20"  # date missed, last word lost
    tokenizer = lambda text, add_special_tokens: {"input_ids": text.split()}

    residual, hits = evaluate.compute_residual([anonymized], [gt], evaluate.PII_FIELDS)
    retention, per_sample = evaluate.compute_retention([original], [anonymized], [gt], tokenizer)

    assert residual["residual_hits"] == 1 and residual["gt_pii"] == 2
    assert hits == [[{"category": "intervention_dates", "value": "1.2.20"}]]
    # Context without GT PII: "gave avastin on today" (4) vs "gave avastin on" (3)
    assert per_sample[0]["orig_tokens"] == 4 and per_sample[0]["anon_tokens"] == 3
    assert retention["mean_retention"] == 0.75
