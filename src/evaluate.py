"""
evaluate.py

Evaluates one LLM on the anonymization pipeline and writes a report (JSON + TXT) with the
three metrics of the original experiments and every example used to compute them:

1. Synthetic PII Injection Recall - PII with known values is injected into real notes
   (a frozen evaluation set, so every model sees the same inputs); recall = found / injected.
2. Residual PII Rate - on the manually annotated notes (data/tagged_samples.json), the share of
   ground-truth PII values still present in the de-identified text.
3. Token Retention Ratio - on the same annotated notes, tokens kept after de-identification
   divided by original tokens, both counted after removing the ground-truth PII.

Inference goes through the pipeline functions in utils.py with the Ollama backend, so the scores
are those of the code that runs on the target machine.

Usage:
    # once: freeze the evaluation set (from the report behind the presentation results)
    python src/evaluate.py build-eval-set --from-report results/evaluation_20260228_170735.json
    # per model (an Ollama tag)
    python src/evaluate.py run --model gemma4-12b-q8 --host http://127.0.0.1:11500
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, config
from eval_injection import InjectedPII, _check_detection, inject_pii_into_existing_text
from utils import (DATE_FIELDS, MedicalInformation, extract_pii_from_text,
                   get_system_prompt, load_model, remove_pii_from_text)


DEFAULT_EVAL_SET = PROJECT_ROOT / "data" / "eval_sets" / "injection_feb2026_n500.json"
DEFAULT_ANNOTATED = PROJECT_ROOT / "data" / "tagged_samples.json"
# One tokenizer for every model, so the retention ratio does not depend on the evaluated model's tokenizer.
# A local Hugging Face tokenizer folder (we use Llama-3.3's), set in .env or with --tokenizer.
DEFAULT_TOKENIZER = os.environ.get("TOKENIZER_PATH")
PII_FIELDS = list(MedicalInformation.model_json_schema()["properties"].keys())
REDACTED_FIELDS = [field for field in PII_FIELDS if field not in DATE_FIELDS]


# ---------------------------------------------------------------------------
# Evaluation set
# ---------------------------------------------------------------------------

def build_eval_set(output_path: Path, from_report: str = None, seed: int = 42, n_samples: int = 500) -> None:
    """
    Freeze the injected samples to a file, so all models (and machines) are scored on the same inputs.

    With --from-report, the samples are taken from an old evaluation report: its random draw was
    made by code that no longer exists, so this is the only way to reuse the presentation's inputs.
    Otherwise new samples are drawn with the sampling of the old harness (seed, then
    random.sample of n + 250 texts, the first n are injected).
    """
    if from_report:
        details = json.loads(Path(from_report).read_text(encoding="utf-8"))["synthetic_recall"]["sample_details"]
        samples = [
            {"text": d["generated_text"], "injections": [{"category": i["category"], "value": i["value"]} for i in d["injections"]]}
            for d in details
        ]
        source = {"from_report": str(from_report)}
    else:
        column = config["COLUMNS_TO_DEIDENTIFY"][0]
        data = pd.read_csv(config["DATA_PATH"])
        texts = data[data[column].notnull()][column].tolist()
        random.seed(seed)
        # The old harness drew n + 250 texts and used the first n; min() lets small local CSVs work too
        # (for files with >= n + 250 texts the draw is unchanged).
        base_texts = random.sample(texts, min(n_samples + 250, len(texts)))[:n_samples]
        samples = []
        for text in base_texts:
            augmented, injections = inject_pii_into_existing_text(text)
            samples.append({"text": augmented, "injections": [{"category": i.category, "value": i.value} for i in injections]})
        source = {"data_path": config["DATA_PATH"], "seed": seed}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"source": source, "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_injected = sum(len(s["injections"]) for s in samples)
    print(f"Saved {len(samples)} samples ({n_injected} injected PII) to {output_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_recall(samples: list[dict], extracted: list[dict]) -> tuple[dict, list[dict]]:
    """Metric 1. Samples without injected PII are skipped, as in the original harness."""
    totals, found, details = {}, {}, []
    for index, (sample, pii) in enumerate(zip(samples, extracted)):
        if not sample["injections"]:
            continue
        injections = [InjectedPII(**i) for i in sample["injections"]]
        _check_detection(injections, pii)
        for inj in injections:
            totals[inj.category] = totals.get(inj.category, 0) + 1
            found[inj.category] = found.get(inj.category, 0) + int(inj.detected)
        details.append({
            "sample_idx": index,
            "text": sample["text"],
            "injections": [{"category": i.category, "value": i.value, "detected": i.detected} for i in injections],
            "extracted_pii": pii,
        })

    def _ratio(categories):
        total = sum(totals.get(c, 0) for c in categories)
        return {"recall": sum(found.get(c, 0) for c in categories) / total if total else None,
                "detected": sum(found.get(c, 0) for c in categories), "total": total}

    metrics = {
        "overall": _ratio(totals),
        "redacted_fields_only": _ratio([c for c in totals if c in REDACTED_FIELDS]),
        "per_category": {c: _ratio([c]) for c in sorted(totals)},
    }
    return metrics, details


def _gt_values(gt: dict, fields) -> list[tuple[str, str]]:
    """Flatten a ground-truth dict into (field, value) pairs for the given fields."""
    pairs = []
    for field in fields:
        value = gt.get(field)
        values = value if isinstance(value, list) else [value]
        pairs += [(field, str(v)) for v in values if v not in (None, "")]
    return pairs


def compute_residual(anonymized: list[str], gts: list[dict], fields) -> tuple[dict, list[list[dict]]]:
    """Metric 2: a ground-truth value that still appears in the de-identified text is a residual hit."""
    counts = {f: 0 for f in fields}
    hits = {f: 0 for f in fields}
    per_sample = []
    for text, gt in zip(anonymized, gts):
        sample_hits = []
        for field, value in _gt_values(gt, fields):
            counts[field] += 1
            if value in text:
                hits[field] += 1
                sample_hits.append({"category": field, "value": value})
        per_sample.append(sample_hits)
    total, total_hits = sum(counts.values()), sum(hits.values())
    metrics = {"residual_rate": total_hits / total if total else None, "residual_hits": total_hits,
               "gt_pii": total, "per_category": {f: {"hits": hits[f], "gt": counts[f]} for f in fields if counts[f]}}
    return metrics, per_sample


def _strip_gt(text: str, gt: dict) -> str:
    """Remove every ground-truth PII value, so retention only counts the clinical context."""
    for _, value in _gt_values(gt, PII_FIELDS):
        text = text.replace(value, "")
    return re.sub(r"\s+", " ", text).strip()


def compute_retention(originals: list[str], anonymized: list[str], gts: list[dict], tokenizer) -> tuple[dict, list[dict]]:
    """
    Metric 3. GT PII is removed from both texts (a missed PII must not count as kept context,
    a found one must not count as lost context); placeholders are removed from the output.
    """
    def _count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    per_sample = []
    for original, anon, gt in zip(originals, anonymized, gts):
        orig_tokens = _count(_strip_gt(original, gt))
        anon_tokens = _count(_strip_gt(re.sub(r"\[REDACTED_[A-Z_]+\]", "", anon), gt))
        ratio = min(anon_tokens / orig_tokens, 1.0) if orig_tokens else 1.0
        per_sample.append({"orig_tokens": orig_tokens, "anon_tokens": anon_tokens, "retention": ratio})
    mean = sum(s["retention"] for s in per_sample) / len(per_sample) if per_sample else None
    return {"mean_retention": mean}, per_sample


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _timed_extract(texts, model, system_prompt):
    start = time.perf_counter()
    extracted = extract_pii_from_text(texts=texts, model=model, system_prompt=system_prompt)
    return extracted, time.perf_counter() - start


def _run_info(client, args, eval_set_path: Path) -> dict:
    """Everything needed to reproduce the run: model, runtime, options, code and data versions."""
    def _git(*cmd):
        return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

    with urllib.request.urlopen(f"{args.host}/api/version") as response:
        ollama_version = json.load(response)["version"]
    details = client.show(args.model).details
    return {
        "model": args.model,
        "model_details": details.model_dump() if details else None,
        "gguf_path": args.gguf,
        "ollama_version": ollama_version,
        "ollama_options": config["OLLAMA_CONFIG"].get("OPTIONS", {}),
        "num_parallel": config["OLLAMA_CONFIG"].get("NUM_PARALLEL", 1),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "config")),
        "eval_set": str(eval_set_path),
        "eval_set_sha256": hashlib.sha256(eval_set_path.read_bytes()).hexdigest(),
        "annotated_set": str(args.annotated) if args.annotated else None,
        "tokenizer": args.tokenizer,
        "limit": args.limit,
    }


def _annotated_metrics(originals, gts, extracted, tokenizer) -> tuple[dict, list[dict]]:
    """Metrics 2 and 3 on the annotated notes, plus one example record per note."""
    # Both de-identifications come from the same extraction: (a) the original experiments' protocol,
    # every field redacted; (b) the current pipeline, dates kept, so residual is scored on non-date fields only.
    anon_all = [remove_pii_from_text(t, p, skip_fields=()) for t, p in zip(originals, extracted)]
    anon_pipeline = [remove_pii_from_text(t, p) for t, p in zip(originals, extracted)]
    residual_all, hits_all = compute_residual(anon_all, gts, PII_FIELDS)
    residual_pipeline, hits_pipeline = compute_residual(anon_pipeline, gts, REDACTED_FIELDS)
    retention_all, tokens_all = compute_retention(originals, anon_all, gts, tokenizer)
    retention_pipeline, tokens_pipeline = compute_retention(originals, anon_pipeline, gts, tokenizer)
    metrics = {
        "residual_pii": {"all_fields_redacted": residual_all, "pipeline_dates_kept": residual_pipeline},
        "token_retention": {"all_fields_redacted": retention_all, "pipeline_dates_kept": retention_pipeline},
    }
    examples = [
        {"text": t, "ground_truth": gt, "extracted_pii": p,
         "deidentified_all_fields": a, "deidentified_pipeline": b,
         "residual_hits_all_fields": ha, "residual_hits_pipeline": hb,
         "tokens_all_fields": ta, "tokens_pipeline": tb}
        for t, gt, p, a, b, ha, hb, ta, tb in zip(originals, gts, extracted, anon_all, anon_pipeline,
                                                 hits_all, hits_pipeline, tokens_all, tokens_pipeline)
    ]
    return metrics, examples


def run(args) -> None:
    config["BACKEND"] = "ollama"
    config["OLLAMA_CONFIG"].update({"MODEL": args.model, "HOST": args.host})

    eval_set_path = Path(args.eval_set)
    samples = json.loads(eval_set_path.read_text(encoding="utf-8"))["samples"][:args.limit]
    # Without manual annotations (e.g. a site with raw notes only) only metric 1 can be computed.
    originals, gts = [], []
    if args.annotated:
        annotated = json.loads(Path(args.annotated).read_text(encoding="utf-8"))
        originals, gts = list(annotated)[:args.limit], list(annotated.values())[:args.limit]

    client = load_model(config["MODEL_PATH"])
    info = _run_info(client, args, eval_set_path)
    system_prompt = get_system_prompt()

    # Load the model before timing; an empty request only loads it. Loading a large GGUF over NFS can
    # also exceed Ollama's load timeout, which would turn the first requests into failures.
    # Same options as the real requests: a different num_ctx makes Ollama reload the model on the first one.
    client.generate(model=args.model, prompt="", options=config["OLLAMA_CONFIG"].get("OPTIONS", {}))

    print(f"[1/2] Extracting PII from {len(samples)} injected notes with {args.model}...")
    extracted_inj, time_inj = _timed_extract([s["text"] for s in samples], client, system_prompt)
    print(f"[2/2] Extracting PII from {len(originals)} annotated notes...")
    extracted_ann, time_ann = _timed_extract(originals, client, system_prompt) if originals else ([], 0.0)

    recall, recall_details = compute_recall(samples, extracted_inj)
    annotated_metrics, annotated_examples = {"residual_pii": None, "token_retention": None}, []
    if originals:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        annotated_metrics, annotated_examples = _annotated_metrics(originals, gts, extracted_ann, tokenizer)

    n_notes = len(samples) + len(originals)
    report = {
        "run_info": info,
        "metrics": {
            "synthetic_recall": recall,
            **annotated_metrics,
            "json_failures": sum("raw_response" in p for p in extracted_inj + extracted_ann),
            "seconds_per_note": (time_inj + time_ann) / n_notes if n_notes else None,
            "seconds_total": time_inj + time_ann,
        },
        "examples": {"synthetic_recall": recall_details, "annotated": annotated_examples},
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"eval_{args.model.replace(':', '_').replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    # Not with_suffix(): model tags such as "llama3.3-70b" contain dots that it would treat as a suffix.
    Path(f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text_report = format_report(report)
    Path(f"{stem}.txt").write_text(text_report, encoding="utf-8")
    print(text_report.split("\n\nRECALL MISSES")[0])
    print(f"\nReport: {stem}.json / .txt")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_report(report: dict) -> str:
    """Human-readable report: summary first, then the errors, then every example."""
    info, m, ex = report["run_info"], report["metrics"], report["examples"]
    rec, res, ret = m["synthetic_recall"], m["residual_pii"], m["token_retention"]
    lines = [
        "=" * 70, "ANONYMIZATION EVALUATION REPORT", "=" * 70,
        f"Model: {info['model']}  ({info['model_details']})",
        f"Ollama {info['ollama_version']}, options {info['ollama_options']}, commit {info['git_commit'][:8]}"
        f"{' (dirty)' if info['git_dirty'] else ''}",
        f"Eval set: {info['eval_set']} (sha256 {info['eval_set_sha256'][:12]}), annotated: {info['annotated_set']}",
        "",
        "1. SYNTHETIC PII INJECTION RECALL (higher is better)",
        f"   Overall:              {_pct(rec['overall']['recall'])}  ({rec['overall']['detected']}/{rec['overall']['total']})",
        f"   Redacted fields only: {_pct(rec['redacted_fields_only']['recall'])}  "
        f"({rec['redacted_fields_only']['detected']}/{rec['redacted_fields_only']['total']})",
    ]
    lines += [f"     {c:24s} {_pct(v['recall'])}  ({v['detected']}/{v['total']})" for c, v in rec["per_category"].items()]
    lines += ["", "2. RESIDUAL PII RATE (lower is better)"]
    if res is None:
        lines.append("   Not computed: no annotated set (needs manually labelled notes)")
    else:
        for name, label in [("all_fields_redacted", "All fields redacted (original protocol)"),
                            ("pipeline_dates_kept", "Pipeline, dates kept (non-date GT only)")]:
            r = res[name]
            lines.append(f"   {label}: {_pct(r['residual_rate'])}  ({r['residual_hits']}/{r['gt_pii']})")
            lines += [f"     {c:24s} {v['hits']}/{v['gt']}" for c, v in r["per_category"].items()]
    lines += ["", "3. TOKEN RETENTION RATIO (higher is better, 1.0 = all context kept)"]
    if ret is None:
        lines.append("   Not computed: no annotated set (needs manually labelled notes)")
    else:
        lines += [f"   All fields redacted:  {ret['all_fields_redacted']['mean_retention']:.4%}",
                  f"   Pipeline, dates kept: {ret['pipeline_dates_kept']['mean_retention']:.4%}"]
    lines += ["", f"JSON failures: {m['json_failures']}   Seconds per note: {m['seconds_per_note']:.2f}"
              f"   Total: {m['seconds_total'] / 60:.1f} min"]

    lines += ["", "", "RECALL MISSES", "-" * 70]
    for d in ex["synthetic_recall"]:
        for i in d["injections"]:
            if not i["detected"]:
                lines.append(f"[{d['sample_idx']}] {i['category']}: {i['value']!r} -> extracted "
                             f"{d['extracted_pii'].get(i['category'], d['extracted_pii'].get('raw_response'))!r}")
    lines += ["", "RESIDUAL HITS (all fields redacted)", "-" * 70]
    for index, a in enumerate(ex["annotated"]):
        for h in a["residual_hits_all_fields"]:
            lines.append(f"[{index}] {h['category']}: {h['value']!r}")

    lines += ["", "", "ALL SYNTHETIC RECALL EXAMPLES", "=" * 70]
    for d in ex["synthetic_recall"]:
        lines += [f"--- Sample {d['sample_idx']} ---", d["text"],
                  "Injected: " + ", ".join(f"{i['category']}={i['value']!r}{'' if i['detected'] else ' (MISSED)'}"
                                           for i in d["injections"]),
                  "Extracted: " + json.dumps(d["extracted_pii"], ensure_ascii=False), ""]
    lines += ["", "ALL ANNOTATED EXAMPLES", "=" * 70]
    for index, a in enumerate(ex["annotated"]):
        lines += [f"--- Annotated {index} ---", "Original:  " + a["text"],
                  "GT:        " + json.dumps({k: v for k, v in a["ground_truth"].items() if v}, ensure_ascii=False),
                  "Extracted: " + json.dumps(a["extracted_pii"], ensure_ascii=False),
                  "De-identified (all fields): " + a["deidentified_all_fields"],
                  "De-identified (pipeline):   " + a["deidentified_pipeline"],
                  f"Tokens (all fields): {a['tokens_all_fields']}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an LLM on the anonymization pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-eval-set", help="Freeze the injected samples to a file")
    build.add_argument("--output", default=str(DEFAULT_EVAL_SET))
    build.add_argument("--from-report", help="Take the samples from an old evaluation_*.json report")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--n-synthetic", type=int, default=500)

    evaluate = commands.add_parser("run", help="Evaluate one Ollama model")
    evaluate.add_argument("--model", required=True, help="Ollama model tag")
    evaluate.add_argument("--host", default=config["OLLAMA_CONFIG"]["HOST"])
    evaluate.add_argument("--gguf", help="GGUF file the tag was created from (recorded in the report only)")
    evaluate.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    evaluate.add_argument("--annotated", default=str(DEFAULT_ANNOTATED) if DEFAULT_ANNOTATED.exists() else None,
                          help="Manually annotated notes for metrics 2-3 (default: data/tagged_samples.json if "
                               "present; without it only recall is computed)")
    evaluate.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                          help="Tokenizer folder for the retention metric (default: TOKENIZER_PATH in .env)")
    evaluate.add_argument("--limit", type=int, default=None, help="Use only the first N notes of each set (smoke test)")
    evaluate.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))

    args = parser.parse_args()
    if args.command == "run" and args.annotated and not args.tokenizer:
        parser.error("--tokenizer (or TOKENIZER_PATH in .env) is needed for the retention metric")
    if args.command == "build-eval-set":
        build_eval_set(Path(args.output), args.from_report, args.seed, args.n_synthetic)
    else:
        run(args)


if __name__ == "__main__":
    main()
