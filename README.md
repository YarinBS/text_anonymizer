# Text Anonymizer

Text Anonymizer de-identifies clinical free text (Hebrew and English) with a local language model. The model finds the personal details in each note (names, IDs, ages, dates, places), and fixed rules replace them with placeholders such as `[REDACTED_PATIENT_NAME]`. Nothing is sent to an external service.

What is removed, what is kept and what risks remain, for reviewers of the output: [DEIDENTIFICATION.md](DEIDENTIFICATION.md). Step-by-step model evaluation on a Windows/Ollama machine: [EVALUATION.md](EVALUATION.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/main.py` | the pipeline flow only (entry point) |
| `src/utils.py` | model loading, the prompt and JSON schema, extraction (vLLM or Ollama), replacement, and the pipeline's input/output helpers (input loading, structured dates, `age_at_visit`, shareable file, saved extractions) |
| `src/redaction.py` | placeholder-safe replacement and the regex backstop |
| `src/date_shift.py` | dates in the text → offsets from the visit date |
| `src/evaluate.py`, `src/eval_injection.py` | model evaluation (below) |
| `scripts/audit_output.py` | counts possible PII left in an output file (counts only, never text) |
| `config/config.yaml` | default config (original output format); `--config` files override it |
| `config/hy.yaml`, `config/rambam.yaml` | overrides for the HY and Rambam data |
| `tests/` | unit tests, synthetic text only |
| `src/legacy/` | earlier scripts, not used by the pipeline |

## How the pipeline works

For every configured text column, `src/main.py`:

1. **Selects the records:** cells that contain at least one letter. Other cells get an empty output.
2. **Extracts** the PII of each note with the model. The answer must follow a JSON schema (`MedicalInformation` in `src/utils.py`) in which every key is required. A note whose prompt is too long, or whose answer is cut off or is not valid JSON, counts as failed.
3. **Replaces** what was found, in this order:
   - dates (only with `DATE_MODE: visit_offset`);
   - the model's values: names, IDs, ages, places;
   - the regex backstop (only with `REGEX_BACKSTOP`).

   A failed note becomes `[ANONYMIZATION_ERROR]`, with no text kept.
4. **Writes** the output files (see Outputs).

The prompt (`get_system_prompt`) gives the model the PII categories, few-shot examples, an ophthalmology glossary (terms that are *not* PII), and the rule that every value must be an exact copy from the text. That rule matters, because replacement looks for that exact text.

### Placeholders

Every placeholder, and the date rules: [DEIDENTIFICATION.md](DEIDENTIFICATION.md). Failed notes become `[ANONYMIZATION_ERROR]`.

### Always-on rules (every config)

- The age is replaced only as a whole number. Plain string replacement also hit the same digits inside other numbers: with age 71, "VA 0.71", "6/71", "IOP 171" and "3.4.71".
- The ID keeps plain string replacement. It rarely collides with other numbers, and a missed ID would be a leak.
- Raw model answers and note texts are never logged. Logs hold row numbers only.

### Privacy switches

`config/config.yaml` has every switch off. It keeps the original output format: dates stay in the text, and names are replaced by plain string matching. A `--config` file holds only the keys it changes, on top of `config/config.yaml` (nested blocks such as `OLLAMA_CONFIG` are merged one level deep):
- `config/hy.yaml` (HY data): every switch on except `DATE_MODE: keep`;
- `config/rambam.yaml`: every switch on.

| Switch | Effect |
|---|---|
| `NAME_LISTS` | `patient_name` and `doctor_name` return every form of the name (lists), not one string. |
| `SAFE_TEXT_MATCHING` | Longest value first. Never edits inside a placeholder. Names of 3 letters or less must stand as a whole word, and Hebrew prefixes (ו ב כ ל מ ש ה) are kept; if that finds nothing, plain matching is used. Longer values tolerate extra spaces and Latin letter case. |
| `REGEX_BACKSTOP` | After the model: phones (Israeli, +972), emails, URLs, 8–9 digit numbers with a valid Israeli ID check digit, and "בן/בת/גיל/age + number". |
| `DATE_MODE` | `keep` (dates stay), or `visit_offset`: every date found becomes its distance from the row's visit date. Full dates the model missed are caught by a pattern (d/m/y with one separator, ISO y-m-d, "d month y"). Two-part shapes such as 6/9 or 21/18 are touched only if the model marked them as dates, because they are usually visual acuity or eye pressure. Needs `VISIT_DATE_COLUMN`. |
| `FLAGS_COLUMN` | Adds `<column>_deid_flags`, holding field names, never values: `unmatched:<field>` (a value not found in the text), `backstop:<pattern>`, `no_visit_date`, `anonymization_error`. |
| `SAVE_EXTRACTIONS` | Writes the model's raw answers to `<input>_extractions.jsonl`, each with a SHA-256 hash of its note. |
| `SHARE_COLUMNS` | Also writes `<input>_shareable.csv` with only these columns. |

Date rules (`visit_offset`): [DEIDENTIFICATION.md](DEIDENTIFICATION.md). Structured date columns without a configured format (`VISIT_DATE_FORMAT`, `BIRTHDATE_FORMAT`) are read day first, except ISO dates. A value that cannot be read stops the run.

## Requirements and installation

- Python 3.12. For vLLM: Linux with a CUDA GPU large enough for the model. For Ollama (e.g. Windows): an Ollama server.
- Exact package versions are in `requirements.txt`.

```bash
bash scripts/setup.sh                       # creates .venv and installs requirements.txt
# or, in an existing environment:
python -m pip install -r requirements.txt   # on Windows/Ollama: everything except vllm
```

## Configuration

Set the model path in `.env` (template: `.env.example`) or in the environment:

```dotenv
MODEL_PATH=/path/to/Llama-3.3-70B-Instruct
```

In the config file:
- `DATA_PATH`: the input CSV or TXT file (relative paths start at the repository root);
- `COLUMNS_TO_DEIDENTIFY`: the text columns (TXT input: exactly one name);
- `BACKEND`: `vllm` (with `VLLM_CONFIG`) or `ollama` (with `OLLAMA_CONFIG`);
- the privacy switches above.

**vLLM:**
- `tensor_parallel_size` is the number of GPUs.
- Notes whose prompt plus `max_tokens` exceeds `max_model_len` are not sent; they count as failed.

**Ollama:**
- The model tag goes in `OLLAMA_CONFIG.MODEL`, and the server should run with `OLLAMA_NUM_PARALLEL` equal to `NUM_PARALLEL`.
- Keep `num_ctx` at 8192 or more. The prompt alone is about 3,000 tokens, and Ollama silently cuts longer prompts. The check on the reported prompt length is only partial.

## Usage

```bash
python src/main.py                                   # config/config.yaml
python src/main.py --config config/rambam.yaml       # all privacy switches on
python src/main.py --config config/hy.yaml --data-path <file.csv>   # HY data (injections: config/hy_injections.yaml)
python src/main.py --config <cfg> --reuse-extractions <input>_extractions.jsonl   # re-apply replacement, no model
```

`--reuse-extractions` uses the saved model answers. A row whose text changed since then (different hash) counts as failed, so a reordered file cannot pair a note with another note's answer. With it, a change to the replacement rules needs no model run.

### Running on the Rambam data

1. Install as above (on Windows without `vllm`), and set up Ollama with the model `gemma4-26b-a4b-q3` ([EVALUATION.md](EVALUATION.md), section 1).
2. Open `config/rambam.yaml` and replace every `FILL_ME`:

   | Key | What to write |
   |---|---|
   | `DATA_PATH` | the notes CSV, e.g. `data/rambam_notes.csv` |
   | `COLUMNS_TO_DEIDENTIFY` | the free-text column(s), e.g. `["note_text"]` |
   | `VISIT_DATE_COLUMN` | the column with each row's visit date (set `VISIT_DATE_FORMAT` if it is not day/month/year or ISO) |
   | `BIRTHDATE_COLUMN` | the birthdate column, or `null` if there is none |
   | `SHARE_COLUMNS` | the columns of the file that leaves the hospital: the patient ID column, the visit-date column, and `<text column>_deidentified` and `<text column>_deid_flags` for each text column |

   The run stops before loading the model if a `FILL_ME` is left, or if a column is not in the file.
3. Run `python src/main.py --config config/rambam.yaml`.
4. Check the output: `python scripts/audit_output.py <input>_shareable.csv`. Send only `<input>_shareable.csv` and the audit's printed counts. `<input>_deidentified_final.csv` and `<input>_extractions.jsonl` hold the original text and the PII: keep them on the hospital machine.

### Outputs (written next to the input)

| File | Contents | Share? |
|---|---|---|
| `<input>_deidentified_final.csv` | every input column, in the same row order, plus `<column>_deidentified` (and `<column>_deid_flags`) | **no**: holds the original text |
| `<input>_extractions.jsonl` | the model's answers (`SAVE_EXTRACTIONS`) | **no**: holds the PII |
| `<input>_shareable.csv` | only `SHARE_COLUMNS`. Raw text and birthdate columns are refused; the birthdate becomes `age_at_visit` and `AGE_COLUMNS` are top-coded (every age ≥ 90 → 90). | after review and `audit_output.py` |
| `logs/run_<time>.log` | progress and the row numbers of failed notes | no text |

For TXT input, each non-empty line is one record. If the model returns a different number of results than records, the run stops before writing.

## Evaluation

### Model quality: `src/evaluate.py` (Ollama backend)

`src/evaluate.py` scores one Ollama model. It writes `results/eval_<model>_<time>.json` and `.txt`, with every example used. It scores the switches of `--config` (default: `config/config.yaml`); compare models only when they were scored with the same config.

1. **Synthetic PII injection recall.** Real notes get injected PII of known value (`eval_injection.py`). The frozen set is in `data/eval_sets/`, so every model gets the same inputs. Recall = injected values the model returned / injected values, per category.
2. **Residual PII rate**, on the 100 annotated notes (`data/tagged_samples.json`): the share of ground-truth PII values still in the de-identified text.
   - A failed note counts all its PII as residual.
   - The age counts only as a whole number ("VA 0.71" is not a leaked age 71). Because of this rule, age residuals from reports before 2026-11-04 are not comparable.
3. **Token retention ratio**, on the same notes: the tokens kept, divided by the original tokens. Both counts are taken after removing the ground-truth PII and every placeholder, with one fixed tokenizer for all models (`TOKENIZER_PATH`).

Metrics 2 and 3 are given twice: with every field redacted (the original experiments), and with the pipeline default (dates kept).

```bash
python src/evaluate.py build-eval-set --from-report results/evaluation_20260228_170735.json   # once
python src/evaluate.py --config <cfg> build-eval-set --n-synthetic 50 --output data/eval_sets/local_n50.json  # from the config's DATA_PATH notes
python src/evaluate.py --config <cfg> run --model <ollama tag> --host http://127.0.0.1:11434 [--eval-set <file>] [--limit 5]
```

The reports contain real clinical notes: keep them on the secured server.

### Output check: `scripts/audit_output.py`

```bash
python scripts/audit_output.py <file.csv> [<file.csv> ...]
```

For every `*_deidentified` column it counts the cells that still contain:
- a phone number, an email, a URL, or a valid Israeli ID;
- an age phrase;
- a full date (d/m/y or ISO);
- a doctor title followed by a word;
- error rows, or broken placeholders.

It prints counts only, so its output can be shared. Run it on every file before it is shared, and compare it with the `_deid_flags` column.

## Testing

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests
```

The tests use synthetic text only:
- `tests/test_privacy.py`: the always-on rules, every switch, the date rules, the outputs, the audit, the residual metric;
- `tests/test_pipeline.py`: input handling and row alignment;
- `tests/test_evaluation.py`: the evaluation code.

## Privacy

Only process data you are authorized to use. Never commit input data, outputs, `.env`, or model paths; `.gitignore` excludes the usual output names. `_deidentified_final.csv` and `_extractions.jsonl` must never leave the secured machine.
