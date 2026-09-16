# text_anonimyzer

`text_anonimyzer` anonymizes clinical free-text by extracting personally identifiable information (PII) and replacing it with redaction placeholders.

For **end users**, the important runtime files are the Python modules directly under
`src/`:

1. `src/main.py` — pipeline entry point. Loads the configured model and data, runs
   PII extraction and redaction for each configured column, writes per-column text
   outputs, and saves a final CSV with deidentified columns.
2. `src/utils.py` — core anonymization logic. Defines the PII schema, loads the
   vLLM model, loads text/CSV data, builds the extraction prompt, parses model
   output, and replaces extracted PII in the source text.
3. `src/config.py` — loads `config/config.yaml` and `.env` values for the runtime
   modules.
4. `src/logger.py` — configures console logging and timestamped log files under
   `logs/`.

Files under `src/legacy/` are old experiment and evaluation scripts. They are kept
for reference, but they are not needed to run the current pipeline.

## What the pipeline does

The pipeline:
1. Loads clinical notes from `DATA_PATH` (`.csv` or `.txt`)
2. Loads the configured vLLM model from `MODEL_PATH`
3. Extracts structured PII fields from each note
4. Replaces detected PII spans in the original text with placeholders like `[REDACTED_PATIENT_NAME]`
5. Adds the deidentified columns back to the source DataFrame
6. Writes a final CSV. CSV fields containing newlines are preserved correctly.

## Configure paths and model settings

Edit `config/config.yaml` before running:
- `DATA_PATH`: input file path (`.csv` or `.txt`)
- `COLUMNS_TO_DEIDENTIFY`: CSV columns to process
- `MODEL_PATH`: Hugging Face model path used by vLLM
- `VLLM_CONFIG`: vLLM model and generation settings
- `LLAMA_CPP_CONFIG`: legacy llama.cpp settings kept for older code paths

If the input is CSV, every column in `COLUMNS_TO_DEIDENTIFY` must exist in the input
file.

## Install and run

```bash
pip install -r requirements.txt
python src/main.py
```

## Repository structure

```text
.
├── README.md
├── NOTES.md
├── TODO.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── injections_merged.csv
│   ├── only_injections_freetext.csv
│   └── tagged_samples.json
├── notebooks/
│   └── exploratory notebooks and intermediate analysis files
├── results/
│   └── historical run outputs, evaluation reports, and analysis artifacts
├── scripts/
│   ├── install_cuda_env.sh
│   └── setup.sh
└── src/
    ├── main.py
    ├── utils.py
    ├── config.py
    ├── logger.py
    └── legacy/
        └── old experiment and evaluation scripts
```

## Notes for users

- `src/main.py` is the script you run.
- `src/utils.py`, `src/config.py`, and `src/logger.py` are the supporting runtime modules.
- Ignore `src/legacy/` unless you are inspecting older experiments or evaluation code.
