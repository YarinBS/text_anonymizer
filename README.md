# Text Anonymizer

Text Anonymizer processes clinical free text with a local vLLM language model. It extracts configured personally identifiable information (PII) and replaces detected values with redaction placeholders such as `[REDACTED_PATIENT_NAME]`.

The repository contains the current pipeline under `src/`, configuration under `config/`, installation scripts under `scripts/`, and automated tests under `tests/`.

## Requirements

- Python 3.12
- A local Hugging Face model supported by vLLM
- A compatible CUDA/GPU environment for the configured model

The exact Python package versions are listed in `requirements.txt`.

## Installation

Create the virtual environment and install the pinned dependencies:

```bash
bash scripts/setup.sh
```

Alternatively, install the dependencies in an existing environment:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Copy the environment template and set the local model path:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
MODEL_PATH=/path/to/your/Llama-3.3-70B-Instruct
```

Edit `config/config.yaml` to set:

- `DATA_PATH`: the input CSV or TXT file. Relative paths are resolved from the repository root.
- `COLUMNS_TO_DEIDENTIFY`: CSV columns to process. TXT input requires exactly one column name.
- `VLLM_CONFIG`: model and generation settings.

The configuration file uses a repository-relative default data path. Replace it with the path to data that you are authorized to process.

## Usage

Run the pipeline from any directory:

```bash
python src/main.py
```

For CSV input, the pipeline:

1. Loads the complete input DataFrame.
2. Processes valid text values in each configured column.
3. Adds a `<column_name>_deidentified` column for each processed column.
4. Preserves the original columns and row order.
5. Writes `<input_name>_deidentified_final.csv` beside the input file.

For TXT input, each non-empty line is treated as one record. The configured column name is used for the original text column, and the output CSV contains both that column and its deidentified counterpart.

Fields containing embedded newlines are preserved correctly in CSV output. If the model returns a different number of results than the number of input records, the pipeline stops before writing output.

## Privacy

Only process data you are authorized to use. Do not commit input data, generated outputs, environment files, model credentials, or local model paths. These are excluded by `.gitignore` where appropriate.

The current pipeline intentionally retains dates in the text and preserves the original input columns in the output CSV. Review and remove sensitive original data before sharing results.

## Testing

Run the tests after installing the dependencies:

```bash
python -m pytest
```
