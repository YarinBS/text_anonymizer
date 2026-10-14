# Running the evaluation (Ollama, Windows or Linux)

This test measures how well a model finds personal details (PII) in clinical notes. It takes real notes, adds **fake** PII with known values (name, ID, age, doctor, dates, city), runs the anonymizer, and counts how many of the fake values the model found (**recall**). No labelled data is needed.

Tested setup: Ollama 0.33.2, Python 3.12, on an RTX A4000 16 GB (Windows) and an H200 (Linux). Commands are given for both systems; run them from the repository folder.

## 1. One-time setup

### Code

Clone or `git pull` this repository, create a virtual environment, and install everything in `requirements.txt` **except `vllm` and `pytest`**:

| Windows (PowerShell) | Linux |
|---|---|
| `py -3.12 -m venv .venv` | `python3.12 -m venv .venv` |
| `.venv\Scripts\activate` | `source .venv/bin/activate` |

Then, on both:

```
pip install pandas==3.0.5 pydantic==2.12.4 PyYAML==6.0.2 python-dotenv==1.2.1 ollama==0.6.3 transformers==5.13.1
```

### Models

Download both files (12.7 GB each):

| Ollama name | File | SHA-256 |
|---|---|---|
| `gemma4-26b-a4b-q3` (first choice) | [gemma-4-26B-A4B-it-UD-Q3_K_M.gguf](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf) | `811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052` |
| `gemma4-12b-q8` (backup) | [gemma-4-12b-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q8_0.gguf) | `f20e7ff1be28c283eeeb18fc895733791c56a5851d5cd3fe9691b7f7d12afa72` |

On Linux you can download with `curl -L -O <link>`. Then check each file; the result must match the SHA-256 above:

| Windows | Linux |
|---|---|
| `Get-FileHash <file> -Algorithm SHA256` | `sha256sum <file>` |

### Ollama

Use Ollama 0.33.2 (older versions do not support Gemma 4). Ollama must run with **`OLLAMA_NUM_PARALLEL=1`**: with more parallel requests the model does not fit in 16 GB of GPU memory.

| Windows | Linux |
|---|---|
| Install Ollama 0.33.2. Add the environment variable `OLLAMA_NUM_PARALLEL=1` (System settings → Environment variables), then quit and restart Ollama. | With root: `curl -fsSL https://ollama.com/install.sh \| OLLAMA_VERSION=0.33.2 sh`, then `sudo systemctl edit ollama`, add `[Service]` and `Environment="OLLAMA_NUM_PARALLEL=1"`, and run `sudo systemctl restart ollama`. |
| | Without root: download [ollama-linux-amd64.tar.zst](https://github.com/ollama/ollama/releases/download/v0.33.2/ollama-linux-amd64.tar.zst), unpack it (`mkdir -p ~/ollama && tar --zstd -xf ollama-linux-amd64.tar.zst -C ~/ollama`), and keep it running in a separate terminal (or tmux): `OLLAMA_NUM_PARALLEL=1 ~/ollama/bin/ollama serve`. On a machine with several GPUs, add `CUDA_VISIBLE_DEVICES=<id>` to choose one. |

Register each model once. Create a text file `Modelfile` with one line, `FROM <full path to the .gguf file>`, and run:

```
ollama create gemma4-26b-a4b-q3 -f Modelfile
ollama create gemma4-12b-q8 -f Modelfile
```

(change the `FROM` line between the two commands). Ollama copies the model, so you can then delete the `.gguf` files (about 25 GB). Check with `ollama list`.

### Config

In `config/config.yaml`:

- `BACKEND: ollama`
- `DATA_PATH`: the CSV with the notes
- `COLUMNS_TO_DEIDENTIFY`: the first name must be the text column

Leave `OLLAMA_CONFIG` as it is (`num_ctx: 8192`, `NUM_PARALLEL: 1`). If Ollama does not run at its default address (`http://127.0.0.1:11434`), add `--host <address>` to the `run` commands below.

## 2. Build the test set (once)

This picks random notes and adds the fake PII. It takes seconds and does not use the model. (On Windows, `/` in the paths also works.)

| | Command |
|---|---|
| **Subset** (quick check, 50 notes) | `python src/evaluate.py build-eval-set --n-synthetic 50 --output data/eval_sets/local_n50.json` |
| **Full** (500 notes, same size as the lab's evaluation) | `python src/evaluate.py build-eval-set --n-synthetic 500 --output data/eval_sets/local_n500.json` |

The same seed (42) gives the same set every time. Run both models on the **same** file, so their scores can be compared.

## 3. Run the evaluation, once per model

```
python src/evaluate.py run --model gemma4-26b-a4b-q3 --eval-set data/eval_sets/local_n50.json
python src/evaluate.py run --model gemma4-12b-q8 --eval-set data/eval_sets/local_n50.json
```

For the full evaluation, use `local_n500.json` instead. Run the subset first: it shows the time per note, so you can estimate the full run (500 × seconds per note). For a run that takes hours on Linux, start it inside `tmux` or with `nohup`, so it does not stop when the terminal closes.

**While it runs**, open a second terminal:

- `ollama ps`: the PROCESSOR column must show **100% GPU**. If it shows any CPU share, the model does not fit. It still works, but much more slowly; please report it.
- `nvidia-smi`: note the memory used.

## 4. Results

Each run prints a summary and saves a full report in `results/eval_<model>_<time>.json` and `.txt`.

- **Section 1** gives the recall, overall and per type of PII. Higher is better.
- **Sections 2 and 3** say "Not computed". This is expected: they need manually labelled notes.
- **Last line**: JSON failures (should be 0), seconds per note, total time.

**Please send back**, for each model: the printed summary (from `ANONYMIZATION EVALUATION REPORT` to the `JSON failures` line, numbers only), the `ollama ps` line, the `nvidia-smi` memory value, and any error message.

## Privacy

The files in `data/eval_sets/` and `results/` contain real notes (with fake PII added). **Keep them on this machine; never send or upload them.** Only the printed summary may be shared.

## With labelled notes (optional)

If a file of manually labelled notes is available (format: `{"note text": {"patient_name": ..., "intervention_dates": [...], ...}}`, as in `data/tagged_samples.json`), add `--annotated <file> --tokenizer <folder>` to the run command. This also computes the residual PII rate and the token retention ratio. The tokenizer folder must hold `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` from [unsloth/Llama-3.3-70B-Instruct](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/tree/main) (the same tokenizer for every model, so the ratios are comparable).
