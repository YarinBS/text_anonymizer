"""
utils.py

This module contains utility functions for loading the model, loading data, 
extracting personally identifiable information (PII) from clinical texts, and removing PII from texts.
It also defines the schema for the extracted medical information using Pydantic models.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from config import config
from date_shift import shift_dates
from logger import get_logger
from redaction import age_pattern, apply_regex_backstop, replace_text_value, sub_outside_placeholders


logger = get_logger()

# Dates are extracted but kept in the de-identified text on purpose (they are needed downstream).
DATE_FIELDS = ("intervention_dates", "other_dates", "patient_date_of_birth")


class MedicalInformation(BaseModel):
    """Schema for extracted medical information"""
    patient_name: Optional[str] = Field(default=None, description='Full or partial name of the patient', min_length=2)
    patient_id: Optional[str] = Field(default=None, description='ID (תעודת זהות, או ת"ז) number of the patient', min_length=7, max_length=9)
    patient_age: Optional[int] = Field(default=None, description='Age of the patient')
    patient_date_of_birth: Optional[str] = Field(default=None, description='Date of birth of the patient')
    intervention_dates: List[str] = Field(..., description='List of dates related to medical interventions (e.g., surgeries, treatments)', default_factory=list)
    other_dates: List[str] = Field(..., description='List of other relevant dates that appear in the text (e.g., admission, discharge)', default_factory=list)
    doctor_name: Optional[str] = Field(default=None, description='Full or partial name of the doctor, usually appended with ד"ר or דר', min_length=2)
    other_mentioned_names: List[str] = Field(..., description='List of other mentioned names, full or partial, in the text', default_factory=list)
    locations: List[str] = Field(..., description='List of physical geographic locations mentioned in the text (e.g., street address, city, county, precinct, ZIP code), NOT including eye side (Left, Right, LE, RE, שמאל, ימין, etc.)', default_factory=list)


class MedicalInformationNameLists(MedicalInformation):
    """
    Same schema, but every form of the patient's and the doctors' names can be returned (NAME_LISTS).

    With one string per field, a second name form or a second doctor had nowhere to go: in the HY notes
    ~38.9k doctor titles were followed by a word, against only ~5.3k doctor tags. Field names stay the
    same, so evaluate.py and eval_injection.py keep working (they already accept list values).
    """
    patient_name: List[str] = Field(default_factory=list, description='Every form of the patient name (full or partial) in the text')
    doctor_name: List[str] = Field(default_factory=list, description='Every doctor name (full or partial) in the text, usually after ד"ר or דר')


def response_model() -> type[BaseModel]:
    """The extraction schema chosen by the NAME_LISTS switch (off = the original one-string fields)."""
    return MedicalInformationNameLists if config.get("NAME_LISTS", False) else MedicalInformation


def _response_schema() -> dict:
    """
    JSON schema for structured output, with every key required, for both backends.

    Without `required`, the grammar lets the model close the JSON after the first key: Llama-3.3 on
    Ollama did that in 493/500 notes, hiding all other PII. vLLM had no such guard before.
    """
    schema = response_model().model_json_schema()
    schema["required"] = list(schema["properties"])
    return schema


def load_model(model_path: str):
    """
    Load the model for the configured BACKEND.

    Parameters:
    - model_path (str): The path to the model file (vLLM only; Ollama models are named in OLLAMA_CONFIG).

    Returns:
    - The loaded vLLM model instance, or an Ollama client for BACKEND "ollama".
    """

    if config.get("BACKEND", "vllm") == "ollama":
        # Ollama serves the model itself; we only need a client for its HTTP API.
        import ollama
        return ollama.Client(host=config["OLLAMA_CONFIG"]["HOST"])

    if not model_path:
        raise RuntimeError("MODEL_PATH must be set in the environment or .env")

    # Imported here so that the Ollama backend works on machines without vLLM (e.g. Windows).
    from vllm import LLM

    vllm_config = config["VLLM_CONFIG"]["MODEL_CONFIG"]
    model = LLM(
        model=model_path,
        tensor_parallel_size=vllm_config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.95),
        max_model_len=vllm_config.get("max_model_len", 4096),
        enable_prefix_caching=vllm_config.get("enable_prefix_caching", True),
        dtype=vllm_config.get("dtype", "bfloat16"),
        disable_log_stats=True,
        enforce_eager=vllm_config.get("enforce_eager", False),
    )
    
    return model


def is_valid_text(value) -> bool:
    """Return whether a value is text worth sending to the model."""
    return isinstance(value, str) and any(char.isalpha() for char in value)


def get_valid_text_records(df: pd.DataFrame, column: str) -> list[tuple[object, str]]:
    """Return ``(original_index, text)`` pairs using the pipeline's text filter."""
    return [
        (index, value)
        for index, value in df[column].items()
        if is_valid_text(value)
    ]


def load_data(data_path: str, column: str) -> List[str]:
    """
    Load clinical text data from a specified file path.

    Parameters:
    - data_path (str): The path to the data file, either a .txt file or a .csv file.
    - column (str): The name of the column to extract from the CSV file. Ignored if the file is a .txt file.

    Returns:
    - List[str]: A list of clinical text entries loaded from the file.
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        return [text for _, text in get_valid_text_records(df, column)]
    
    elif data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            # A non-empty TXT line is a record; the shared CSV filter is applied
            # later when deciding which records are safe to send to the model.
            return [line for line in f.read().splitlines() if line.strip()]
        
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")


def get_system_prompt(response_schema: BaseModel = None) -> str:
    """
    Return the system prompt for the model, 
    which includes instructions for extracting personally identifiable information (PII) from clinical texts, 
    as well as a glossary of ophthalmology terms and few-shot examples.

    Parameters:
    - response_schema (BaseModel): The Pydantic model schema that defines the expected output format for the extracted information. 
    Default is the MedicalInformation class.

    Returns:
    - str: The system prompt string to be used as input for the model.
    """
    
    OPHTHALMOLOGY_GLOSSARY = """
    Here are some common ophthalmology terms that may appear in the clinical texts. Note that there may be typos in the actual texts, so these terms may appear somewhat differently. These should NOT be extracted as PII:
    - AMD (Age-related Macular Degeneration) - This may also appear in texts like "web AMD", "dry AMD", "LE AMD", "RE AMD", "AMD רטוב", "AMD יבש", etc.
    - EAMD (Exudative Age-related Macular Degeneration)
    - PRN (Pro Re Nata) - a treatment regimen where medication is given as needed
    - CNV (Choroidal Neovascularization)
    - OCT (Optical Coherence Tomography)
    - Avastin (אווסטין) - a medication used for treating AMD. This treatment uses the Bevacizumab molecule
    - Eylea (אילייה or איליה or אייליה or איליא) - another medication used for treating AMD. This treatment uses the Aflibercept molecule.
    - Lucentis (לוסנטיס or לוסנטס) - another medication used for treating AMD. This treatment uses the Ranibizumab molecule.
    - ימין (right eye) and שמאל (left eye) - these refer to the patient's eyes and should not be extracted as locations. May also appear as "Left", "Right", "LE", "RE", etc.
    """

    FEW_SHOT_EXAMPLES = """
    Here are some examples for medical texts and their expected corresponding outputs:

    Input text:
    מטופל אבי כהן מחיפה, תז 216334425, בן 51, נולד ב- 06.08.1972, התקבל בתאריך 15 לינואר 2024, עבר ניתוח חירום בתאריך 17.1.24 ושוחרר בתאריך 02/02/2024. בשיתוף עם ד"ר פרידמן

    Output:
    {
        "patient_name": "אבי כהן",
        "patient_id": 216334425,
        "patient_age": 51,
        "patient_date_of_birth": "06.08.1972",
        "intervention_dates": [
            "17.1.24"
        ],
        "other_dates": [
            "15 לינואר 2024",
            "02/02/2024"
        ],
        "doctor_name": "ד\\"ר פרידמן",
        "other_mentioned_names": [],
        "locations": [
            "חיפה"
        ]
    }

    Input text:
    AMD רטוב, נשלחה לזריקות אווסטין מימין (אינדקציה 22/11/21)עין ימין17/01/22 זריקת אווסטין #114/02/22 זריקת אווסטין #214.3.22 זריקת אבסטין #3 מהלך תקיןתכניתעוד חודש ביקורת מרפאת רשתית AMD עם OCT מקולרי

    Output:
    {
        "patient_name": null,
        "patient_id": null,
        "patient_age": null,
        "patient_date_of_birth": null,
        "intervention_dates": [
            "17/01/22",
            "14/02/22",
            "14.3.22"
        ],
        "other_dates": [
            "22/11/21"
        ],
        "doctor_name": null,
        "other_mentioned_names": [],
        "locations": []
    }

    Input text:
    wet AMD נשלחה לזריקות אווסטין בעין שמאל (אינדקציה 04/03/2020)עין שמאל12/03/2020 זריקת אווסטין מספר 116/04/2020 זריקת אווסטין מספר 2מהלך תקיןתכניתהמשך זריקות

    Output:
    {
        "patient_name": null,
        "patient_id": null,
        "patient_age": null,
        "patient_date_of_birth": null,
        "intervention_dates": [
            "16/04/2020",
            "12/03/2020"
        ],
        "other_dates": [
            "04/03/2020"
        ],
        "doctor_name": null,
        "other_mentioned_names": [],
        "locations": []
    }

    Input text:
    LE AMD 

    Output:
    {
        "patient_name": null, 
        "patient_id": null, 
        "patient_age": null, 
        "patient_date_of_birth": null, 
        "intervention_dates": [], 
        "other_dates": [], 
        "doctor_name": null, 
        "other_mentioned_names": [],
        "locations": []
    }

    Input text:
    לסיכום בן 71 עין שמאל (העין הטובה) עם CNV פעיל,לאחר שלל סוגי anti VEGF, כעת שיפור הדמייתי תחת אילייה חודשיות 

    Output:
    {
        "patient_name": null, 
        "patient_id": null, 
        "patient_age": 71, 
        "patient_date_of_birth": null, 
        "intervention_dates": [], 
        "other_dates": [], 
        "doctor_name": null, 
        "other_mentioned_names": [], 
        "locations": []
    }

    Input text:
    1.7.24 EAMD בשתי העיניים 

    Output:
    {
        "patient_name": null,
        "patient_id": null,
        "patient_age": null,
        "patient_date_of_birth": null,
        "intervention_dates": [
            "1.7.24"
        ],
        "other_dates": [],
        "doctor_name": null,
        "other_mentioned_names": [],
        "locations": []
    }
    """

    response_schema = response_schema or response_model()
    if response_schema is MedicalInformationNameLists:
        # Show the list form in the examples too; the default prompt stays byte-identical.
        FEW_SHOT_EXAMPLES = (FEW_SHOT_EXAMPLES
                             .replace('"patient_name": "אבי כהן"', '"patient_name": ["אבי כהן"]')
                             .replace('"doctor_name": "ד\\"ר פרידמן"', '"doctor_name": ["ד\\"ר פרידמן"]')
                             .replace('"patient_name": null', '"patient_name": []')
                             .replace('"doctor_name": null', '"doctor_name": []'))

    system_prompt = f"""
    You are an expert performing conservative information extraction. If information is not explicitly present, you must return null. Given a clinical text in Hebrew or English, your task is to identify the following personally identifiable information (PII):

    1. Names, full or partial (e.g., patient names, doctor names)
    2. All geographic subdivisions smaller than a state (street address, city, county, precinct, ZIP code)
    3. All elements of dates directly related to the individual (birth date, admission/discharge dates, diagnosis/treatment dates, date of death, ages > 89 years, and any date elements indicating such)
    4. ID number(s) of the patient, relative or a doctor (e.g., medical record numbers, account numbers, Social Security numbers)

    Notes:
    - Sometimes, the text may not contain some or all of the PIIs - in this case, return a null value in the field where no value was found.
    - Do NOT fabricate any information - only extract what is explicitly mentioned in the text, and do NOT hallucinate/guess information if it is not explicitly mentioned in the given text.
    
    - The text may contain injection entries where line breaks were lost during extraction, causing an injection serial number (e.g., #1, #2, #3) to merge with the date on the following line. For example, the original text:
    "
    17/01/22 זריקת אווסטין #1
    14/02/22 זריקת אווסטין #2
    14.3.22 זריקת אווסטין #3
    "

    may appear in the extracted text as:
    "17/01/22 זריקת אווסטין #114/02/22 זריקת אווסטין #214.3.22 זריקת אווסטין #3"

    Similarly, "זריקת איליא מספר 116/04/2020" is actually "מספר 1" followed by the date "16/04/2020".
    In such cases, you must separate the injection serial number (the digit(s) immediately after # or מספר) from the date that follows it, and extract ONLY the date portion as an intervention date. The serial number prefix (1, 2, 3, etc.) is NOT part of the date. For example:
    * "#114/02/22" → injection #1, date is "14/02/22"
    * "#214.3.22" → injection #2, date is "14.3.22"
    * "מספר 116/04/2020" → injection מספר 1, date is "16/04/2020"

    - The texts may also contain indication (אינדקציה , אינדיקציה , etc.) dates. Only if a date appears immediately adjacent to an indication keyword (e.g., "אינדקציה 22/11/21", "אינדיקציה 04/03/2020"), it should be labeled as other_dates. If a date appears without an adjacent indication keyword, it should be labeled as intervention_dates.
    - Return a JSON of all detected PII. The JSON must follow this schema:
    {response_schema.model_json_schema()}

    {FEW_SHOT_EXAMPLES}

    {OPHTHALMOLOGY_GLOSSARY}

    CRITICAL RULES:
    - Every extracted value MUST be an exact substring copied verbatim from the input text
    - Return ONLY the JSON object, without any additional text or explanation
    - Do NOT reformat dates, names, or numbers
    - Do NOT infer missing information
    - If you cannot copy an exact span, return null
    """

    return system_prompt


def _build_conversation(system_prompt: str, text: str) -> list[dict]:
    """Return the chat messages for one text; shared by both backends so they see the same prompt."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract the PII from the following clinical text:\n{text}"},
    ]


def _parse_response(response: str) -> dict:
    """Parse the model's JSON; keep the raw text when it is not valid JSON."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # The raw answer holds the extracted names/IDs, so it is never printed: stdout ends up in
        # tmux/nohup logs. main.py logs the row index instead.
        logger.warning(f"Failed to parse JSON response ({len(response)} characters, content not logged)")
        return {"raw_response": response}


def _truncated(reason: str) -> dict:
    """An answer we cannot trust because the input or the output was cut; handled like a parse failure."""
    logger.warning(f"Extraction skipped: {reason}")
    return {"raw_response": "", "error": reason}


def _extract_pii_ollama(texts: List[str], client, system_prompt: str) -> list[dict]:
    """
    Extract PII with an Ollama server, enforcing the same JSON schema as the vLLM path.

    Requests are sent in parallel (NUM_PARALLEL) because Ollama handles one request per call;
    `map` keeps the results in input order. Thinking is turned off so the reply is the JSON only.
    """
    import ollama

    ollama_config = config["OLLAMA_CONFIG"]
    schema = _response_schema()
    options = ollama_config.get("OPTIONS", {})
    num_ctx, num_predict = options.get("num_ctx"), options.get("num_predict", 0)

    def _extract_one(text: str) -> dict:
        try:
            response = client.chat(
                model=ollama_config["MODEL"],
                messages=_build_conversation(system_prompt, text),
                format=schema,
                options=ollama_config.get("OPTIONS", {}),
                think=False,
            )
        except ollama.ResponseError as error:
            # A failed request is treated like an unparsable answer, so one bad text does not stop the run.
            logger.warning(f"Ollama request failed: {error}")
            return {"raw_response": f"OLLAMA_ERROR: {error}"}
        # Ollama silently cuts prompts longer than num_ctx, so the end of a long note would never be
        # scanned. A prompt that fills the context up to the output budget may have been cut: treat it
        # as an error. Not a full guard: with prompt caching the reported count may be lower than the
        # real prompt (not verified), so a cut prompt can still pass. Keep num_ctx well above the prompt.
        prompt_tokens = getattr(response, "prompt_eval_count", None)
        if num_ctx and prompt_tokens and prompt_tokens >= num_ctx - num_predict:
            return _truncated(f"prompt of {prompt_tokens} tokens may exceed num_ctx {num_ctx}")
        if getattr(response, "done_reason", None) == "length":
            return _truncated("answer cut at num_predict")
        return _parse_response(response.message.content)

    with ThreadPoolExecutor(max_workers=ollama_config.get("NUM_PARALLEL", 1)) as executor:
        return list(executor.map(_extract_one, texts))


def extract_pii_from_text(texts: List[str], model, system_prompt: str) -> dict:
    """
    Extract personally identifiable information (PII) from a list of clinical texts using the configured backend.

    Parameters:
    - texts (List[str]): A list of clinical text entries from which to extract PII.
    - model: The loaded vLLM model instance, or an Ollama client (see load_model).
    - system_prompt (str): The system prompt string to be used as input for the model.

    Returns:
    - dict: A list of dictionaries containing the extracted PII for each text entry, following the schema defined in the MedicalInformation class.

    Note: if extraction fails, the dictionary will contain a "raw_response" key with the raw model output.
    """

    if config.get("BACKEND", "vllm") == "ollama":
        return _extract_pii_ollama(texts, model, system_prompt)

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.entrypoints.chat_utils import ChatTemplateResolutionError

    gen_config = config["VLLM_CONFIG"]["GENERATION_CONFIG"]
    max_tokens = gen_config.get("max_tokens", 1024)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=gen_config.get("temperature", 0),
        structured_outputs=StructuredOutputsParams(json=_response_schema()),
    )

    conversations = [_build_conversation(system_prompt, text) for text in texts]
    plain_prompts = [f"{system_prompt}\nExtract the PII from the following clinical text:\n{text}" for text in texts]

    # A prompt longer than the context would fail or be cut, and its PII would go unseen: such texts
    # are not sent and get the error marker instead.
    tokenizer = model.get_tokenizer()
    limit = config["VLLM_CONFIG"]["MODEL_CONFIG"].get("max_model_len", 4096) - max_tokens

    def _n_tokens(i: int) -> int:
        try:
            # return_dict=False: newer transformers return a dict here, whose len() is its number of keys.
            ids = tokenizer.apply_chat_template(conversations[i], tokenize=True, add_generation_prompt=True, return_dict=False)
            return len(ids["input_ids"] if isinstance(ids, dict) else ids)
        except Exception:  # no chat template: count the plain prompt used by the fallback below
            return len(tokenizer.encode(plain_prompts[i]))

    results = [None] * len(texts)
    send = []
    for i in range(len(texts)):
        n_tokens = _n_tokens(i)
        if n_tokens > limit:
            results[i] = _truncated(f"text {i}: prompt of {n_tokens} tokens exceeds the limit of {limit}")
        else:
            send.append(i)

    if send:
        try:
            outputs = model.chat(messages=[conversations[i] for i in send], sampling_params=sampling_params)
        except ChatTemplateResolutionError:
            # Fallback to model.generate() if chat() fails (e.g., due to schema issues)
            outputs = model.generate(prompts=[plain_prompts[i] for i in send], sampling_params=sampling_params)

        for i, output in zip(send, outputs):
            completion = output.outputs[0]
            # An answer cut at max_tokens is incomplete JSON at best: some PII would be missing.
            results[i] = (_truncated(f"text {i}: answer cut at max_tokens") if completion.finish_reason == "length"
                          else _parse_response(completion.text))

    return results


def remove_pii_from_text(text: str, extracted_pii: dict, skip_fields=DATE_FIELDS, visit_date=None, birthdate=None) -> str:
    """
    Remove personally identifiable information (PII) from a clinical text based on the extracted PII.

    Parameters:
    - text (str): The original clinical text from which to remove PII.
    - extracted_pii (dict): A dictionary containing the extracted PII, following the schema defined in the MedicalInformation class.
    - skip_fields: Fields that are left in the text. The default keeps dates; the evaluation passes ()
      to also redact them, as in the original experiments. Ignored for dates when DATE_MODE is visit_offset.
    - visit_date (datetime.date, optional): the row's visit date, needed by DATE_MODE: visit_offset.
    - birthdate (datetime.date, optional): the patient's birthdate; a date equal to it is removed, not offset.

    Returns:
    - str: The clinical text with PII replaced by placeholders (e.g., [REDACTED_PATIENT_NAME], [REDACTED_PATIENT_ID], etc.).

    Note: if the model failed to extract valid JSON, the function returns only "[ANONYMIZATION_ERROR]".
    """
    return deidentify_text(text, extracted_pii, skip_fields, visit_date, birthdate)[0]


def _values(value) -> list:
    """A field value as a list of non-empty items (fields hold a string, a number, a list or null)."""
    items = value if isinstance(value, list) else [value]
    return [item for item in items if item is not None and str(item).strip()]


def deidentify_text(text: str, extracted_pii: dict, skip_fields=DATE_FIELDS, visit_date=None, birthdate=None) -> tuple[str, list[str]]:
    """
    remove_pii_from_text, plus flags: names of fields whose extracted value matched nothing, backstop hits
    and date problems (never the values themselves, so the flags column holds no PII).

    With the default switches the text output is the same as before these switches existed (apart from
    the always-on fixes: error rows carry no text, and the age is matched as a whole number), so runs on
    the HY data stay reproducible. The switches (see config/rambam.yaml) add, in this order: date
    offsets, placeholder-safe matching of names, and the regex backstop. Dates go first because the LLM
    date spans are exact copies of the original text and must be matched before anything edits it.
    """
    # A failed extraction cannot be trusted to have found the PII. The original text used to be written
    # after the marker, which put a whole un-redacted note into the _deidentified column.
    if "raw_response" in extracted_pii:
        return "[ANONYMIZATION_ERROR]", ["anonymization_error"]

    flags = []
    deidentified_text = text
    if config.get("DATE_MODE", "keep") == "visit_offset":
        date_spans = [str(v) for key in ("intervention_dates", "other_dates") for v in _values(extracted_pii.get(key))]
        dob_spans = [str(v) for v in _values(extracted_pii.get("patient_date_of_birth"))]
        deidentified_text, date_flags = shift_dates(deidentified_text, date_spans, dob_spans, visit_date, birthdate)
        flags += date_flags
        skip_fields = DATE_FIELDS  # already handled

    safe = config.get("SAFE_TEXT_MATCHING", False)
    pairs = [(key, item) for key, value in extracted_pii.items() if key not in skip_fields for item in _values(value)]
    if safe:
        # Longest first, so "אבי כהן" is replaced before "כהן" and no half-name is left.
        pairs.sort(key=lambda pair: len(str(pair[1])), reverse=True)

    for key, value in pairs:
        tag = f"[REDACTED_{key.upper()}]"
        # The age only as a whole number (plain replace also hit it inside VA, IOP and dates). Everything else
        # literally: re.escape + subn is exactly str.replace. The ID never gets word boundaries: it rarely
        # collides, and a missed ID is worse than a damaged number.
        pattern = age_pattern(value) if key == "patient_age" else re.compile(re.escape(str(value)))
        if not safe:
            deidentified_text, n = pattern.subn(tag, deidentified_text)
        elif key in ("patient_age", "patient_id"):
            deidentified_text, n = sub_outside_placeholders(pattern, tag, deidentified_text)
        else:
            deidentified_text, n = replace_text_value(deidentified_text, str(value), tag)
        if n == 0:
            flags.append(f"unmatched:{key}")

    if config.get("REGEX_BACKSTOP", False):
        deidentified_text, backstop_flags = apply_regex_backstop(deidentified_text)
        flags += backstop_flags

    return deidentified_text, flags



# ---------------------------------------------------------------------------
# Pipeline input/output helpers (used by main.py)
# ---------------------------------------------------------------------------

AGE_TOP_CODE = 90  # HIPAA: ages over 89 are identifiers, so every age >= 90 is written as 90


def save_extractions(path: Path, rows: list[dict]) -> None:
    """Write the model's answers, one JSON row per note. Holds the PII itself: internal only, never shared."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_input(data_path: Path, columns: list[str]) -> pd.DataFrame:
    """The input as a DataFrame: a CSV as is, or a TXT file as one record per non-empty line."""
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix == ".txt":
        if len(columns) != 1:
            raise ValueError("TXT input requires exactly one configured column")
        return pd.DataFrame({columns[0]: load_data(str(data_path), columns[0])})
    raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")


FILL_ME = "FILL_ME"  # placeholder in config/rambam.yaml for values only the site running it knows


def validate_config(cfg: dict, df: pd.DataFrame = None) -> None:
    """
    Stop before the model loads if the config does not fit the data.

    Called twice by main.py: without `df` before the input is read (a FILL_ME left in DATA_PATH would
    otherwise give a confusing file error), and with `df` once it is read (column names need the data),
    always before load_model, which takes minutes on the hospital machine.
    """
    unfilled = [key for key, value in cfg.items()
                if value == FILL_ME or (isinstance(value, list) and FILL_ME in value)]
    if unfilled:
        raise ValueError(f"Fill in these config values first (still {FILL_ME}): {unfilled}")
    if df is None:
        return
    text_columns = cfg.get("COLUMNS_TO_DEIDENTIFY") or []
    if not text_columns:
        raise ValueError("COLUMNS_TO_DEIDENTIFY is empty")
    needed = list(text_columns)
    if cfg.get("DATE_MODE", "keep") == "visit_offset":
        if not cfg.get("VISIT_DATE_COLUMN"):
            raise ValueError("DATE_MODE: visit_offset needs VISIT_DATE_COLUMN")
        needed.append(cfg["VISIT_DATE_COLUMN"])
    if cfg.get("BIRTHDATE_COLUMN"):
        needed.append(cfg["BIRTHDATE_COLUMN"])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the data: {missing}. Columns in the file: {list(df.columns)}")
    created = {f"{c}_deidentified" for c in text_columns} | {f"{c}_deid_flags" for c in text_columns}
    unknown = [c for c in cfg.get("SHARE_COLUMNS") or [] if c not in df.columns and c not in created]
    if unknown:
        raise ValueError(f"SHARE_COLUMNS not found in the data or the output: {unknown}")


def parse_dates(values: pd.Series, date_format) -> pd.Series:
    """
    Parse a structured date column, failing loudly on values that cannot be read.

    Without a format, day first (Israeli notation). pandas' own guess reads "05/02/2024" as 2 May and
    turns "20/02/2024" into nothing, which would silently shift every text-date offset in those rows.
    ISO values (2024-02-05) are parsed apart: with dayfirst=True pandas would swap their month and day too.
    """
    if date_format:
        dates = pd.to_datetime(values, errors="coerce", format=date_format)
    else:
        text = values.astype("string").str.strip()
        iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").fillna(False).astype(bool)
        dates = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        dates[iso] = pd.to_datetime(text[iso].str[:10], errors="coerce", format="%Y-%m-%d")
        dates[~iso] = pd.to_datetime(text[~iso], errors="coerce", format="mixed", dayfirst=True)
    unread = values.notna() & values.astype(str).str.strip().ne("") & dates.isna()
    if unread.any():
        raise ValueError(f"{int(unread.sum())} values of {values.name!r} could not be read as dates (rows "
                         f"{list(values.index[unread][:10])}); set the column's format in the config")
    return dates


def date_column(df: pd.DataFrame, column, date_format=None):
    """A structured date column as datetime.date per row (None where empty), or None if no column is set."""
    if not column:
        return None
    dates = parse_dates(df[column], date_format)
    return dates.map(lambda d: None if pd.isna(d) else d.date())


def age_at_visit(birthdates: pd.Series, visits: pd.Series, birthdate_format=None, visit_format=None) -> pd.Series:
    """Whole years between birth and visit, top-coded at 90; empty when a date is missing or impossible."""
    born = parse_dates(birthdates, birthdate_format)
    seen = parse_dates(visits, visit_format)
    had_birthday = (seen.dt.month > born.dt.month) | ((seen.dt.month == born.dt.month) & (seen.dt.day >= born.dt.day))
    age = (seen.dt.year - born.dt.year - (~had_birthday).astype(int)).astype("Int64")
    age = age.mask(age < 0)  # a birthdate after the visit is a data error, not an age
    return age.clip(upper=AGE_TOP_CODE)


def write_shareable(df: pd.DataFrame, output_path: Path, cfg: dict) -> None:
    """
    Write only the allowlisted columns (SHARE_COLUMNS): a column nobody listed never leaves.

    Raw text columns and the birthdate cannot be listed. The birthdate becomes age_at_visit, and the
    AGE_COLUMNS are top-coded, because exact ages over 89 are HIPAA identifiers. `cfg` is the run's config
    (passed in, not read from the global, so callers and tests control it).
    """
    columns = cfg.get("SHARE_COLUMNS") or []
    if not columns:
        return
    birthdate_column = cfg.get("BIRTHDATE_COLUMN")
    forbidden = [c for c in columns if c in cfg["COLUMNS_TO_DEIDENTIFY"] or c == birthdate_column]
    if forbidden:
        raise ValueError(f"SHARE_COLUMNS must not contain raw text or birthdate columns: {forbidden}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"SHARE_COLUMNS not found in the data: {missing}")

    shared = df[columns].copy()
    for column in cfg.get("AGE_COLUMNS") or []:
        if column in shared:
            shared[column] = pd.to_numeric(shared[column], errors="coerce").clip(upper=AGE_TOP_CODE).astype("Int64")
    if birthdate_column:
        shared["age_at_visit"] = age_at_visit(df[birthdate_column], df[cfg["VISIT_DATE_COLUMN"]],
                                              cfg.get("BIRTHDATE_FORMAT"), cfg.get("VISIT_DATE_FORMAT"))
    shared.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Shareable file (allowlisted columns only): {output_path}")


def text_hash(text: str) -> str:
    """SHA-256 of a note, stored with its saved extraction so reuse can check the pairing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_extractions(path: Path) -> dict:
    """(column, row) -> (text hash, extraction), from a JSONL written by an earlier run."""
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["column"], str(r["row"])): (r.get("text_sha256"), r["extraction"]) for r in rows}


def reused_extraction(reused: dict, column: str, index, text: str) -> dict:
    """
    The saved extraction of this row, only if it was made from this exact text.

    Rows are keyed by position, so a re-exported or reordered CSV would pair a note with another note's
    extraction and leave its own names in the output. A missing or different hash counts as a failure.
    """
    saved_hash, extraction = reused.get((column, str(index)), (None, None))
    if extraction is None or saved_hash != text_hash(text):
        return {"raw_response": ""}
    return extraction
