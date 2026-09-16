"""
utils.py

This module contains utility functions for loading the model, loading data, 
extracting personally identifiable information (PII) from clinical texts, and removing PII from texts.
It also defines the schema for the extracted medical information using Pydantic models.
"""

import json
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.entrypoints.chat_utils import ChatTemplateResolutionError

from config import config
from logger import get_logger


logger = get_logger()


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


def load_model(model_path: str) -> LLM:
    """
    Load a vLLM model from a specified path.

    Parameters:
    - model_path (str): The path to the model file.

    Returns:
    - LLM: The loaded vLLM model instance.
    """

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


def get_system_prompt(response_schema: BaseModel = MedicalInformation) -> str:
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


def extract_pii_from_text(texts: List[str], model: LLM, system_prompt: str) -> dict:
    """
    Extract personally identifiable information (PII) from a list of clinical texts using a vLLM model.

    Parameters:
    - texts (List[str]): A list of clinical text entries from which to extract PII.
    - model (LLM): The loaded vLLM model instance to use for extraction.
    - system_prompt (str): The system prompt string to be used as input for the model.

    Returns:
    - dict: A list of dictionaries containing the extracted PII for each text entry, following the schema defined in the MedicalInformation class.

    Note: if extraction fails, the dictionary will contain a "raw_response" key with the raw model output.
    """

    def _get_sampling_params() -> SamplingParams:
        """Build vLLM sampling parameters with JSON schema enforcement"""
        gen_config = config["VLLM_CONFIG"]["GENERATION_CONFIG"]

        structured_outputs = StructuredOutputsParams(
            json=MedicalInformation.model_json_schema(),
        )

        return SamplingParams(
            max_tokens=gen_config.get("max_tokens", 1024),
            temperature=gen_config.get("temperature", 0),
            structured_outputs=structured_outputs,
        )

    results = []

    sampling_params = _get_sampling_params()
    try:
        # Build all conversations upfront
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the PII from the following clinical text:\n{text}"},
            ]
            for text in texts
        ]

        outputs = model.chat(
            messages=conversations,
            sampling_params=sampling_params,
        )
    
    except ChatTemplateResolutionError:
        # Fallback to model.generate() if chat() fails (e.g., due to schema issues)
        prompts = [f"{system_prompt}\nExtract the PII from the following clinical text:\n{text}" for text in texts]
        outputs = model.generate(
            prompts=prompts,
            sampling_params=sampling_params
        )

    results = []
    for output in outputs:
        response = output.outputs[0].text
        try:
            results.append(json.loads(response))
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Raw response:")
            print(response)
            results.append({"raw_response": response})

    return results


def remove_pii_from_text(text: str, extracted_pii: dict) -> str:
    """
    Remove personally identifiable information (PII) from a clinical text based on the extracted PII.

    Parameters:
    - text (str): The original clinical text from which to remove PII.
    - extracted_pii (dict): A dictionary containing the extracted PII, following the schema defined in the MedicalInformation class.

    Returns:
    - str: The clinical text with PII replaced by placeholders (e.g., [REDACTED_PATIENT_NAME], [REDACTED_PATIENT_ID], etc.).

    Note: if the model failed to extract valid JSON, the function returns the original text prefixed with "[ANONYMIZATION_ERROR]".
    """

    # If the model failed to extract valid JSON, we cannot reliably remove PII, so return the original text
    if "raw_response" in extracted_pii:
        return f"[ANONYMIZATION_ERROR] {text}"

    deidentified_text = text
    for key, value in extracted_pii.items():
        if key in ["intervention_dates", "other_dates", "patient_date_of_birth"]:  # Skip fields that are not relevant for de-identification
            continue
        if isinstance(value, list):
            for item in value:
                if item is not None:
                    deidentified_text = deidentified_text.replace(str(item), f"[REDACTED_{key.upper()}]")
        elif value is not None:
            deidentified_text = deidentified_text.replace(str(value), f"[REDACTED_{key.upper()}]")
    
    return deidentified_text


def add_deidentified_texts_to_df(df, deidentified_texts_paths, output_path):
    """
    Add deidentified texts to a DataFrame as new columns.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - deidentified_texts_paths (list): The list of paths to the .txt files containing the deidentified texts to add.
    - output_path (str): The file path where the updated CSV will be saved.

    Returns:
    - pd.DataFrame: The updated DataFrame with the new deidentified text columns.
    """
    
    for file_path in deidentified_texts_paths:
        # Find which column this file corresponds to
        column_name = next((col for col in config["COLUMNS_TO_DEIDENTIFY"] if col in file_path), None)
        
        if not column_name:
            logger.warning(f"Warning: Could not find matching column for file {file_path}. Skipping.")
            continue

        deidentified_column_name = f"{column_name}_deidentified"
        
        with open(file_path, "r", encoding="utf-8") as f:
            deidentified_texts = [line.strip() for line in f.readlines()]
            
        # Find rows matching the filter used in load_data (not null and contains at least one letter)
        valid_mask = df[column_name].notna() & df[column_name].apply(lambda s: isinstance(s, str) and any(c.isalpha() for c in s))
        
        if valid_mask.sum() != len(deidentified_texts):
            logger.warning(f"Warning: {valid_mask.sum()} non-empty rows in {column_name}, but {len(deidentified_texts)} lines in txt file.")
            
        df[deidentified_column_name] = ""
        df.loc[valid_mask, deidentified_column_name] = deidentified_texts

    # Finally, save the df
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    return df


if __name__ == "__main__":
    print("Hello from utils.py!")

    # from huggingface_hub import snapshot_download

    # snapshot_download(
    #     repo_id="meta-llama/Llama-3.3-70B-Instruct",
    #     local_dir="/homes/joachim_lab_users/yarinbs/PythonProjects/models/Llama-3.3-70B-Instruct",
    #     local_dir_use_symlinks=False  # Set to False to download actual files instead of symlinks
    # )
