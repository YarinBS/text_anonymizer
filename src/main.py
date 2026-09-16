"""
main.py

This script is the main entry point for the text de-identification process. This script:
1. Loads the model
2. Processes the specified columns in the input CSV file - extracts PII and replaces it
3. Replaces extracted PII while preserving the input row alignment
4. Saves the original and deidentified columns to a new CSV file.
"""

from pathlib import Path

import pandas as pd

from config import config
from logger import get_logger
from utils import (extract_pii_from_text,
                   get_system_prompt,
                   get_valid_text_records,
                   load_data,
                   load_model, 
                   remove_pii_from_text)


logger = get_logger()


def main():
    data_path = Path(config["DATA_PATH"])
    input_suffix = data_path.suffix.lower()

    if input_suffix == ".csv":
        df = pd.read_csv(data_path)
    elif input_suffix == ".txt":
        columns = config["COLUMNS_TO_DEIDENTIFY"]
        if len(columns) != 1:
            raise ValueError("TXT input requires exactly one configured column")
        df = pd.DataFrame({columns[0]: load_data(str(data_path), columns[0])})
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")

    model = load_model(model_path=config["MODEL_PATH"])
    logger.info("Model loaded successfully")

    system_prompt = get_system_prompt()
    
    for column_name in config["COLUMNS_TO_DEIDENTIFY"]:
        logger.info(f"Processing column: {column_name}")
        records = get_valid_text_records(df, column_name)
        indices = [index for index, _ in records]
        data = [text for _, text in records]

        logger.info("Extracting PII from text...")
        extracted_piis = extract_pii_from_text(
            texts=data,
            model=model,
            system_prompt=system_prompt
        )

        if len(data) != len(extracted_piis):
            raise RuntimeError(
                f"Model returned {len(extracted_piis)} results for {len(data)} "
                f"input texts in column {column_name!r}"
            )

        logger.info("Removing PII from text...")
        # vLLM preserves batch order; the length check above makes positional pairing safe.
        deidentified_texts = [remove_pii_from_text(text, extracted_pii) for text, extracted_pii in zip(data, extracted_piis)]

        deidentified_column = f"{column_name}_deidentified"
        df[deidentified_column] = ""
        df.loc[indices, deidentified_column] = deidentified_texts

    output_path = data_path.with_name(f"{data_path.stem}_deidentified_final.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Done!")
    

if __name__ == "__main__":
    main()
