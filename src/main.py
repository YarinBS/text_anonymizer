"""
main.py

Entry point of the de-identification pipeline. For every configured text column it:
1. extracts the PII of each note with the model (or reuses saved extractions),
2. replaces it (utils.deidentify_text), keeping the input row order,
3. writes <input>_deidentified_final.csv, and optionally the extractions (SAVE_EXTRACTIONS) and an
   allowlisted <input>_shareable.csv (SHARE_COLUMNS).

Helpers live in utils.py; this file holds only the pipeline flow. See README.md for the switches.

Usage: python src/main.py [--config config/rambam.yaml] [--data-path <file>] [--reuse-extractions <file>.jsonl]
"""

import argparse
import sys
from pathlib import Path

import config as config_module
from config import config
from logger import get_logger
from utils import (date_column,
                   deidentify_text,
                   extract_pii_from_text,
                   get_system_prompt,
                   get_valid_text_records,
                   load_extractions,
                   load_input,
                   load_model,
                   reused_extraction,
                   save_extractions,
                   text_hash,
                   validate_config,
                   write_shareable)


logger = get_logger()


def parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="De-identify clinical free text with a local LLM.")
    parser.add_argument("--config", default=None, help="config file (default: config/config.yaml)")
    parser.add_argument("--data-path", default=None, help="input file; overrides DATA_PATH of the config")
    parser.add_argument("--reuse-extractions", type=Path, default=None,
                        help="JSONL written by an earlier run (SAVE_EXTRACTIONS): skip the LLM, only re-apply replacement")
    return parser.parse_args(argv)


def main(argv=()):
    args = parse_args(list(argv))
    if args.config:
        config_module.load_config(args.config)

    validate_config(config)
    data_path = Path(args.data_path or config["DATA_PATH"])
    df = load_input(data_path, config["COLUMNS_TO_DEIDENTIFY"])
    validate_config(config, df)

    reused = load_extractions(args.reuse_extractions) if args.reuse_extractions else None
    model = None
    if reused is None:
        model = load_model(model_path=config["MODEL_PATH"])
        logger.info("Model loaded successfully")
    system_prompt = get_system_prompt()
    visit_dates = date_column(df, config.get("VISIT_DATE_COLUMN"), config.get("VISIT_DATE_FORMAT"))
    birthdates = date_column(df, config.get("BIRTHDATE_COLUMN"), config.get("BIRTHDATE_FORMAT"))
    saved_extractions = []

    for column_name in config["COLUMNS_TO_DEIDENTIFY"]:
        logger.info(f"Processing column: {column_name}")
        records = get_valid_text_records(df, column_name)
        indices = [index for index, _ in records]
        data = [text for _, text in records]

        if reused is not None:
            # A row without a saved extraction is treated as failed, never as "no PII".
            extracted_piis = [reused_extraction(reused, column_name, i, text) for i, text in zip(indices, data)]
        else:
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

        failed = [index for index, pii in zip(indices, extracted_piis) if "raw_response" in pii]
        if failed:
            # Row indices only: the texts and the raw answers contain PII.
            logger.warning(f"{len(failed)} rows in {column_name!r} failed and were replaced by [ANONYMIZATION_ERROR]: rows {failed}")

        logger.info("Removing PII from text...")
        # vLLM preserves batch order; the length check above makes positional pairing safe.
        results = [
            deidentify_text(text, extracted_pii,
                            visit_date=None if visit_dates is None else visit_dates.loc[index],
                            birthdate=None if birthdates is None else birthdates.loc[index])
            for index, text, extracted_pii in zip(indices, data, extracted_piis)
        ]

        deidentified_column = f"{column_name}_deidentified"
        df[deidentified_column] = ""
        df.loc[indices, deidentified_column] = [text for text, _ in results]
        if config.get("FLAGS_COLUMN", False):
            df[f"{column_name}_deid_flags"] = ""
            df.loc[indices, f"{column_name}_deid_flags"] = [";".join(flags) for _, flags in results]
        saved_extractions += [{"column": column_name, "row": index, "text_sha256": text_hash(text), "extraction": pii}
                              for index, text, pii in zip(indices, data, extracted_piis)]

    output_path = data_path.with_name(f"{data_path.stem}_deidentified_final.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    if config.get("SAVE_EXTRACTIONS", False) and reused is None:
        extractions_path = data_path.with_name(f"{data_path.stem}_extractions.jsonl")
        save_extractions(extractions_path, saved_extractions)
        logger.info(f"Raw extractions (internal, contains PII): {extractions_path}")

    write_shareable(df, data_path.with_name(f"{data_path.stem}_shareable.csv"), config)
    logger.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
