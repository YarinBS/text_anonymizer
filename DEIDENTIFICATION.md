# De-identification method

This page describes what the pipeline removes from clinical free text when it runs with `config/rambam.yaml`, what it keeps, and the risks that remain. It is written for the people who review the output before it leaves the hospital.

## How it works

Each note goes through two stages, both on the local machine. No text is sent to any external service.

1. **A local language model** reads the note and returns the personal details it finds (names, IDs, ages, dates, places) as structured JSON. Every field must be present in the answer. If the answer cannot be parsed or is cut off, or the note is too long for the model, the whole note is replaced by `[ANONYMIZATION_ERROR]`, and none of its text is kept. (With the Ollama backend the length check is partial; see Known limits.)
2. **Fixed rules** then replace what the model found, and also catch formats the model may miss.

## What is replaced

| Identifier | Replaced by | Found by |
|---|---|---|
| Patient names (every form), relatives' and other names | `[REDACTED_PATIENT_NAME]`, `[REDACTED_OTHER_MENTIONED_NAMES]` | model |
| Doctors' names | `[REDACTED_DOCTOR_NAME]` | model |
| ID numbers | `[REDACTED_PATIENT_ID]` | model, plus any 8–9 digit number with a valid Israeli ID check digit |
| Ages (any value) | `[REDACTED_PATIENT_AGE]` | model, plus "בן / בת / גיל / age" followed by a number |
| Places smaller than a country (city, street, ZIP) | `[REDACTED_LOCATIONS]` | model |
| Phone numbers | `[REDACTED_PHONE]` | rules (Israeli mobile, landline, +972) |
| Email addresses, URLs | `[REDACTED_EMAIL]`, `[REDACTED_URL]` | rules |
| Date of birth written in the text | `[REDACTED_DATE_OF_BIRTH]` | model |
| Other dates | days, months or years from the visit date (see below) | model, plus rules for full dates |

Short names (3 letters or less) are replaced only as whole words, so that they are not cut out of other words. If such a name appears nowhere as a whole word, it is replaced anyway: a damaged word is accepted to avoid a leak. Nothing is ever replaced inside an existing placeholder.

## Dates

A date in the text is replaced by its distance from the visit date of the same row:

| Written | Becomes |
|---|---|
| a full date (e.g. 01/01/2024, 2024-01-01, 1 בינואר 2024) | `[DATE_DAY_-35]`: 35 days before the visit |
| month and year (03/2024) | `[DATE_MONTH_-2]` |
| a year alone, when the model marks it as a date | `[DATE_YEAR_-3]` |
| a date without a year, an invalid date, or any date in a row without a visit date | `[REDACTED_DATE]` |
| a date equal to the patient's birthdate (when `BIRTHDATE_COLUMN` is set) | `[REDACTED_DATE_OF_BIRTH]` |
| a date more than 89 years before the visit (a likely date of birth) | `[REDACTED_DATE]` |

Every date that is found is written this way, so no calendar date is kept for it, and the distance from the visit reveals no age. The structured visit-date column itself stays real: the research team needs it to match notes with the patient's scans. That column must therefore be handled as identifying data wherever the output goes. Two-digit years are read as 20yy, unless that is more than 10 years after the visit; then 19yy. Numbers such as visual acuity (6/9, 0.5) and eye pressure (21/18) are not treated as dates unless the model marks them as dates.

## Output files

| File | Contains | May leave the hospital |
|---|---|---|
| `<input>_shareable.csv` | only the columns listed in `SHARE_COLUMNS`. Raw text and birthdate columns are refused; the birthdate becomes `age_at_visit`, with every age of 90 or more written as 90. | yes, after review |
| `<input>_deidentified_final.csv` | the original columns **and** the de-identified ones | **no** |
| `<input>_extractions.jsonl` | the personal details the model found | **no** |

The `<column>_deid_flags` column lists, per row, the names of fields whose value could not be found in the text, rule hits, and date problems. It never lists the values themselves. Log files hold row numbers only, never text or model answers.

## Known limits

- **The model can miss details.** A name with no title in front of it, an unusual spelling, or a detail in an unusual place may not be found. The fixed rules only cover fixed formats (phones, emails, URLs, valid IDs, ages after "בן/בת/גיל", full dates).
- **The patient ID column** in `SHARE_COLUMNS` must already be a hashed or study ID, never a national ID or medical record number.
- **Years alone** that the model does not mark as dates stay in the text. HIPAA Safe Harbor permits years.
- **The real visit date** stays in the structured column (see Dates).
- **A date of birth the model misses**, for a patient under 90, becomes an ordinary day offset, which reveals the age, unless `BIRTHDATE_COLUMN` is set (then any date equal to the birthdate is removed). Set it whenever the data has a birthdate.
- **Ollama backend:** a note longer than `num_ctx` may be cut silently and only partly scanned. The check on the reported prompt length may miss this. Keep `num_ctx` well above the longest prompt.

## How to check an output file

```bash
python scripts/audit_output.py <input>_shareable.csv
```

It prints, per check, how many cells still contain a possible phone number, email, URL, valid ID, age phrase, full date, or a doctor title followed by a word. It prints counts only, never text, so its output can be shared. Any count above 0 should be looked at by a reviewer before the file is released. The model's recall can be measured with `src/evaluate.py` (see `EVALUATION.md`).
