"""
audit_output.py

Count signs of PII left in de-identified columns. Prints COUNTS ONLY, never text, so it can be run by
hospital staff on any output file and the summary can be shared.

Usage: python scripts/audit_output.py <file.csv> [<file.csv> ...]
Checks every column whose name ends in "_deidentified".
"""

import re
import sys

import pandas as pd

sys.path.insert(0, "src")
from date_shift import FULL_DATE_IN_TEXT, ISO_DATE_IN_TEXT  # noqa: E402
from redaction import AGE_PHRASE, EMAIL, ID_CANDIDATE, PHONE, PLACEHOLDER, URL, _valid_israeli_id  # noqa: E402

CHECKS = {
    "phone number": PHONE,
    "email": EMAIL,
    "url": URL,
    "valid Israeli ID": ID_CANDIDATE,  # counted only when the check digit is valid (see below)
    "age phrase (בן/בת/גיל + number)": AGE_PHRASE,
    "full date d/m/y": FULL_DATE_IN_TEXT,  # the pipeline's own date patterns, so audit and pipeline agree
    "ISO date y-m-d": ISO_DATE_IN_TEXT,
    "doctor title + word": re.compile(r"(?:ד\"ר|ד״ר|דר'|פרופ'?|\bDr\.?)\s*[א-תA-Z][א-תa-z]+"),
    "error rows": re.compile(r"^\[ANONYMIZATION_ERROR\]"),
    "broken placeholder": re.compile(r"\[[A-Z_]*\[|\][A-Z_]+\]"),
}


def audit(path: str) -> None:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    columns = [c for c in df.columns if c.endswith("_deidentified")]
    texts = [v for c in columns for v in df[c] if isinstance(v, str) and v.strip()]
    print(f"\n{path}: {len(texts)} non-empty cells in {len(columns)} column(s)")
    for name, pattern in CHECKS.items():
        cells = 0
        for text in texts:
            # Error rows and broken placeholders are checked on the raw text. Elsewhere placeholders become
            # a NUL character, which no pattern matches (not even \s), so 'ד"ר [REDACTED_DOCTOR_NAME] בדק'
            # is no false alarm.
            outside = text if name in ("error rows", "broken placeholder") else PLACEHOLDER.sub("\x00", text)
            matches = [m.group() for m in pattern.finditer(outside)]
            if name == "valid Israeli ID":
                matches = [m for m in matches if _valid_israeli_id(m)]
            cells += bool(matches)
        print(f"  {name:36s} {cells} cells")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for file_path in sys.argv[1:]:
        audit(file_path)
