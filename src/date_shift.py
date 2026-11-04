"""
date_shift.py

Replace dates in free text by their distance from the row's real visit date (DATE_MODE: visit_offset).

Why relative to the visit date: the structured visit-date column stays real (the lab pairs notes with
scans by it), so the text itself carries no absolute date and no age, yet every text date can be rebuilt
exactly at the lab as visit_date + offset. Anchoring on the date of birth was rejected because the day
count would reveal the exact age (also over 89); a random per-patient shift was rejected because, with
the real visit date in the same row, the offset could be read off any note that writes its own date.
"""

import datetime as dt
import re
from typing import Optional

from redaction import sub_outside_placeholders

MONTHS = {
    "ינואר": 1, "פברואר": 2, "מרץ": 3, "מרס": 3, "אפריל": 4, "מאי": 5, "יוני": 6, "יולי": 7,
    "אוגוסט": 8, "ספטמבר": 9, "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_NAMES = "|".join(sorted(MONTHS, key=len, reverse=True))  # longest first, so "sept" beats "sep"

# Exact forms of one extracted span (the LLM copies the date verbatim).
# The same separator twice: mixed forms such as "+1.25-0.75" (refraction) or "12-10.50" are not dates.
_FULL = re.compile(r"(\d{1,2})([./-])(\d{1,2})\2(\d{4}|\d{2})")
_ISO = re.compile(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})")  # 2024-02-05, as some EMRs write dates
_NAMED_FULL = re.compile(rf"(\d{{1,2}})\s*(?:ב|ל)?-?\s*({_MONTH_NAMES})\.?\s*,?\s*(\d{{4}})", re.IGNORECASE)
_MONTH_YEAR = re.compile(r"(\d{1,2})[./-](\d{4})")
_NAMED_MONTH_YEAR = re.compile(rf"(?:ב|ל)?({_MONTH_NAMES})\.?\s*,?\s*(\d{{4}})", re.IGNORECASE)
_YEAR = re.compile(r"(19\d\d|20\d\d)")

# Backstop for dates the LLM missed: FULL dates only. Two-part shapes (6/9, 0.5, 21/18) are mostly
# visual acuity and IOP values in eye notes, so they are only touched when the LLM extracted them.
FULL_DATE_IN_TEXT = re.compile(r"(?<![\d./-])\d{1,2}([./-])\d{1,2}\1(?:\d{4}|\d{2})(?!\d)(?![./-]\d)")
ISO_DATE_IN_TEXT = re.compile(r"(?<![\d./-])(?:19|20)\d\d([-/.])\d{1,2}\1\d{1,2}(?!\d)(?![./-]\d)")
_NAMED_FULL_IN_TEXT = re.compile(
    rf"(?<!\d)\d{{1,2}}\s*(?:ב|ל)?-?\s*(?:{_MONTH_NAMES})\.?\s*,?\s*\d{{4}}(?!\d)", re.IGNORECASE
)

REDACTED_DATE = "[REDACTED_DATE]"
REDACTED_DOB = "[REDACTED_DATE_OF_BIRTH]"
MAX_AGE_YEARS = 89  # HIPAA: ages over 89 are identifiers; a date this far back is likely a date of birth


def _year(text: str, visit: dt.date) -> int:
    """
    Two-digit years: 20yy, unless that lies more than 10 years after the visit, then 19yy.
    The margin keeps follow-up dates ("ביקורת 15/01/25" written in December 2024) in the future.
    """
    year = int(text)
    if len(text) == 2:
        year += 2000 if 2000 + year <= visit.year + 10 else 1900
    return year


def _parse(span: str, visit: dt.date):
    """(kind, value) of a date span: ("day", date), ("month", (year, month)), ("year", year) or (None, None)."""
    if m := _FULL.fullmatch(span):
        return "day", dt.date(_year(m.group(4), visit), int(m.group(3)), int(m.group(1)))  # day first (Israel)
    if m := _ISO.fullmatch(span):
        return "day", dt.date(int(m.group(1)), int(m.group(3)), int(m.group(4)))
    if m := _NAMED_FULL.fullmatch(span):
        return "day", dt.date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    if m := _MONTH_YEAR.fullmatch(span) or _NAMED_MONTH_YEAR.fullmatch(span):
        month, year = m.groups()
        month = int(month) if month.isdigit() else MONTHS[month.lower()]
        return ("month", (int(year), month)) if 1 <= month <= 12 else (None, None)
    if m := _YEAR.fullmatch(span):
        return "year", int(m.group(1))
    return None, None


def to_offset_token(span: str, visit: Optional[dt.date], birthdate: Optional[dt.date] = None) -> str:
    """
    Turn one date span into its offset from the visit date.

    full date -> [DATE_DAY_n], month/year -> [DATE_MONTH_n], year alone -> [DATE_YEAR_n]. Anything that
    cannot be placed exactly (invalid day, day/month with no year, no visit date) becomes
    [REDACTED_DATE]: a guessed offset would be wrong data, a kept date would be a leak.

    A date of birth the LLM missed or labelled as another date would, as an offset, give the exact age.
    So a date equal to the known birthdate (or in its birth month/year) is removed, and so is any date
    more than 89 years before the visit (the likely birthdate of a patient over 89).
    """
    span = span.strip()
    if visit is None:
        return REDACTED_DATE
    try:
        kind, value = _parse(span, visit)
    except ValueError:  # e.g. 31/02/2024
        return REDACTED_DATE
    if kind == "day":
        if birthdate is not None and value == birthdate:
            return REDACTED_DOB
        if visit.year - value.year > MAX_AGE_YEARS:
            return REDACTED_DATE
        return f"[DATE_DAY_{(value - visit).days}]"
    if kind == "month":
        year, month = value
        if (birthdate is not None and (year, month) == (birthdate.year, birthdate.month)) or visit.year - year > MAX_AGE_YEARS:
            return REDACTED_DATE
        return f"[DATE_MONTH_{(year * 12 + month) - (visit.year * 12 + visit.month)}]"
    if kind == "year":
        if (birthdate is not None and value == birthdate.year) or visit.year - value > MAX_AGE_YEARS:
            return REDACTED_DATE
        return f"[DATE_YEAR_{value - visit.year}]"
    return REDACTED_DATE


def _is_full(span: str) -> bool:
    return bool(_FULL.fullmatch(span) or _ISO.fullmatch(span) or _NAMED_FULL.fullmatch(span))


def shift_dates(text: str, date_spans: list[str], dob_spans: list[str], visit: Optional[dt.date],
                birthdate: Optional[dt.date] = None) -> tuple[str, list[str]]:
    """
    Replace every date in the text by its offset from the visit date; returns (text, flags).

    Runs before every other replacement: the LLM spans are exact copies of the original text, so they
    must be matched before anything else edits it. A written date of birth is removed completely
    ([REDACTED_DATE_OF_BIRTH]); as an offset it would give the exact age.

    Order: full-date spans (and the DOB) first, then the full-date backstop, then the partial spans
    (month/year, year, day/month). A partial span such as "2019" must not cut into a full date the
    LLM missed ("05/06/2019" would become "05/06/[DATE_YEAR_-5]" and keep its day and month).
    """
    flags = [] if visit is not None else ["no_visit_date"]
    spans = [(s.strip(), REDACTED_DOB) for s in dob_spans if s and s.strip()]
    spans += [(s.strip(), None) for s in date_spans if s and s.strip()]
    full = [job for job in spans if job[1] or _is_full(job[0])]
    partial = [job for job in spans if job not in full]

    def _replace(jobs, left_boundary: str):
        nonlocal text
        # Longest first, so "14/02/2022" is not cut by a shorter span "14/02".
        for span, token in sorted(jobs, key=lambda job: len(job[0]), reverse=True):
            token = token or to_offset_token(span, visit, birthdate)
            # Never followed by another digit: the span "2/3" must not hit inside "IOP 12/30".
            pattern = re.compile(left_boundary + re.escape(span) + r"(?!\d)(?![./-]\d)")
            text, n = sub_outside_placeholders(pattern, token, text)
            if n == 0:
                flags.append("unmatched:date")

    # Full dates may have a digit glued on the left (a merged injection serial: "#114/02/22").
    _replace(full, "")
    for pattern in (_NAMED_FULL_IN_TEXT, ISO_DATE_IN_TEXT, FULL_DATE_IN_TEXT):
        text, n = sub_outside_placeholders(pattern, lambda m: to_offset_token(m.group(), visit, birthdate), text)
        if n:
            flags.append("backstop:date")
    # Partial spans are short and ambiguous, so they must stand alone on both sides.
    _replace(partial, r"(?<![\d./-])")
    return text, flags
