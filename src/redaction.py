"""
redaction.py

Placeholder-safe text replacement and a regex backstop for PII the LLM schema does not cover.
Used only when the opt-in switches SAFE_TEXT_MATCHING / REGEX_BACKSTOP are on (see config/rambam.yaml);
the default config keeps the old plain str.replace behaviour, so outputs of past runs stay reproducible.
"""

import re

# Only the exact markers this pipeline writes, e.g. [REDACTED_DOCTOR_NAME], [DATE_DAY_-45],
# [ANONYMIZATION_ERROR]. A looser pattern would also protect bracketed text in the note itself, such as
# "seen by [SMITH]", which then could never be redacted.
PLACEHOLDER = re.compile(r"\[(?:REDACTED_[A-Z_]+|DATE_(?:DAY|MONTH|YEAR)_-?\d+|ANONYMIZATION_ERROR)\]")

HEBREW_PREFIXES = "ובכלמשה"
LETTER = r"[^\W\d_]"  # any Unicode letter (Hebrew or Latin)


def sub_outside_placeholders(pattern: re.Pattern, repl, text: str) -> tuple[str, int]:
    """
    re.subn, but never inside an existing placeholder.

    Why: placeholders are made of letters, so a short name matched case-insensitively ("Red", "Day")
    would otherwise break "[REDACTED_PHONE]" into "[[REDACTED_DOCTOR_NAME]ACTED_PHONE]", which the
    leak audit and downstream code can no longer recognise.
    """
    parts = re.split(f"({PLACEHOLDER.pattern})", text)
    total = 0
    for i in range(0, len(parts), 2):  # even indices are the text between placeholders
        parts[i], n = pattern.subn(repl, parts[i])
        total += n
    return "".join(parts), total


def _flexible(value: str) -> str:
    """Pattern for a value that tolerates any run of whitespace between its words."""
    return r"\s+".join(re.escape(word) for word in value.split())


def replace_text_value(text: str, value: str, tag: str) -> tuple[str, int]:
    """
    Replace a name or location, without cutting it out of longer words when it is short.

    Values of <= 3 letters (e.g. "Tal") hit inside other words ("Talmud"), so they must stand as a
    whole word; up to 3 Hebrew prefix letters (ו ב כ ל מ ש ה, as in "מחיפה") may stick to the left and
    are kept. If that finds nothing, fall back to a plain match: damaging a word is better than leaking
    a name that a lost line break glued to the next word. Longer values rarely collide, so a plain match
    (whitespace- and case-tolerant) is used directly.
    """
    if not value.split():
        return text, 0
    plain = re.compile(_flexible(value), re.IGNORECASE)
    if sum(ch.isalpha() for ch in value) <= 3:
        whole_word = re.compile(
            rf"(?<!{LETTER})([{HEBREW_PREFIXES}]{{0,3}}){_flexible(value)}(?!{LETTER})", re.IGNORECASE
        )
        text, n = sub_outside_placeholders(whole_word, lambda m: m.group(1) + tag, text)
        if n:
            return text, n
    return sub_outside_placeholders(plain, tag, text)


def age_pattern(age) -> re.Pattern:
    """
    The age as a whole number only.

    Plain str.replace also hit the same digits inside other numbers (age 71 -> "VA 0.71", "6/71",
    "IOP 171", "3.4.71"), which damaged ~7.5% of age tags in the HY clinic notes.
    """
    return re.compile(rf"(?<!\d)(?<!\d[./,]){re.escape(str(age))}(?!\d)(?![./,]\d)")


def _valid_israeli_id(digits: str) -> bool:
    """Israeli ID check digit (Luhn-like): weights 1,2,1,2,...; digits of each product are summed."""
    digits = digits.zfill(9)
    total = 0
    for i, ch in enumerate(digits):
        product = int(ch) * (1 if i % 2 == 0 else 2)
        total += product - 9 if product > 9 else product
    return total % 10 == 0


# The LLM schema has no field for phones, emails or URLs, and the model also misses some IDs and ages.
# These patterns run after the LLM values, outside placeholders, and only when REGEX_BACKSTOP is on.
PHONE = re.compile(r"(?<![\d./])(?:\+972[-\s]?|0)(?:5\d|7\d|[2-489])[-\s]?\d{3}[-\s]?\d{4}(?![\d./])")
# The domain must end in letters (.com, .co.il): notes use "@" in measurements such as "0.5@12.3".
EMAIL = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# Stops before spaces and brackets, and never ends on punctuation, so "see www.x.org, then" keeps the comma.
URL = re.compile(r"(?:https?://|www\.)[^\s()<>\[\]\"']*[^\s()<>\[\]\"'.,;:!?]", re.IGNORECASE)
ID_CANDIDATE = re.compile(r"(?<![\d./])\d{8,9}(?![\d./])")
AGE_PHRASE = re.compile(
    rf"(?<!{LETTER})(בן|בת|גיל|גילו|גילה|age|aged)(\s*)(\d{{1,3}})(?!\d)(?![./,]\d)", re.IGNORECASE
)


def apply_regex_backstop(text: str) -> tuple[str, list[str]]:
    """Redact PII found by fixed patterns; returns the text and the names of the patterns that hit."""
    hits = []
    # Emails and URLs first: they can contain digit runs that look like phones or IDs.
    steps = [
        ("email", EMAIL, "[REDACTED_EMAIL]"),
        ("url", URL, "[REDACTED_URL]"),
        ("phone", PHONE, "[REDACTED_PHONE]"),
        ("id", ID_CANDIDATE, lambda m: "[REDACTED_PATIENT_ID]" if _valid_israeli_id(m.group()) else m.group()),
        ("age", AGE_PHRASE, lambda m: f"{m.group(1)}{m.group(2)}[REDACTED_PATIENT_AGE]"),
    ]
    for name, pattern, repl in steps:
        new_text, _ = sub_outside_placeholders(pattern, repl, text)
        if new_text != text:
            hits.append(f"backstop:{name}")
        text = new_text
    return text, hits
