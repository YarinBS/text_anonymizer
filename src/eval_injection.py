"""
eval_injection.py

Synthetic PII injection used by evaluate.py (metric 1, Synthetic PII Injection Recall).

The code below is copied verbatim from the February experiments
(src/legacy/metrics_vllm.py). It is copied, not rewritten, because the order of
`random` calls decides which PII values are injected: with the same seed, the
evaluation set is identical to the one behind the presentation results.
"""

import random
import string
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Hebrew PII pools for synthetic injection
# ---------------------------------------------------------------------------

HEBREW_FIRST_NAMES = [
    "אבי", "יוסי", "דוד", "משה", "שרה", "רחל", "מרים", "יעקב",
    "חיים", "יצחק", "רבקה", "לאה", "דניאל", "נועה", "תמר", "אורי",
    "גיל", "עמית", "שלמה", "אהרון", "אסתר", "אלון", "מיכל", "ענת",
]

HEBREW_LAST_NAMES = [
    "כהן", "לוי", "מזרחי", "פרץ", "ביטון", "דהן", "אברהם", "פרידמן",
    "גולדשטיין", "שפירא", "רוזנברג", "ברקוביץ", "אזולאי", "חדד", "גבאי",
    "מלכה", "אוחיון", "בנימין", "סויסה", "אלבז",
]

HEBREW_CITIES = [
    "תל אביב", "חיפה", "ירושלים", "באר שבע", "נתניה", "אשדוד",
    "ראשון לציון", "פתח תקווה", "הרצליה", "רמת גן", "חולון",
    "בת ים", "רחובות", "כפר סבא", "רעננה", "עכו", "נהריה",
]

# Medication names with real spelling variants found in the corpus
_MEDICATIONS: List[Tuple[str, List[str]]] = [
    # (canonical, [variants that appear in the real texts])
    ("avastin",  ["אווסטין", "AVASTIN", "אבסטין", "avastin"]),
    ("eylea",    ["איליה", "אייליה", "אילייה", "EYLEA", "איליא", "אליא", "אליה"]),
    ("lucentis", ["לוסנטיס", "Lucentis", "לוסנטס"]),
]

# Injection entry templates — {date}, {med}, {n} will be filled in.
# These mirror the concatenated, low-whitespace style of the real data.
_INJECTION_TEMPLATES = [
    "{date} זריקת {med} מספר {n}",
    "{date} זריקה מספר {n} {med}",
    "{date} זריקת {med} #{n}",
    "{date} זריקה #{n} {med}",
    "{date} זריקה מס {n}   סוג {med}",
    "{date} הזרקת {med} מספר {n}",
    "{date} זריקה {n} {med}",
]

_PII_PROFILES: List[Tuple[float, str, List[str]]] = [
    # weight, profile_name, categories_to_inject
    # "minimal" — only intervention dates (the most common PII in this corpus)
    (0.40, "minimal",       ["intervention_dates"]),
    # "dates_only" — intervention + other dates, no personal identifiers
    (0.15, "dates_only",    ["intervention_dates", "other_dates"]),
    # "age_and_dates" — adds patient age (בן 76)
    (0.15, "age_and_dates", ["patient_age", "intervention_dates", "other_dates"]),
    # "doctor_ref" — doctor name in indication phrase
    (0.12, "doctor_ref",    ["doctor_name", "intervention_dates", "other_dates"]),
    # "moderate" — age + doctor + dates
    (0.10, "moderate",      ["patient_age", "doctor_name", "intervention_dates", "other_dates"]),
    # "full" — all PII categories (rare in real data, but important to test)
    (0.08, "full",          ["patient_name", "patient_id", "patient_age",
                             "patient_date_of_birth", "doctor_name",
                             "intervention_dates", "other_dates", "locations"]),
]


def _select_pii_profile() -> Tuple[str, List[str]]:
    """
    Randomly select a PII profile based on the calibrated weights.

    Returns:
        (profile_name, list_of_pii_categories_to_inject)
    """
    weights = [p[0] for p in _PII_PROFILES]
    chosen = random.choices(_PII_PROFILES, weights=weights, k=1)[0]
    return chosen[1], list(chosen[2])


def _random_israeli_id() -> str:
    """Generate a random 9-digit Israeli-style ID number."""
    return "".join(random.choices(string.digits, k=9))


def _random_date() -> str:
    """Generate a random date string in a format common in Hebrew clinical texts."""
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2018, 2025)
    fmt = random.choice([
        f"{day:02d}/{month:02d}/{year}",
        f"{day:02d}.{month:02d}.{year}",
        f"{day}/{month:02d}/{str(year)[2:]}",
        f"{day}.{month}.{str(year)[2:]}",
    ])
    return fmt


def _random_age() -> int:
    return random.randint(30, 95)


def _random_doctor_name() -> str:
    prefix = random.choice(['ד"ר', "דר"])
    last = random.choice(HEBREW_LAST_NAMES)
    return f"{prefix} {last}"


def _random_patient_name() -> Tuple[str, str, str]:
    """Return (full_name, first, last)."""
    first = random.choice(HEBREW_FIRST_NAMES)
    last = random.choice(HEBREW_LAST_NAMES)
    return f"{first} {last}", first, last


def _pick_medication() -> Tuple[str, str]:
    """
    Pick a random medication and a random spelling variant.

    Returns:
        (canonical_name, display_variant) — e.g. ("avastin", "אבסטין")
    """
    canonical, variants = random.choice(_MEDICATIONS)
    return canonical, random.choice(variants)


def _random_injection_date() -> str:
    """
    Generate a date in the style dominant in the real corpus.

    Real distribution: ~60% slash dates, ~40% dot dates.
    Some dates use 2-digit years, some use 4-digit years.
    Leading zeros are sometimes omitted (e.g. 1/12/20 vs 01/12/2020).
    """
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2018, 2025)

    # Decide on separator: slash (60%) vs dot (40%)  — matches corpus ratio
    use_slash = random.random() < 0.6
    sep = "/" if use_slash else "."

    # Decide on year format: 4-digit (60%) vs 2-digit (40%)
    yr_str = str(year) if random.random() < 0.6 else str(year)[2:]

    # Decide on leading zeros: sometimes omitted in real texts
    if random.random() < 0.5:
        return f"{day}{sep}{month}{sep}{yr_str}"         # no leading zeros
    else:
        return f"{day:02d}{sep}{month:02d}{sep}{yr_str}" # with leading zeros


def _ends_with_digit(s: str) -> bool:
    """True if the string ends with a digit (or digit-like: slash, dot)."""
    return bool(s) and s.rstrip()[-1:] in set("0123456789/.")


def _starts_with_digit(s: str) -> bool:
    """True if the string starts with a digit."""
    return bool(s) and s.lstrip()[:1].isdigit()


def _safe_concat(left: str, right: str, allow_no_space: bool = True) -> str:
    """
    Concatenate two text fragments with a randomly chosen separator,
    but NEVER allow zero-width joining when both sides are numeric/date.

    In real clinical texts, concatenation without spaces happens between
    Hebrew words and dates (e.g. "תקין14/02/22"), but never between two
    dates or two numbers — that would create an unreadable merged token
    like "26.06.238.4.2018".

    Args:
        left:  The preceding text fragment.
        right: The following text fragment.
        allow_no_space: If True, may omit the space when safe to do so.
    """
    # If both ends are digit-adjacent, always insert at least a space
    if _ends_with_digit(left) and _starts_with_digit(right):
        return left + " " + right

    if allow_no_space and random.random() < 0.4:
        return left + right  # no separator — safe (Hebrew↔digit boundary)
    return left + " " + right


@dataclass
class InjectedPII:
    """Tracks a single injected PII element so we can check if it was detected."""
    category: str          # e.g. "patient_name", "patient_id", ...
    value: str             # the exact string injected into the text
    detected: bool = False # set after pipeline evaluation


def inject_pii_into_existing_text(text: str) -> Tuple[str, List[InjectedPII]]:
    """
    Inject PII into an existing clinical text at *natural* positions.

    Unlike the older `inject_pii_into_text` (which prepends a comma-separated
    header containing ALL PII categories), this function:

      1. Selects a random PII profile so only a realistic subset of categories
         is injected (most real notes have 0-2 PII fields).
      2. Inserts each PII element at the position where it naturally occurs
         in real ophthalmology notes — inline, not as a bolted-on header.

    Injection positions:
      - patient_age     → prepended as "בן {age}, " before the existing text
      - patient_name    → prepended as "מטופל {name}, " before the age/text
      - patient_id      → inserted after the name/age as 'ת"ז {id}, '
      - doctor_name     → appended inside an indication-style parenthetical:
                           "(אינדיקציה {date} ד"ר {name})" spliced into text
      - intervention_dates → inserted as additional injection entries styled
                           like "{date} זריקת {med} מספר {n}" in the body
      - other_dates     → inserted as indication/admission date phrase
      - locations       → inserted after name/age as "מ{city}, "
      - patient_date_of_birth → inserted as "נולד ב- {dob}, " near age

    Args:
        text: An existing clinical text (typically from the real corpus).

    Returns:
        (augmented_text, list_of_InjectedPII) with ground-truth PII tracking.
    """
    injections: List[InjectedPII] = []
    profile_name, categories = _select_pii_profile()

    # Pre-generate PII values
    full_name, first, last = _random_patient_name()
    patient_id = _random_israeli_id()
    age = _random_age()
    dob = _random_date()
    doctor = _random_doctor_name()
    location = random.choice(HEBREW_CITIES)
    _, med_display = _pick_medication()

    # --- Build prefix (demographic info prepended to the note) -------------
    # In real data, demographic info (when present) appears at the very start
    # of the note: "בן 76, עם AMD רטוב מימין."
    # We prepend only the categories selected by the current profile.
    prefix_parts: List[str] = []

    if "patient_name" in categories:
        prefix_parts.append(f"מטופל {full_name}")
        injections.append(InjectedPII(category="patient_name", value=full_name))

    if "patient_age" in categories:
        gender_word = random.choice(["בן", "בת"])
        prefix_parts.append(f"{gender_word} {age}")
        injections.append(InjectedPII(category="patient_age", value=str(age)))

    if "patient_id" in categories:
        id_choice = random.choice([f'ת"ז {patient_id}', f'תז {patient_id}', f'ת.ז. {patient_id}', f'תעודת זהות {patient_id}'])
        prefix_parts.append(id_choice)
        injections.append(InjectedPII(category="patient_id", value=patient_id))

    if "patient_date_of_birth" in categories:
        prefix_parts.append(f"נולד ב- {dob}")
        injections.append(InjectedPII(category="patient_date_of_birth", value=dob))

    if "locations" in categories:
        prefix_parts.append(f"מ{location}")
        injections.append(InjectedPII(category="locations", value=location))

    # Combine prefix with the original text using ", " — the standard
    # separator seen in real notes for demographic fragments
    if prefix_parts:
        prefix = ", ".join(prefix_parts) + ", "
        text = prefix + text

    # --- Insert doctor name via an indication-style phrase -----------------
    # In real notes, the doctor name appears inside an indication phrase
    # like "(אינדיקציה 2.11.21 ד"ר לוי)" — we append it after the text body.
    if "doctor_name" in categories:
        indication_date = _random_injection_date()
        dr_phrase = random.choice([
            f"(אינדיקציה {indication_date} {doctor})",
            f"בהחלטת {doctor}",
            f"הופנה ממרפאת {doctor} לזריקות",
        ])
        injections.append(InjectedPII(category="doctor_name", value=doctor))

        # Also track the indication date if we used one with the doctor
        if "other_dates" in categories and indication_date in dr_phrase:
            injections.append(InjectedPII(category="other_dates", value=indication_date))

        # Insert the doctor phrase — either before the last sentence or appended
        text = text.rstrip() + " " + dr_phrase

    # --- Insert injection-date entries into the body -----------------------
    # We add 1-2 synthetic injection entries that look like real entries.
    if "intervention_dates" in categories:
        n_dates = random.randint(1, 2)
        date_entries: List[str] = []
        for j in range(1, n_dates + 1):
            inj_date = _random_injection_date()
            tmpl = random.choice(_INJECTION_TEMPLATES)
            entry = tmpl.format(date=inj_date, med=med_display, n=j)
            date_entries.append(entry)
            injections.append(InjectedPII(category="intervention_dates", value=inj_date))

        # Append injection entries at the end of the text (before any plan section)
        # with the dense, partially-concatenated style of real notes.
        # Use _safe_concat to avoid merging two dates together.
        for entry in date_entries:
            text = _safe_concat(text.rstrip(), entry)

    # --- Insert an "other date" (admission/discharge) ----------------------
    if "other_dates" in categories and not any(
        inj.category == "other_dates" for inj in injections
    ):
        other_date = _random_injection_date()
        other_phrase = f"(אינדיקציה {other_date})"
        text = text.rstrip() + " " + other_phrase
        injections.append(InjectedPII(category="other_dates", value=other_date))

    return text, injections


def _check_detection(injections: List[InjectedPII], extracted_pii: dict) -> None:
    """Mark each injection as detected/not based on extracted PII dict."""
    for inj in injections:
        pii_value = extracted_pii.get(inj.category)
        if pii_value is None:
            continue
        if isinstance(pii_value, list):
            # Check if any extracted span contains / matches the injected value
            for span in pii_value:
                if span is not None and (inj.value in str(span) or str(span) in inj.value):
                    inj.detected = True
                    break
        else:
            if inj.value in str(pii_value) or str(pii_value) in inj.value:
                inj.detected = True
