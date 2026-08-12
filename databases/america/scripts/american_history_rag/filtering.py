from __future__ import annotations

import re
from typing import Any


FICTION_TERMS = {
    "fiction", "novel", "novels", "short stories", "romance", "romances",
    "poetry", "poems", "drama", "plays", "juvenile fiction", "fairy tales",
}
NONFICTION_TERMS = {
    "history", "biography", "autobiography", "personal narratives", "diaries",
    "correspondence", "description and travel", "politics and government",
    "social life and customs", "social conditions", "labor", "working class",
    "slavery", "education", "handbooks, manuals, etc", "law", "trials",
    "public health", "medicine", "agriculture", "cookery", "domestic economy",
    "religion", "sermons", "statistics", "reports",
}
NONFICTION_TITLE_PATTERNS = (
    r"\breport of\b", r"\bannual report\b", r"\bproceedings of\b",
    r"\btestimony\b", r"\bhearing(?:s)?\b", r"\bhistory of\b",
    r"\bautobiograph", r"\bmemoir", r"\brecollections?\b", r"\bjournal of\b",
    r"\bletters? of\b", r"\bmanual of\b", r"\bhandbook\b", r"\bguide to\b",
    r"\bsermon", r"\baddress(?:es)?\b", r"\bconstitution\b", r"\bby-laws\b",
)
FICTION_TITLE_PATTERNS = (
    r"\ba novel\b", r"\ba romance\b", r"\bnovel\b", r"\bpoems?\b",
    r"\bshort stories\b", r"\bfairy tales\b", r"\ba play in\b",
)

# Conservative: these LC classes are predominantly documentary/nonfiction for this corpus.
NONFICTION_LCC_PREFIXES = {
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
    "N", "Q", "R", "S", "T", "U", "V", "Z",
}


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_strings(item))
        return output
    return [str(value)]


def loc_metadata(record: dict) -> dict[str, list[str]]:
    """Normalize fields that occur at either LOC's item or nested item level."""
    nested = record.get("item") if isinstance(record.get("item"), dict) else {}

    def both(*keys: str) -> list[str]:
        values: list[str] = []
        for key in keys:
            values.extend(_strings(record.get(key)))
            values.extend(_strings(nested.get(key)))
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    return {
        "subjects": both("subject", "subjects", "subject_headings"),
        "call_numbers": both("call_number"),
        "formats": both("format", "original_format", "type"),
        "descriptions": both("description", "notes"),
        "contributors": both("contributor", "contributors", "contributor_names"),
        "languages": both("language", "languages"),
        "locations": both("location", "locations", "place_of_publication"),
    }


def classify_loc_nonfiction(record: dict) -> dict:
    fields = loc_metadata(record)
    title = str(record.get("title") or (record.get("item") or {}).get("title") or "")
    subject_text = " | ".join(fields["subjects"]).lower()
    descriptive_text = " | ".join(fields["descriptions"]).lower()
    combined = f"{title.lower()} | {subject_text} | {descriptive_text}"
    reasons: list[str] = []

    fiction_hits = sorted(term for term in FICTION_TERMS if term in subject_text)
    if fiction_hits:
        reasons.append("fiction subject: " + ", ".join(fiction_hits))
    for pattern in FICTION_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            reasons.append(f"fiction title pattern: {pattern}")

    prefixes: list[str] = []
    for call_number in fields["call_numbers"]:
        match = re.match(r"\s*([A-Za-z]{1,3})", call_number)
        if match:
            prefixes.append(match.group(1).upper())
    if any(prefix.startswith("PZ") for prefix in prefixes):
        reasons.append("fiction-heavy LC class: PZ")

    if reasons:
        return {"status": "fiction", "confidence": "high", "reasons": reasons, **fields}

    nonfiction_hits = sorted(term for term in NONFICTION_TERMS if term in subject_text)
    if nonfiction_hits:
        reasons.append("nonfiction subject: " + ", ".join(nonfiction_hits))
    for pattern in NONFICTION_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            reasons.append(f"nonfiction title pattern: {pattern}")
    if any(prefix[:1] in NONFICTION_LCC_PREFIXES and not prefix.startswith("P") for prefix in prefixes):
        reasons.append("nonfiction LC class: " + ", ".join(prefixes))

    if reasons:
        return {"status": "nonfiction", "confidence": "high", "reasons": reasons, **fields}
    return {
        "status": "uncertain", "confidence": "low",
        "reasons": ["no decisive subject, title, or LC-class signal"], **fields,
    }


EVANS_REVIEW_TERMS = {
    "fiction", "novel", "romance", "poetry", "poems", "verse", "drama", "play",
    "song", "songs", "hymn", "hymns", "psalm", "psalms", "elegy", "ballad",
}


def evans_exclusion_reasons(title: str, subjects: list[str]) -> list[str]:
    title_lower = title.lower()
    subject_text = " | ".join(subjects).lower()
    reasons = []
    for term in sorted(EVANS_REVIEW_TERMS):
        if term in subject_text:
            reasons.append(f"subject contains '{term}'")
        elif re.search(rf"\b{re.escape(term)}\b", title_lower):
            reasons.append(f"title contains '{term}'")
    return reasons


CHRONICLING_LITERARY_TERMS = {
    "fiction", "novel", "novels", "short stories", "poetry", "poems", "drama",
    "plays", "juvenile fiction", "literary periodicals",
}
CHRONICLING_TITLE_PATTERNS = (
    r"\bliterary (?:magazine|journal|gazette|review)\b",
    r"\bfiction (?:magazine|weekly)\b",
    r"\bstory paper\b",
)
CHRONICLING_STRONG_CHUNK_PATTERNS = (
    r"(?im)^\s*chapter\s+(?:[ivxlcdm]+|\d+)\b",
    r"(?i)\bto be continued\b",
    r"(?i)\bcontinued in (?:our|the) next\b",
    r"(?i)\b(?:our|a|the) new serial (?:story|novel)\b",
    r"(?i)\bserial (?:story|romance|novel)\b",
    r"(?i)\ba thrilling romance\b",
)
CHRONICLING_WEAK_CHUNK_PATTERNS = (
    r"(?i)\bcontinued from (?:yesterday|last week|page \w+)\b",
    r"(?im)^\s*(?:poetry|selected verse|a poem)\s*$",
    r"(?i)\binstallment of (?:our|the|this) (?:story|novel|romance)\b",
    r"(?i)\bcopyright(?:ed)?[, ]+\d{4}.*?by\b",
)


def classify_chronicling_metadata(record: dict) -> dict:
    title = str(record.get("title", ""))
    subjects = _strings(record.get("subject")) + _strings(record.get("subjects"))
    genres = _strings(record.get("genre"))
    formats = _strings(record.get("format")) + _strings(record.get("original_format"))
    subject_text = " | ".join(subjects).lower()
    reasons = []
    hits = sorted(term for term in CHRONICLING_LITERARY_TERMS if term in subject_text)
    if hits:
        reasons.append("literary subject: " + ", ".join(hits))
    for pattern in CHRONICLING_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            reasons.append(f"literary publication-title pattern: {pattern}")
    if reasons:
        return {"status": "literary", "confidence": "high", "reasons": reasons,
                "subjects": subjects, "genres": genres, "formats": formats}
    newspaper_signal = any("newspaper" in value.lower() for value in genres + formats)
    if newspaper_signal:
        return {"status": "newspaper", "confidence": "high",
                "reasons": ["LOC genre/format identifies a newspaper"],
                "subjects": subjects, "genres": genres, "formats": formats}
    return {"status": "uncertain", "confidence": "low",
            "reasons": ["record lacks a decisive newspaper genre/format"],
            "subjects": subjects, "genres": genres, "formats": formats}


def classify_chronicling_chunk(text: str) -> dict:
    strong = [pattern for pattern in CHRONICLING_STRONG_CHUNK_PATTERNS if re.search(pattern, text)]
    weak = [pattern for pattern in CHRONICLING_WEAK_CHUNK_PATTERNS if re.search(pattern, text)]
    if strong or len(weak) >= 2:
        return {
            "status": "literary", "confidence": "high" if strong else "medium",
            "reasons": [f"OCR pattern: {pattern}" for pattern in strong + weak],
        }
    return {"status": "retain", "confidence": "medium",
            "reasons": ["no decisive serialized-fiction or poetry markers"]}
