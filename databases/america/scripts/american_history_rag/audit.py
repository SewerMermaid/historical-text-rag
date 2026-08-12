from __future__ import annotations

import csv
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

from .filtering import evans_exclusion_reasons
from .util import clean_text
from .util import get_with_retries, session


def _values(root: ET.Element, name: str) -> list[str]:
    return list(dict.fromkeys(
        clean_text("".join(node.itertext())) for node in root.iter()
        if node.tag.rsplit("}", 1)[-1] == name and clean_text("".join(node.itertext()))
    ))


def audit_evans(repo: Path, output: Path) -> tuple[int, int]:
    rows = []
    total = 0
    for path in sorted(set(repo.rglob("*.xml")) | set(repo.rglob("*.XML"))):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        total += 1
        titles = _values(root, "title")
        subjects = _values(root, "term")
        reasons = evans_exclusion_reasons(titles[0] if titles else path.stem, subjects)
        if not reasons:
            continue
        rows.append({
            "tcp_id": path.stem,
            "title": titles[0] if titles else path.stem,
            "publication_year": (_values(root, "publicationYear") or [""])[0],
            "authors": " | ".join(_values(root, "author")),
            "subjects": " | ".join(subjects),
            "candidate_reasons": " | ".join(reasons),
            "word_count": (_values(root, "wordCount") or [""])[0],
            "quality_grade": (_values(root, "finalGrade") or [""])[0],
            "url": f"https://texts.earlyprint.org/works/{path.stem}",
            "decision": "REVIEW—NOT EXCLUDED",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["tcp_id"])
        writer.writeheader()
        writer.writerows(rows)
    return total, len(rows)


def chronicling_field_report(records: list[dict]) -> dict:
    fields = Counter()
    values: dict[str, Counter] = {"genre": Counter(), "language": Counter(), "type": Counter()}
    for payload in records:
        item = payload.get("item", payload)
        for key, value in item.items():
            if value not in (None, "", []):
                fields[key] += 1
        for key in values:
            raw = item.get(key, [])
            raw = raw if isinstance(raw, list) else [raw]
            for value in raw:
                values[key][str(value)] += 1
    return {
        "records": len(records),
        "field_coverage": {key: {"count": count, "percent": round(100 * count / len(records), 1)}
                           for key, count in sorted(fields.items())} if records else {},
        "selected_values": {key: dict(counter.most_common(25)) for key, counter in values.items()},
    }


def audit_chronicling(sample_per_period: int, output: Path, delay: float = 1.0) -> dict:
    """Sample multiple periods because one issue cannot represent collection metadata."""
    client = session()
    periods = [(1800, 1849), (1850, 1879), (1880, 1909), (1910, 1929)]
    payloads: list[dict] = []
    seen: set[str] = set()
    errors: list[str] = []
    for start, end in periods:
        try:
            response = get_with_retries(
                client,
                "https://www.loc.gov/newspapers/?" + urlencode({
                    "fo": "json", "c": sample_per_period, "dates": f"{start}/{end}",
                    "at": "results",
                }),
                timeout=120,
            ).json()
        except Exception as exc:
            errors.append(f"{start}-{end}: {type(exc).__name__}: {exc}")
            continue
        for summary in response.get("results", []):
            item_url = summary.get("id") or summary.get("url")
            if not item_url or item_url in seen:
                continue
            item_url = item_url.replace("http://www.loc.gov/", "https://www.loc.gov/")
            seen.add(item_url)
            separator = "&" if "?" in item_url else "?"
            try:
                payload = get_with_retries(client, item_url + separator + "fo=json&at=item,resources",
                                           timeout=120, attempts=6).json()
                payloads.append(payload)
            except Exception:
                # A broad audit should retain its completed sample if one item is unavailable.
                pass
            time.sleep(delay)
    report = chronicling_field_report(payloads)
    report["periods"] = [f"{start}-{end}" for start, end in periods]
    report["errors"] = errors
    report["important_limitation"] = (
        "Metadata describes newspaper issues/pages, not article genre; serialized fiction cannot "
        "be filtered reliably without OCR/layout classification."
    )
    write_json(output, report)
    samples_path = output.with_name(output.stem + "_records.jsonl")
    with samples_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload.get("item", payload), ensure_ascii=False) + "\n")
    return report


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
