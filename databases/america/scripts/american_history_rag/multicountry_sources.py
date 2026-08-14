from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import urlencode

from .countries import CountryManifest, SourceManifest, country_manifest
from .schema import Document
from .util import cached_json, cached_text, clean_text, read_jsonl, stable_id


DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"
RIGHTS_PATTERNS = (
    r"\bpublic[ -]domain\b", r"\bdominio p[uú]blico\b",
    r"\bdom[ií]nio p[uú]blico\b", r"\bdomaine public\b", r"\bgemeinfrei\b",
    r"保護期間満了", r"저작권 만료", r"creativecommons\.org/publicdomain/(?:zero|mark)",
    r"\bcc0(?:[ -]?1\.0)?\b", r"\bpublic domain mark\b", r"\bpdm\b",
    r"creativecommons\.org/licenses/by(?:-sa)?/", r"\bcc[ -]by(?:[ -]sa)?\b",
    r"\bopen parliament licence\b", r"\bopendatacommons\.org/licenses/pddl\b",
    r"\bpddl\b", r"\bno known copyright restrictions?\b",
    r"\bno restrictions?\b",
)
RIGHTS_NEGATION_PATTERNS = (
    r"\bnot (?:in )?(?:the )?public[ -]domain\b",
    r"\bpublic[ -]domain (?:status )?(?:unknown|unavailable|not established)\b",
    r"\blicen[cs](?:e|ing) (?:unknown|unavailable|not available)\b",
    r"\bno derivatives?\b", r"\bno-derivatives?\b", r"\bby-nd\b", r"\bby-nc-nd\b",
    r"creativecommons\.org/licenses/by-(?:nc-)?nd/",
)


def _values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _publication_year(values: list[str]) -> int | None:
    years: list[int] = []
    for value in values:
        years.extend(int(match) for match in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", value))
    return min(years) if years else None


def rights_are_eligible(value: str) -> bool:
    """Require an affirmative reuse/derivative grant; cautions and negation fail closed."""
    normalized = re.sub(r"\s+", " ", value.casefold()).strip()
    if not normalized or any(re.search(pattern, normalized) for pattern in RIGHTS_NEGATION_PATTERNS):
        return False
    return any(re.search(pattern, normalized) for pattern in RIGHTS_PATTERNS)


def parse_oai_page(xml: str) -> tuple[list[dict], str]:
    """Return normalized Dublin Core records and the optional resumption token."""
    root = ET.fromstring(xml)
    records: list[dict] = []
    for record in root.findall(f".//{{{OAI_NAMESPACE}}}record"):
        header = record.find(f"{{{OAI_NAMESPACE}}}header")
        if header is None or header.attrib.get("status") == "deleted":
            continue
        fields: dict[str, list[str]] = {}
        metadata = record.find(f"{{{OAI_NAMESPACE}}}metadata")
        if metadata is not None:
            for node in metadata.iter():
                if node.tag.startswith("{" + DC_NAMESPACE + "}"):
                    value = clean_text("".join(node.itertext()))
                    if value:
                        fields.setdefault(node.tag.rsplit("}", 1)[-1], []).append(value)
        identifier = header.findtext(f"{{{OAI_NAMESPACE}}}identifier", default="")
        records.append({"oai_identifier": identifier, "fields": fields})
    token_node = root.find(f".//{{{OAI_NAMESPACE}}}resumptionToken")
    token = (token_node.text or "").strip() if token_node is not None else ""
    return records, token


def _text_url(fields: dict[str, list[str]]) -> str:
    candidates = fields.get("identifier", []) + fields.get("relation", [])
    return next((value for value in candidates
                 if value.lower().split("?", 1)[0].endswith((".txt", ".text"))), "")


def _oai_documents(
    country: CountryManifest,
    source: SourceManifest,
    limit: int,
    *,
    cache_dir: Path | None,
    max_bytes: int,
    fetch_metadata: Callable[[str, Path | None], str] = cached_text,
    fetch_text: Callable[[str, Path | None], str] = cached_text,
) -> Iterator[Document]:
    options = source.options
    params = {"verb": "ListRecords", "metadataPrefix": options.get("metadata_prefix", "oai_dc")}
    if options.get("set"):
        params["set"] = options["set"]
    next_url = source.endpoint + ("&" if "?" in source.endpoint else "?") + urlencode(params)
    emitted = 0
    pages = 0
    while next_url and emitted < limit and pages < 100:
        payload = fetch_metadata(next_url, cache_dir / "oai" if cache_dir else None)
        records, token = parse_oai_page(payload)
        pages += 1
        for record in records:
            fields = record["fields"]
            year = _publication_year(fields.get("date", []))
            # Unknown dates are quarantined: the cutoff is a filter, not a legal opinion.
            if year is None or year > country.publication_cutoff:
                continue
            text_url = _text_url(fields)
            if not text_url:
                continue
            rights = " | ".join(fields.get("rights", []))
            # A source caution is not a rights grant. Unknown records stay quarantined.
            if not rights_are_eligible(rights):
                continue
            try:
                text = clean_text(fetch_text(text_url, cache_dir / "text" if cache_dir else None))
            except Exception:
                continue
            if len(text) < 100 or len(text.encode("utf-8")) > max_bytes:
                continue
            identifiers = fields.get("identifier", [])
            canonical = next((value for value in identifiers if value.startswith(("http://", "https://"))), text_url)
            oai_id = record["oai_identifier"] or canonical
            yield Document(
                id=f"{source.id}:{stable_id(oai_id)}", source=source.id,
                title=(fields.get("title") or [oai_id])[0], text=text, url=canonical,
                date=" | ".join(fields.get("date", [])),
                authors=fields.get("creator", []) + fields.get("contributor", []),
                place=" | ".join(fields.get("coverage", [])),
                genre=" | ".join(fields.get("type", [])), rights=rights,
                country=country.id, language=country.language, content_mode="native",
                metadata={
                    "protocol": "oai-pmh", "oai_identifier": record["oai_identifier"],
                    "oai_endpoint": source.endpoint, "text_url": text_url,
                    "publication_year_filter": country.publication_cutoff,
                    "rights_caution": country.cutoff_note, "dc": fields,
                },
            )
            emitted += 1
            if emitted >= limit:
                return
        if not token:
            return
        next_url = source.endpoint + ("&" if "?" in source.endpoint else "?") + urlencode({
            "verb": "ListRecords", "resumptionToken": token,
        })


def _ia_documents(
    country: CountryManifest,
    source: SourceManifest,
    limit: int,
    *,
    cache_dir: Path | None,
    max_bytes: int,
    fetch_json: Callable[..., object] = cached_json,
    fetch_text: Callable[[str, Path | None], str] = cached_text,
) -> Iterator[Document]:
    query = source.options["query"]
    url = source.endpoint + "?" + urlencode({
        "q": query, "fl[]": ["identifier", "title", "date", "year", "creator",
                                "description", "subject", "licenseurl"],
        "rows": max(limit * 4, 20), "page": 1, "output": "json",
    }, doseq=True)
    payload = fetch_json(url, cache_dir / "search" if cache_dir else None)
    docs = payload.get("response", {}).get("docs", []) if isinstance(payload, dict) else []
    emitted = 0
    for metadata in docs:
        identifier = str(metadata.get("identifier", ""))
        if not identifier:
            continue
        year = _publication_year(_values(metadata.get("year")) + _values(metadata.get("date")))
        if year is None or year > country.publication_cutoff:
            continue
        item = fetch_json(f"https://archive.org/metadata/{identifier}",
                          cache_dir / "items" if cache_dir else None)
        item_metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        rights = str(item_metadata.get("licenseurl") or metadata.get("licenseurl") or "")
        if not rights_are_eligible(rights):
            continue
        files = item.get("files", []) if isinstance(item, dict) else []
        candidates = [file for file in files if (
            str(file.get("format", "")).lower() in {"djvutxt", "full text"}
            or str(file.get("name", "")).lower().endswith(("_djvu.txt", ".txt"))
        )]
        candidates.sort(key=lambda row: 0 if str(row.get("format", "")).lower() == "djvutxt" else 1)
        file = next((row for row in candidates
                     if int(row.get("size") or 0) <= max_bytes), None)
        if file is None:
            continue
        text_url = f"https://archive.org/download/{identifier}/{file['name']}"
        try:
            text = clean_text(fetch_text(text_url, cache_dir / "text" if cache_dir else None))
        except Exception:
            continue
        if len(text) < 100 or len(text.encode("utf-8")) > max_bytes:
            continue
        yield Document(
            id=f"{source.id}:{identifier}", source=source.id,
            title=str(item_metadata.get("title") or metadata.get("title") or identifier),
            text=text, url=f"https://archive.org/details/{identifier}",
            date=str(item_metadata.get("date") or metadata.get("date") or year),
            authors=_values(item_metadata.get("creator") or metadata.get("creator")),
            genre="historical_text_ocr", rights=rights, country=country.id,
            language=country.language, content_mode="native",
            metadata={
                "protocol": "internet-archive", "archive_identifier": identifier,
                "text_url": text_url, "publication_year_filter": country.publication_cutoff,
                "rights_caution": country.cutoff_note,
                "subjects": _values(item_metadata.get("subject") or metadata.get("subject")),
            },
        )
        emitted += 1
        if emitted >= limit:
            return


def _local_manifest_documents(
    country: CountryManifest,
    source: SourceManifest,
    limit: int,
    *,
    input_root: Path,
    max_bytes: int,
    rights_override: str = "",
) -> Iterator[Document]:
    """Ingest an audited local export described by ``records.jsonl``.

    This is the standards-neutral path for sources that publish bulk TEI/XML/text
    but do not expose a stable remote discovery API. Files never escape the chosen
    root, and every row needs an explicit eligible rights statement or recorded
    operator override.
    """
    root = input_root.resolve()
    records_path = root / "records.jsonl"
    if not records_path.is_file():
        raise ValueError(f"Local bulk import requires {records_path}")
    if rights_override and not rights_override.strip():
        raise ValueError("rights_override cannot be blank")
    emitted = 0
    for row in read_jsonl(records_path):
        if row.get("source") not in (None, "", source.id):
            continue
        year = _publication_year(_values(row.get("date")))
        if year is None or year > country.publication_cutoff:
            continue
        rights = str(row.get("rights") or rights_override)
        if not rights_are_eligible(rights) and not rights_override:
            continue
        relative_path = Path(str(row.get("path", "")))
        if not relative_path or relative_path.is_absolute():
            raise ValueError("Each local record needs a relative path")
        content_path = (root / relative_path).resolve()
        try:
            content_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Local record escapes input root: {relative_path}") from exc
        size = content_path.stat().st_size
        if size > max_bytes:
            continue
        if content_path.suffix.lower() in {".xml", ".tei"}:
            xml_root = ET.parse(content_path).getroot()
            text = clean_text(" ".join(value.strip() for value in xml_root.itertext() if value.strip()))
        else:
            text = clean_text(content_path.read_text(encoding="utf-8-sig"))
        if len(text) < 100:
            continue
        record_id = str(row.get("id") or relative_path.as_posix())
        url = str(row.get("url") or "")
        if not url:
            raise ValueError(f"Local record {record_id!r} needs a canonical url")
        yield Document(
            id=f"{country.id}:{source.id}:{stable_id(record_id)}", source=source.id,
            title=str(row.get("title") or content_path.stem), text=text, url=url,
            date=str(row.get("date") or year), authors=_values(row.get("authors")),
            place=str(row.get("place") or ""), genre=str(row.get("genre") or ""),
            rights=rights, country=country.id, language=country.language,
            content_mode="native", metadata={
                "protocol": source.protocol, "local_export": True,
                "local_record_id": record_id, "source_file": relative_path.as_posix(),
                "publication_year_filter": country.publication_cutoff,
                "rights_status": "operator_override" if rights_override else "source_record",
                "rights_caution": country.cutoff_note,
            },
        )
        emitted += 1
        if emitted >= limit:
            return


def collect_country_source(country_id: str, source_id: str, limit: int, *,
                           cache_dir: Path | None = None,
                           max_bytes: int = 20_000_000,
                           input_root: Path | None = None,
                           rights_override: str = "") -> Iterator[Document]:
    if limit < 1:
        raise ValueError("limit must be positive")
    country = country_manifest(country_id)
    source = country.source(source_id)
    if source.status != "enabled":
        reason = source.options.get("reason", "activation conditions have not been met")
        raise ValueError(f"Source {source.id!r} is {source.status}: {reason}")
    if input_root is not None:
        return _local_manifest_documents(
            country, source, limit, input_root=input_root, max_bytes=max_bytes,
            rights_override=rights_override,
        )
    if source.protocol == "oai-pmh":
        return _oai_documents(country, source, limit, cache_dir=cache_dir, max_bytes=max_bytes)
    if source.protocol == "internet-archive":
        return _ia_documents(country, source, limit, cache_dir=cache_dir, max_bytes=max_bytes)
    raise ValueError(
        f"Remote collector for protocol {source.protocol!r} is not implemented; "
        "use --input-root with an audited records.jsonl bulk export"
    )
