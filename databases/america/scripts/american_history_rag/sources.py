from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urljoin

import requests

from .filtering import (classify_chronicling_metadata, classify_loc_nonfiction,
                        evans_exclusion_reasons)
from .schema import Document
from .util import (RateLimitError, RateLimiter, cached_json, cached_text, clean_text,
                   get_with_retries, session, stable_id)


LOC_API = "https://www.loc.gov"
FOUNDERS_METADATA = "https://founders.archives.gov/Metadata/founders-online-metadata.json"
FOUNDERS_API = "https://founders.archives.gov/API/docdata/"
LOC_BOOK_MANIFEST = "https://data.labs.loc.gov/digitized-books/sample-data/manifest.json"
LOC_BOOK_METADATA = "https://data.labs.loc.gov/digitized-books/sample-data/metadata.json"
AMERICA_DATA = Path("databases/america/data")

# Both LOC collections use these same backend services. A single process-wide
# limiter therefore shares each published allowance across both collectors.
LOC_JSON_LIMITER = RateLimiter(20)
LOC_TEXT_LIMITER = RateLimiter(150)
LOC_STORAGE_LIMITER = RateLimiter(150)
CHRONICLING_BULK_LIMITER = RateLimiter(60)


def _limit_download(url: str) -> None:
    if "tile.loc.gov/text-services/" in url:
        LOC_TEXT_LIMITER.wait()
    elif "tile.loc.gov/storage-services/" in url:
        LOC_STORAGE_LIMITER.wait()


def _first(value, default: str = "") -> str:
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def _year(value) -> int | None:
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(value))
    return int(match.group(1)) if match else None


def _batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def collect_founders(limit: int, delay: float = 0.12,
                     metadata_url: str = FOUNDERS_METADATA,
                     cache_dir: Path | None = AMERICA_DATA / "cache/founders",
                     workers: int = 4) -> Iterator[Document]:
    """Collect dated primary documents, excluding undated editorial matter."""
    if Path(metadata_url).is_file():
        records = __import__("json").loads(Path(metadata_url).read_text(encoding="utf-8"))
    else:
        try:
            records = cached_json(metadata_url, cache_dir, timeout=180)
        except RuntimeError as exc:
            raise RuntimeError(
                "Founders Online returned an empty metadata response. Download "
                "founders-online-metadata.json later and pass --metadata-file PATH."
            ) from exc
    if isinstance(records, dict):
        records = records.get("documents", records.get("results", []))
    candidates = []
    for record in records:
        date = record.get("date-from", "")
        permalink = record.get("permalink", "")
        year = _year(date)
        if not date or not permalink or (year is not None and year > 1929):
            continue
        marker = "/documents/"
        if marker not in permalink:
            continue
        api_id = permalink.split(marker, 1)[1].strip("/")
        candidates.append((record, date, permalink, api_id))

    def fetch(candidate):
        record, date, permalink, api_id = candidate
        try:
            payload = cached_json(urljoin(FOUNDERS_API, api_id),
                                  cache_dir / "documents" if cache_dir else None)
        except Exception:
            return None
        text = clean_text(payload.get("content", ""))
        if len(text) < 80:
            return None
        return Document(
            id=f"founders:{api_id}", source="founders_online",
            title=payload.get("title") or record.get("title", "Untitled"),
            text=text, url=payload.get("permalink", permalink), date=date,
            authors=payload.get("authors", record.get("authors", [])),
            genre="correspondence", rights="Check Founders Online terms; historical text and modern editorial material differ",
            metadata={"recipients": payload.get("recipients", []), "project": record.get("project", "")},
        )

    emitted = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for batch in _batches(candidates, max(1, workers) * 4):
            for document in executor.map(fetch, batch):
                if document is None:
                    continue
                yield document
                emitted += 1
                if emitted >= limit:
                    return
            time.sleep(delay)


def _candidate_text_urls(item: dict) -> list[str]:
    urls: list[str] = []
    for resource in item.get("resources", []):
        for key in ("fulltext_file", "url", "download", "text_file"):
            value = resource.get(key)
            if isinstance(value, str) and (value.endswith(".txt") or key == "fulltext_file"):
                urls.append(value)
        files = resource.get("files", [])
        for file_info in files if isinstance(files, list) else []:
            if isinstance(file_info, dict):
                url = file_info.get("url", "")
                if url.endswith(".txt"):
                    urls.append(url)
    return list(dict.fromkeys(urls))


def _alto_text(xml: str) -> str:
    root = ET.fromstring(xml)
    lines = []
    for line in root.iter():
        if line.tag.rsplit("}", 1)[-1] != "TextLine":
            continue
        words = [node.attrib.get("CONTENT", "") for node in line.iter()
                 if node.tag.rsplit("}", 1)[-1] == "String"]
        if words:
            lines.append(" ".join(words))
    return clean_text("\n".join(lines))


def collect_chronicling_batch(limit: int, batch_data_url: str,
                              cache_dir: Path | None = AMERICA_DATA / "cache/chronicling_bulk",
                              workers: int = 4) -> Iterator[Document]:
    """Collect pages from an official NDNP bulk batch without using loc.gov JSON."""
    base = batch_data_url.rstrip("/") + "/"
    batch_xml = cached_text(urljoin(base, "BATCH.xml"), cache_dir,
                            limiter=CHRONICLING_BULK_LIMITER)
    batch_root = ET.fromstring(batch_xml)
    issues = []
    for node in batch_root.iter():
        if node.tag.rsplit("}", 1)[-1] != "issue" or not (node.text or "").strip():
            continue
        relative = (node.text or "").strip().replace("\\", "/")
        marker = "/data/"
        if marker in relative:
            relative = relative.split(marker, 1)[1]
        elif "/" in relative:
            # Batch manifests use ../batch_NAME/<lccn>/... paths.
            parts = relative.split("/")
            relative = "/".join(parts[2:]) if relative.startswith("../") else relative
        issues.append({
            "url": urljoin(base, relative), "lccn": node.attrib.get("lccn", ""),
            "date": node.attrib.get("issueDate", ""),
            "edition": node.attrib.get("editionOrder", "1"),
        })

    page_specs = []
    for issue in issues:
        try:
            issue_xml = cached_text(issue["url"], cache_dir / "issues" if cache_dir else None,
                                    limiter=CHRONICLING_BULK_LIMITER)
            issue_root = ET.fromstring(issue_xml)
        except Exception:
            continue
        label = issue_root.attrib.get("LABEL", f"Newspaper issue {issue['date']}")
        issue_dir = issue["url"].rsplit("/", 1)[0] + "/"
        page_number = 0
        for file_node in issue_root.iter():
            if file_node.tag.rsplit("}", 1)[-1] != "file" or file_node.attrib.get("USE") != "ocr":
                continue
            href = next((n.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                         for n in file_node.iter() if n.tag.rsplit("}", 1)[-1] == "FLocat"), "")
            if not href:
                continue
            page_number += 1
            page_specs.append({**issue, "label": label, "page": page_number,
                               "ocr_url": urljoin(issue_dir, href)})
    def fetch_page(spec):
        try:
            xml = cached_text(spec["ocr_url"], cache_dir / "ocr" if cache_dir else None,
                              timeout=120, limiter=CHRONICLING_BULK_LIMITER)
            text = _alto_text(xml)
        except Exception:
            return None
        if len(text) < 100:
            return None
        canonical = (f"https://www.loc.gov/resource/{spec['lccn']}/{spec['date']}/"
                     f"ed-{spec['edition']}/?sp={spec['page']}")
        return Document(
            id="chronicling:" + stable_id(canonical), source="loc_chronicling_america",
            title=f"{spec['label']}, page {spec['page']}", text=text,
            url=canonical, date=spec["date"], genre="newspaper_page",
            rights="Library of Congress Chronicling America rights statement applies",
            metadata={
                "ocr_url": spec["ocr_url"], "lccn": spec["lccn"],
                "edition_number": spec["edition"], "page_number": spec["page"],
                "batch_data_url": base, "nonfiction_status": "newspaper",
                "nonfiction_confidence": "high",
                "classification_reasons": ["official NDNP Chronicling America batch page"],
            },
        )
    emitted = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for document in executor.map(fetch_page, page_specs):
            if document is None:
                continue
            yield document
            emitted += 1
            if emitted >= limit:
                return


def collect_chronicling(limit: int, *, date_from: int = 1900, date_to: int = 1929,
                        query: str = "", max_bytes: int = 20_000_000,
                        delay: float = 1.0,
                        cache_dir: Path | None = AMERICA_DATA / "cache/chronicling",
                        workers: int = 4) -> Iterator[Document]:
    """Collect page-level newspaper OCR through the supported loc.gov API."""
    client = session()
    params = {"fo": "json", "c": min(100, max(limit * 3, 20)),
              "dates": f"{date_from}/{date_to}", "at": "results,pagination"}
    if query:
        params["q"] = query
    search_url = client.prepare_request(
        requests.Request("GET", f"{LOC_API}/newspapers/", params=params)
    ).url
    search_payload = cached_json(search_url, cache_dir / "searches" if cache_dir else None,
                                 timeout=90, limiter=LOC_JSON_LIMITER)
    summaries = search_payload.get("results", [])

    def fetch(summary):
        item_url = summary.get("id") or summary.get("url")
        if not item_url:
            return None
        separator = "&" if "?" in item_url else "?"
        try:
            payload = cached_json(item_url + separator + "fo=json", cache_dir,
                                  limiter=LOC_JSON_LIMITER)
        except Exception:
            # A single rate-limited or malformed page should not discard the
            # successfully harvested pages already streamed to disk.
            return None
        item = dict(payload.get("item", {}))
        item["resources"] = payload.get("resources", item.get("resources", []))
        classification = classify_chronicling_metadata(item)
        if classification["status"] != "newspaper":
            return None
        text = ""
        text_url = ""
        for candidate in _candidate_text_urls(item):
            try:
                _limit_download(candidate)
                text_response = get_with_retries(session(), candidate)
                if len(text_response.content) > max_bytes:
                    continue
                if "json" in text_response.headers.get("content-type", ""):
                    wrapped = text_response.json()
                    first = next(iter(wrapped.values()), {}) if isinstance(wrapped, dict) else {}
                    text = clean_text(first.get("full_text", "") if isinstance(first, dict) else "")
                else:
                    text = clean_text(text_response.text)
                text_url = candidate
                if len(text) >= 100:
                    break
            except Exception:
                continue
        if len(text) < 100:
            return None
        date = _first(item.get("date") or item.get("dates"))
        year = _year(date)
        if year is not None and not date_from <= year <= date_to:
            return None
        canonical = item.get("id", item_url)
        return Document(
            id="chronicling:" + stable_id(canonical), source="loc_chronicling_america",
            title=item.get("title", summary.get("title", "Newspaper page")), text=text,
            url=canonical, date=date, place=_first(item.get("location")), genre="newspaper_page",
            rights=_first(item.get("rights_advisory"), "Library of Congress rights statement applies"),
            metadata={
                "ocr_url": text_url,
                "partof": item.get("partof", []),
                "newspaper_title": item.get("newspaper_title", []),
                "date_issued": item.get("date_issued", ""),
                "dates_of_publication": item.get("dates_of_publication", ""),
                "place_of_publication": item.get("place_of_publication", ""),
                "location_city": item.get("location_city", []),
                "location_county": item.get("location_county", []),
                "location_state": item.get("location_state", []),
                "location_country": item.get("location_country", []),
                "subjects": item.get("subject", item.get("subjects", [])),
                "subject_headings": item.get("subject_headings", []),
                "genre": item.get("genre", []),
                "nonfiction_status": classification["status"],
                "nonfiction_confidence": classification["confidence"],
                "classification_reasons": classification["reasons"],
                "publication_frequency": item.get("publication_frequency", []),
                "edition_number": item.get("number_edition", []),
                "call_number": item.get("call_number", []),
                "medium": item.get("medium", ""),
                "language": item.get("language", []),
                "iiif_manifest_url": (item.get("resources") or [{}])[0].get("iiif_manifest_url", ""),
            },
        )

    emitted = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for batch in _batches(summaries, max(1, workers) * 4):
            for document in executor.map(fetch, batch):
                if document is None:
                    continue
                yield document
                emitted += 1
                if emitted >= limit:
                    return
            time.sleep(delay)


def _manifest_rows(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload.get("cols"), list) and isinstance(payload.get("rows"), list):
        return [dict(zip(payload["cols"], row)) for row in payload["rows"]]
    for key in ("files", "resources", "items", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def collect_loc_books(limit: int, manifest_url: str = LOC_BOOK_MANIFEST,
                      metadata_url: str = LOC_BOOK_METADATA,
                      max_bytes: int = 20_000_000,
                      include_uncertain: bool = False,
                      cache_dir: Path | None = AMERICA_DATA / "cache/loc_books",
                      workers: int = 4) -> Iterator[Document]:
    """Collect OCR from the LOC Selected Digitized Books data-package manifest."""
    rows = _manifest_rows(cached_json(manifest_url, cache_dir, timeout=180))
    bulk_records = cached_json(metadata_url, cache_dir / "catalog" if cache_dir else None,
                               timeout=180) if metadata_url else []
    metadata_by_id = {}
    for record in bulk_records if isinstance(bulk_records, list) else []:
        keys = [record.get("Object_id"), record.get("id"), record.get("url")]
        for key in keys:
            if key:
                metadata_by_id[str(key).rstrip("/").rsplit("/", 1)[-1]] = record

    def fetch(row):
        if int(row.get("size") or 0) > max_bytes:
            return None
        url = row.get("url") or row.get("download") or row.get("file_url") or row.get("object_key")
        if url and not str(url).startswith(("http://", "https://")):
            url = "https://" + str(url).lstrip("/")
        if not url or not str(url).endswith(".txt"):
            return None
        item_ref = str(row.get("item_id") or row.get("itemid") or "")
        if item_ref.startswith("http://www.loc.gov/"):
            item_ref = "https://www.loc.gov/" + item_ref.removeprefix("http://www.loc.gov/")
        item_id = item_ref.rstrip("/").rsplit("/", 1)[-1] if item_ref else str(row.get("id") or stable_id(url))
        item_metadata_url = item_ref.rstrip("/") + "/?fo=json" if item_ref else re.sub(r"\.txt$", ".json", url)
        metadata = metadata_by_id.get(item_id, {})
        if not metadata:
            try:
                metadata = cached_json(item_metadata_url, cache_dir / "items" if cache_dir else None,
                                       limiter=LOC_JSON_LIMITER)
            except RateLimitError:
                raise
            except Exception:
                pass
        item_metadata = metadata.get("item", metadata)
        classification = classify_loc_nonfiction(item_metadata)
        if classification["status"] == "fiction":
            return None
        if classification["status"] == "uncertain" and not include_uncertain:
            return None
        date = _first(item_metadata.get("date") or item_metadata.get("dates"))
        year = _year(date)
        if year is not None and year > 1929:
            return None
        # Download OCR only after the cheaper metadata filter accepts the record.
        _limit_download(url)
        ocr_response = get_with_retries(session(), url, timeout=180)
        text = clean_text(ocr_response.content.decode("utf-8", errors="replace"))
        if len(text) < 100:
            return None
        canonical = item_metadata.get("id") or item_ref or f"https://www.loc.gov/item/{item_id}/"
        return Document(
            id=f"locbook:{item_id}", source="loc_selected_digitized_books",
            title=item_metadata.get("title", item_id), text=text, url=canonical, date=date,
            authors=classification["contributors"],
            place=_first(classification["locations"]), genre="nonfiction_book",
            rights="Public domain; credit Library of Congress",
            metadata={
                "ocr_url": url,
                "nonfiction_status": classification["status"],
                "nonfiction_confidence": classification["confidence"],
                "classification_reasons": classification["reasons"],
                "subjects": classification["subjects"],
                "call_numbers": classification["call_numbers"],
                "formats": classification["formats"],
                "descriptions": classification["descriptions"],
                "languages": classification["languages"],
                "locations": classification["locations"],
            },
        )

    emitted = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for batch in _batches(rows, max(1, workers) * 4):
            for document in executor.map(fetch, batch):
                if document is None:
                    continue
                yield document
                emitted += 1
                if emitted >= limit:
                    return


def _xml_text(element: ET.Element) -> str:
    paragraphs: list[str] = []
    for node in element.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag in {"p", "head", "l", "item", "note"}:
            value = " ".join("".join(node.itertext()).split())
            if value:
                paragraphs.append(value)
    return "\n\n".join(paragraphs) or " ".join(element.itertext())


def collect_evans(limit: int, repo: Path, include_review_candidates: bool = False) -> Iterator[Document]:
    """Parse a local checkout of the public-domain Evans-TCP TEI/XML corpus."""
    paths = sorted(set(repo.rglob("*.xml")) | set(repo.rglob("*.XML")))
    emitted = 0
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        title_node = next((n for n in root.iter() if n.tag.rsplit("}", 1)[-1] == "title"), None)
        author_nodes = [n for n in root.iter() if n.tag.rsplit("}", 1)[-1] == "author"]
        publication_year = next(
            (clean_text("".join(n.itertext())) for n in root.iter()
             if n.tag.rsplit("}", 1)[-1] == "publicationYear"), ""
        )
        subjects = [clean_text("".join(n.itertext())) for n in root.iter()
                    if n.tag.rsplit("}", 1)[-1] == "term" and clean_text("".join(n.itertext()))]
        def first_value(name: str) -> str:
            return next((clean_text("".join(n.itertext())) for n in root.iter()
                         if n.tag.rsplit("}", 1)[-1] == name), "")
        review_reasons = evans_exclusion_reasons(
            clean_text("".join(title_node.itertext())) if title_node is not None else path.stem,
            subjects,
        )
        if review_reasons and not include_review_candidates:
            continue
        text = clean_text(_xml_text(root))
        if len(text) < 100:
            continue
        rel = path.relative_to(repo).as_posix()
        yield Document(
            id=f"evans:{path.stem}", source="evans_tcp",
            title=clean_text("".join(title_node.itertext())) if title_node is not None else path.stem,
            text=text, url=f"https://texts.earlyprint.org/works/{path.stem}",
            date=publication_year,
            authors=[clean_text("".join(n.itertext())) for n in author_nodes if clean_text("".join(n.itertext()))],
            genre="early_american_imprint", rights="Public-domain Evans-TCP transcription",
            metadata={
                "local_xml": rel,
                "subjects": list(dict.fromkeys(subjects)),
                "publisher": first_value("publisher"),
                "publication_place": first_value("pubPlace"),
                "language": first_value("language"),
                "word_count": first_value("wordCount"),
                "page_count": first_value("pageCount"),
                "defect_rate": first_value("defectRate"),
                "quality_grade": first_value("finalGrade"),
                "tcp_id": first_value("tcp") or path.stem,
                "stc_id": first_value("stc"),
                "nonfiction_status": "review_candidate" if review_reasons else "retained",
                "classification_reasons": review_reasons or ["no literary review terms in title or subjects"],
            },
        )
        emitted += 1
        if emitted >= limit:
            return


def collect(source: str, limit: int, **kwargs) -> Iterable[Document]:
    functions = {"founders-online": collect_founders, "founders": collect_founders,
                 "loc-chronicling-america": collect_chronicling, "chronicling": collect_chronicling,
                 "loc-selected-digitized-books": collect_loc_books, "loc-books": collect_loc_books,
                 "evans-tcp": collect_evans, "evans": collect_evans}
    if source not in functions:
        raise ValueError(f"Unknown source: {source}")
    return functions[source](limit=limit, **kwargs)
