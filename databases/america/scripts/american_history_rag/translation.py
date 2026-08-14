from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import requests

from .countries import country_manifest
from .multicountry_sources import rights_are_eligible
from .schema import Document
from .util import read_jsonl, stable_id


TRANSLATION_MODEL = "gpt-5.6-luna"
MAX_BATCH_SIZE = 50
MAX_OUTPUT_TOKENS = 32_000
DEFAULT_MAX_DOCUMENT_CHARACTERS = 100_000
DEFAULT_MAX_BATCH_CHARACTERS = 200_000
DEFAULT_MAX_INPUT_CHARACTERS = 10_000_000
PROMPT_VERSION = "historical-translation-v1"
SCHEMA_VERSION = "historical-translations-v1"

TRANSLATION_INSTRUCTIONS = (
    "Translate every supplied historical-document field into faithful English. "
    "Preserve paragraph breaks, names, dates, quotations, uncertainty, and OCR artifacts; "
    "do not summarize, modernize, annotate, or omit text. Return JSON only as an object "
    "with a translations array. Each element must contain exactly id, title, text, place, genre."
)
TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {"translations": {
        "type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {key: {"type": "string"} for key in
                           ("id", "title", "text", "place", "genre")},
            "required": ["id", "title", "text", "place", "genre"],
        },
    }},
    "required": ["translations"],
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


PROMPT_SHA256 = _sha256(TRANSLATION_INSTRUCTIONS)
SCHEMA_SHA256 = _sha256(_canonical_json(TRANSLATION_SCHEMA))


def _requested_settings() -> dict[str, Any]:
    return {
        "model": TRANSLATION_MODEL,
        "store": False,
        "tools": [],
        "reasoning": {"effort": "low"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


SETTINGS_SHA256 = _sha256(_canonical_json(_requested_settings()))


@dataclass(frozen=True)
class TranslationBatchResult:
    translations: list[dict[str, str]]
    response_id: str | None
    returned_model: str | None
    usage: dict[str, Any]
    source_input_sha256: str
    output_sha256: str
    request_started_at: str
    completed_at: str


class TranslationClient(Protocol):
    def translate(self, documents: Sequence[Document]) -> TranslationBatchResult | list[dict[str, str]]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_inputs(documents: Sequence[Document]) -> list[dict[str, str]]:
    return [{
        "id": document.id, "title": document.title, "text": document.text,
        "place": document.place, "genre": document.genre,
    } for document in documents]


def _source_input_hash(documents: Sequence[Document]) -> str:
    return _sha256(_canonical_json(_request_inputs(documents)))


class OpenAIResponsesTranslationClient:
    """Strict Responses client. Network use requires an explicit execute opt-in."""

    def __init__(self, api_key: str | None = None, *, timeout: int = 300,
                 execute: bool = False):
        self.api_key = api_key
        self.timeout = timeout
        self.execute = execute

    def translate(self, documents: Sequence[Document], *,
                  execute: bool | None = None) -> TranslationBatchResult:
        if not (self.execute if execute is None else execute):
            raise RuntimeError("A real translation request requires execute=True")
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for an explicitly authorized translation run")

        inputs = _request_inputs(documents)
        source_input_sha256 = _sha256(_canonical_json(inputs))
        request_started_at = _utc_now()
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                **_requested_settings(),
                "instructions": TRANSLATION_INSTRUCTIONS,
                # Canonical compact JSON makes the planned character bound exact.
                "input": _canonical_json(inputs),
                "text": {"format": {
                    "type": "json_schema", "name": "historical_translations", "strict": True,
                    "schema": TRANSLATION_SCHEMA,
                }},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except (ValueError, TypeError) as error:
            raise RuntimeError("Responses API returned malformed JSON") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Responses API returned a malformed response object")
        if payload.get("error"):
            raise RuntimeError("Responses API returned an error response")
        if payload.get("status") != "completed":
            details = payload.get("incomplete_details")
            suffix = f": {details!r}" if details else ""
            raise RuntimeError(f"Responses API did not complete successfully{suffix}")
        if payload.get("model") != TRANSLATION_MODEL:
            raise RuntimeError(f"Responses API returned unexpected model {payload.get('model')!r}")
        response_id = payload.get("id")
        usage = payload.get("usage")
        if not isinstance(response_id, str) or not response_id:
            raise RuntimeError("Responses API returned no response ID")
        if not isinstance(usage, dict):
            raise RuntimeError("Responses API returned malformed usage metadata")

        output_text_parts: list[str] = []
        top_level_text = payload.get("output_text")
        if top_level_text is not None:
            if not isinstance(top_level_text, str):
                raise RuntimeError("Responses API returned malformed output text")
            if top_level_text.strip():
                output_text_parts.append(top_level_text)
        use_top_level_text = bool(output_text_parts)
        output = payload.get("output", [])
        if not isinstance(output, list):
            raise RuntimeError("Responses API returned malformed output items")
        for item in output:
            if not isinstance(item, dict) or not isinstance(item.get("content", []), list):
                raise RuntimeError("Responses API returned malformed output items")
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    raise RuntimeError("Responses API returned malformed content items")
                if content.get("type") == "refusal" or content.get("refusal"):
                    raise RuntimeError("Responses API refused the translation request")
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if not isinstance(text, str):
                        raise RuntimeError("Responses API returned malformed output text")
                    if not use_top_level_text:
                        output_text_parts.append(text)
        output_text = "".join(output_text_parts)
        if not output_text.strip():
            raise RuntimeError("Responses API returned no translation text")
        try:
            decoded = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("Responses API returned malformed translation JSON") from error
        if not isinstance(decoded, dict) or set(decoded) != {"translations"}:
            raise RuntimeError("Responses API translation JSON does not match the required schema")
        translations = _validate_translations(decoded["translations"], documents)
        return TranslationBatchResult(
            translations=translations,
            response_id=response_id,
            returned_model=payload["model"],
            usage=dict(usage),
            source_input_sha256=source_input_sha256,
            output_sha256=_sha256(_canonical_json(translations)),
            request_started_at=request_started_at,
            completed_at=_utc_now(),
        )


def _input_hash(document: Document) -> str:
    return _sha256(_canonical_json({
        "id": document.id, "title": document.title, "text": document.text,
        "place": document.place, "genre": document.genre, "language": document.language,
    }))


def _translation_identity(document: Document) -> dict[str, str]:
    return {
        "source_input_sha256": _input_hash(document),
        "model": TRANSLATION_MODEL,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
        "schema_version": SCHEMA_VERSION, "schema_sha256": SCHEMA_SHA256,
        "settings_sha256": SETTINGS_SHA256,
    }


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(dict(identity)))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    row_ids: set[str] = set()
    source_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise RuntimeError("Refusing to write translation output with duplicate row IDs")
        row_ids.add(row_id)
        source_id = (row.get("metadata", {}).get("translation_provenance", {})
                     .get("source_document_id"))
        if isinstance(source_id, str):
            if source_id in source_ids:
                raise RuntimeError("Refusing to write duplicate derived translation rows")
            source_ids.add(source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_state(state_path: Path, country: str) -> dict[str, Any]:
    if not state_path.exists():
        return {"version": 2, "country": country, "completed": {}, "batches": [],
                "in_flight": None}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Translation state is unreadable or malformed") from error
    if not isinstance(state, dict) or state.get("country") != country:
        raise ValueError("Translation state belongs to a different country")
    if not isinstance(state.get("completed", {}), dict):
        raise RuntimeError("Translation state has malformed completion records")
    state.setdefault("version", 2)
    state.setdefault("completed", {})
    state.setdefault("batches", [])
    state.setdefault("in_flight", None)
    return state


def _load_output_rows(output_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not output_path.exists():
        return [], {}
    rows = list(read_jsonl(output_path))
    by_source: dict[str, dict[str, Any]] = {}
    row_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise RuntimeError("Translation output contains a missing or duplicate row ID")
        row_ids.add(row_id)
        provenance = row.get("metadata", {}).get("translation_provenance", {})
        source_id = provenance.get("source_document_id")
        if isinstance(source_id, str):
            if source_id in by_source:
                raise RuntimeError("Translation output contains duplicate derived rows")
            by_source[source_id] = row
    return rows, by_source


def _validated_documents(input_path: Path, country: str) -> list[Document]:
    manifest = country_manifest(country)
    documents: list[Document] = []
    ids: set[str] = set()
    for raw in read_jsonl(input_path):
        document = Document(**raw)
        if document.id in ids:
            raise ValueError(f"Duplicate source document ID {document.id!r}")
        ids.add(document.id)
        if document.country != country:
            raise ValueError(f"Document {document.id!r} belongs to {document.country}, not {country}")
        if document.language != manifest.language or document.content_mode != "native":
            raise ValueError(f"Document {document.id!r} is not native {manifest.language} content")
        rights_status = document.metadata.get("rights_status")
        if not rights_are_eligible(document.rights) and rights_status != "operator_override":
            raise ValueError(
                f"Document {document.id!r} lacks an affirmative rights basis that permits derivatives"
            )
        documents.append(document)
    return documents


def _is_completed(document: Document, state: Mapping[str, Any],
                  output_by_source: Mapping[str, dict[str, Any]]) -> bool:
    identity = _translation_identity(document)
    completed = state.get("completed", {}).get(document.id, {})
    row = output_by_source.get(document.id)
    if not isinstance(completed, dict) or row is None:
        return False
    provenance = row.get("metadata", {}).get("translation_provenance", {})
    return (completed.get("identity") == identity and provenance.get("identity") == identity
            and completed.get("translated_document_id") == row.get("id"))


def _document_characters(document: Document) -> int:
    return len(_canonical_json(_request_inputs([document])[0]))


def _make_batches(documents: Sequence[Document], *, batch_size: int,
                  max_document_characters: int,
                  max_batch_characters: int) -> tuple[list[list[Document]], list[int]]:
    batches: list[list[Document]] = []
    batch_characters: list[int] = []
    current: list[Document] = []
    # Include the JSON array delimiters and separators in the actual request-input bound.
    current_characters = 2
    for document in documents:
        characters = _document_characters(document)
        if characters > max_document_characters:
            raise ValueError(f"Document {document.id!r} has {characters} translation input characters; "
                             f"the maximum is {max_document_characters}")
        if characters + 2 > max_batch_characters:
            raise ValueError(f"Document {document.id!r} cannot fit within the "
                             f"{max_batch_characters}-character batch ceiling")
        added_characters = characters + (1 if current else 0)
        if current and (len(current) >= batch_size or
                        current_characters + added_characters > max_batch_characters):
            batches.append(current)
            batch_characters.append(current_characters)
            current, current_characters = [], 2
            added_characters = characters
        current.append(document)
        current_characters += added_characters
    if current:
        batches.append(current)
        batch_characters.append(current_characters)
    return batches, batch_characters


def _planning_context(input_path: Path, output_path: Path, state_path: Path, *,
                      country: str, batch_size: int, limit: int | None,
                      max_input_characters: int, max_document_characters: int,
                      max_batch_characters: int
                      ) -> tuple[dict[str, Any], list[list[Document]], dict[str, Any],
                                 list[dict[str, Any]]]:
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
    for name, value in (("max_input_characters", max_input_characters),
                        ("max_document_characters", max_document_characters),
                        ("max_batch_characters", max_batch_characters)):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    manifest = country_manifest(country)
    state = _load_state(state_path, country)
    output_rows, output_by_source = _load_output_rows(output_path)
    pending = [document for document in _validated_documents(input_path, country)
               if not _is_completed(document, state, output_by_source)]
    if limit is not None:
        pending = pending[:max(0, limit)]
    batches, batch_character_counts = _make_batches(
        pending, batch_size=batch_size,
        max_document_characters=max_document_characters,
        max_batch_characters=max_batch_characters,
    )
    estimated_characters = sum(batch_character_counts)
    plan = {
        "model": TRANSLATION_MODEL, "country": country,
        "source_language": manifest.language,
        "input": str(input_path), "output": str(output_path), "state": str(state_path),
        "batch_size": batch_size, "pending_documents": len(pending),
        "planned_batches": len(batches),
        "estimated_input_characters": estimated_characters,
        "max_input_characters": max_input_characters,
        "within_input_character_ceiling": estimated_characters <= max_input_characters,
        "max_document_characters": max_document_characters,
        "max_batch_characters": max_batch_characters,
        "planned_batch_characters": batch_character_counts,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
        "schema_version": SCHEMA_VERSION, "schema_sha256": SCHEMA_SHA256,
        "requested_settings": _requested_settings(), "settings_sha256": SETTINGS_SHA256,
    }
    return plan, batches, state, output_rows


def translation_plan(input_path: Path, output_path: Path, state_path: Path, *,
                     country: str, batch_size: int = 10, limit: int | None = None,
                     max_input_characters: int = DEFAULT_MAX_INPUT_CHARACTERS,
                     max_document_characters: int = DEFAULT_MAX_DOCUMENT_CHARACTERS,
                     max_batch_characters: int = DEFAULT_MAX_BATCH_CHARACTERS) -> dict[str, Any]:
    plan, _, _, _ = _planning_context(
        input_path, output_path, state_path, country=country, batch_size=batch_size,
        limit=limit, max_input_characters=max_input_characters,
        max_document_characters=max_document_characters,
        max_batch_characters=max_batch_characters,
    )
    return plan


def _validate_translations(raw_translations: Any,
                           documents: Sequence[Document]) -> list[dict[str, str]]:
    if not isinstance(raw_translations, list):
        raise RuntimeError("Translation response does not contain a translations array")
    fields = {"id", "title", "text", "place", "genre"}
    translations: list[dict[str, str]] = []
    ids: list[str] = []
    for item in raw_translations:
        if not isinstance(item, dict) or set(item) != fields:
            raise RuntimeError("Translation response item does not match the required schema")
        if any(not isinstance(item[field], str) for field in fields):
            raise RuntimeError("Translation response fields must all be strings")
        translations.append(dict(item))
        ids.append(item["id"])
    expected_ids = [document.id for document in documents]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise RuntimeError("Translation response IDs do not exactly match the requested batch")
    by_id = {item["id"]: item for item in translations}
    return [by_id[document_id] for document_id in expected_ids]


def _normalize_result(result: TranslationBatchResult | list[dict[str, str]],
                      documents: Sequence[Document], *,
                      request_started_at: str) -> TranslationBatchResult:
    if isinstance(result, TranslationBatchResult):
        translations = _validate_translations(result.translations, documents)
        if result.source_input_sha256 != _source_input_hash(documents):
            raise RuntimeError("Translation client returned a mismatched source input hash")
        if result.output_sha256 != _sha256(_canonical_json(translations)):
            raise RuntimeError("Translation client returned a mismatched output hash")
        return result
    translations = _validate_translations(result, documents)
    return TranslationBatchResult(
        translations=translations, response_id=None, returned_model=None, usage={},
        source_input_sha256=_source_input_hash(documents),
        output_sha256=_sha256(_canonical_json(translations)),
        request_started_at=request_started_at, completed_at=_utc_now(),
    )


def translate_corpus(input_path: Path, output_path: Path, state_path: Path, *,
                     country: str, batch_size: int = 10, limit: int | None = None,
                     client: TranslationClient | None = None, dry_run: bool = False,
                     execute: bool = False, retry_ambiguous: bool = False,
                     max_input_characters: int = DEFAULT_MAX_INPUT_CHARACTERS,
                     max_document_characters: int = DEFAULT_MAX_DOCUMENT_CHARACTERS,
                     max_batch_characters: int = DEFAULT_MAX_BATCH_CHARACTERS) -> dict[str, Any]:
    plan, batches, state, output_rows = _planning_context(
        input_path, output_path, state_path, country=country, batch_size=batch_size,
        limit=limit, max_input_characters=max_input_characters,
        max_document_characters=max_document_characters,
        max_batch_characters=max_batch_characters,
    )
    if dry_run:
        return {**plan, "translated_documents": 0, "dry_run": dry_run, "responses": []}
    if state.get("in_flight") and not retry_ambiguous:
        marker = state["in_flight"]
        raise RuntimeError("Translation state contains an ambiguous in-flight batch "
                           f"{marker.get('batch_id')!r}; inspect it or explicitly set "
                           "retry_ambiguous=True")
    if plan["pending_documents"] == 0:
        return {**plan, "translated_documents": 0, "dry_run": False, "responses": []}
    if not plan["within_input_character_ceiling"]:
        raise ValueError(f"Planned translation input has {plan['estimated_input_characters']} characters; "
                         f"the caller-supplied ceiling is {max_input_characters}")
    if client is None:
        if not execute:
            raise RuntimeError("A real translation run requires execute=True")
        translation_client: TranslationClient = OpenAIResponsesTranslationClient(execute=True)
    else:
        translation_client = client
        if isinstance(client, OpenAIResponsesTranslationClient) and not execute:
            raise RuntimeError("A real translation run requires execute=True")

    manifest = country_manifest(country)
    translated_count = 0
    response_records: list[dict[str, Any]] = []
    for batch in batches:
        identities = {document.id: _translation_identity(document) for document in batch}
        batch_source_hash = _source_input_hash(batch)
        batch_id = stable_id(batch_source_hash, SETTINGS_SHA256, PROMPT_SHA256, SCHEMA_SHA256)
        marker_started_at = _utc_now()
        state["in_flight"] = {
            "status": "in_flight", "batch_id": batch_id,
            "source_document_ids": [document.id for document in batch],
            "identities": identities, "source_input_sha256": batch_source_hash,
            "requested_settings": _requested_settings(),
            "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
            "schema_version": SCHEMA_VERSION, "schema_sha256": SCHEMA_SHA256,
            "started_at": marker_started_at,
        }
        _atomic_json(state_path, state)

        if isinstance(translation_client, OpenAIResponsesTranslationClient):
            raw_result = translation_client.translate(batch, execute=True)
        else:
            raw_result = translation_client.translate(batch)
        result = _normalize_result(raw_result, batch, request_started_at=marker_started_at)

        new_rows: list[dict[str, Any]] = []
        for document, translated in zip(batch, result.translations):
            identity = identities[document.id]
            metadata = dict(document.metadata)
            metadata["translation_provenance"] = {
                "source_document_id": document.id,
                "source_language": manifest.language, "target_language": "en",
                "model": TRANSLATION_MODEL, "returned_model": result.returned_model,
                "response_id": result.response_id, "usage": result.usage,
                "identity": identity, "identity_sha256": _identity_sha256(identity),
                "input_sha256": identity["source_input_sha256"],
                "batch_source_input_sha256": result.source_input_sha256,
                "batch_output_sha256": result.output_sha256,
                "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
                "schema_version": SCHEMA_VERSION, "schema_sha256": SCHEMA_SHA256,
                "requested_settings": _requested_settings(),
                "request_started_at": result.request_started_at,
                "completed_at": result.completed_at, "translated_at": result.completed_at,
                "translated_fields": ["title", "text", "place", "genre"],
            }
            new_rows.append(Document(
                id=f"translation:{stable_id(TRANSLATION_MODEL, country, document.id, identity['source_input_sha256'])}",
                source=document.source, title=translated["title"], text=translated["text"],
                url=document.url, date=document.date, authors=document.authors,
                place=translated["place"], genre=translated["genre"], rights=document.rights,
                country=country, language="en", content_mode="english_translation",
                metadata=metadata,
            ).asdict())

        replaced_ids = {document.id for document in batch}
        output_rows = [row for row in output_rows
                       if row.get("metadata", {}).get("translation_provenance", {})
                       .get("source_document_id") not in replaced_ids]
        output_rows.extend(new_rows)
        _atomic_jsonl(output_path, output_rows)

        batch_record = {
            **state["in_flight"], "status": "completed",
            "response_id": result.response_id, "returned_model": result.returned_model,
            "usage": result.usage, "output_sha256": result.output_sha256,
            "request_started_at": result.request_started_at,
            "completed_at": result.completed_at,
        }
        state["batches"].append(batch_record)
        for row, document in zip(new_rows, batch):
            state["completed"][document.id] = {
                "identity": identities[document.id],
                "identity_sha256": _identity_sha256(identities[document.id]),
                "translated_document_id": row["id"],
                "response_id": result.response_id,
                "returned_model": result.returned_model, "usage": result.usage,
                "source_input_sha256": result.source_input_sha256,
                "output_sha256": result.output_sha256,
                "request_started_at": result.request_started_at,
                "completed_at": result.completed_at,
            }
        state["in_flight"] = None
        _atomic_json(state_path, state)
        translated_count += len(new_rows)
        response_records.append(batch_record)
    return {**plan, "translated_documents": translated_count, "dry_run": False,
            "responses": response_records}
