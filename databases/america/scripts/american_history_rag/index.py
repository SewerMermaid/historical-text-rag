from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .filtering import classify_chronicling_chunk
from .schema import Chunk, Document
from .util import read_jsonl, stable_id, write_jsonl


DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "
LEGACY_BGE_MODEL = "BAAI/bge-small-en-v1.5"
LEGACY_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_SCHEMA_VERSION = 2
_CJK_HANGUL_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _model_prefixes(model_name: str) -> tuple[str, str]:
    """Return prefixes required by models whose retrieval format is known."""
    if re.search(r"(?:^|[/_-])(?:multilingual-)?e5(?:$|[/_-])", model_name.casefold()):
        return QUERY_PREFIX, DOCUMENT_PREFIX
    if model_name.casefold() == LEGACY_BGE_MODEL.casefold():
        return LEGACY_BGE_QUERY_PREFIX, ""
    return "", ""


def _split_long_paragraph(paragraph: str, target_words: int,
                          overlap_words: int) -> list[str] | None:
    if _CJK_HANGUL_RE.search(paragraph) and len(paragraph) > target_words:
        step = target_words - overlap_words
        return [paragraph[start:start + target_words]
                for start in range(0, len(paragraph) - overlap_words, step)]
    words = paragraph.split()
    if len(words) > target_words:
        step = target_words - overlap_words
        return [" ".join(words[start:start + target_words])
                for start in range(0, len(words) - overlap_words, step)]
    return None


def chunk_document(doc: Document, target_words: int = 350, overlap_words: int = 50) -> list[Chunk]:
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be non-negative and smaller than target_words")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    units: list[str | list[str]] = []
    for paragraph in paragraphs:
        units.append(_split_long_paragraph(paragraph, target_words, overlap_words) or paragraph)
    for unit in units:
        if isinstance(unit, list):
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            chunks.extend(unit)
            continue
        paragraph = unit
        words = paragraph.split()
        if current and current_words + len(words) > target_words:
            chunks.append("\n\n".join(current))
            available_overlap = min(overlap_words, max(0, target_words - len(words)))
            tail = " ".join("\n\n".join(current).split()[-available_overlap:]) if available_overlap else ""
            current = [tail, paragraph] if tail else [paragraph]
            current_words = len(tail.split()) + len(words)
        else:
            current.append(paragraph)
            current_words += len(words)
    if current:
        chunks.append("\n\n".join(current))
    return [Chunk(
        id=stable_id(doc.id, str(i), text), document_id=doc.id, source=doc.source,
        title=doc.title, text=text, url=doc.url, date=doc.date, authors=doc.authors,
        place=doc.place, genre=doc.genre, rights=doc.rights, country=doc.country,
        language=doc.language, content_mode=doc.content_mode,
        metadata=doc.metadata, ordinal=i,
    ) for i, text in enumerate(chunks)]


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, *, local_files_only: bool = False,
                 query_prefix: str | None = None, document_prefix: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies before building the index") from exc
        self.model_name = model_name
        inferred_query_prefix, inferred_document_prefix = _model_prefixes(model_name)
        self.query_prefix = inferred_query_prefix if query_prefix is None else query_prefix
        self.document_prefix = inferred_document_prefix if document_prefix is None else document_prefix
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        passages = [self.document_prefix + text for text in texts]
        return np.asarray(self.model.encode(passages, batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0], dtype=np.float32)


def _fts_terms(text: str) -> list[str]:
    """Add searchable character n-grams for scripts FTS5 does not segment."""
    output: list[str] = []
    for term in lexical_terms(text):
        output.append(term)
        if _CJK_HANGUL_RE.search(term):
            characters = list(term)
            output.extend(characters)
            output.extend("".join(characters[index:index + 2])
                          for index in range(len(characters) - 1))
    return list(dict.fromkeys(output))


def _write_lexical_index(path: Path, chunks: list[Chunk]) -> int:
    database = sqlite3.connect(path)
    try:
        database.execute("CREATE VIRTUAL TABLE chunks USING fts5("
                         "id UNINDEXED, text, title, source UNINDEXED, date UNINDEXED, lexical, "
                         "tokenize='unicode61 remove_diacritics 2')")
        database.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", (
            (chunk.id, chunk.text, chunk.title, chunk.source, chunk.date,
             " ".join(_fts_terms(f"{chunk.title} {chunk.text}")))
            for chunk in chunks
        ))
        count = int(database.execute("SELECT count(*) FROM chunks").fetchone()[0])
        database.commit()
        return count
    finally:
        database.close()


def _promote_index(staging_dir: Path, index_dir: Path) -> None:
    backup_dir = index_dir.with_name(f".{index_dir.name}.backup")
    if backup_dir.exists():
        raise RuntimeError(f"Refusing to overwrite stale index backup: {backup_dir}")
    if not index_dir.exists():
        os.replace(staging_dir, index_dir)
        return
    os.replace(index_dir, backup_dir)
    try:
        os.replace(staging_dir, index_dir)
    except BaseException:
        os.replace(backup_dir, index_dir)
        raise
    shutil.rmtree(backup_dir)


def build_index(corpus_path: Path, index_dir: Path, *, model_name: str = DEFAULT_EMBED_MODEL,
                target_words: int = 350, overlap_words: int = 50, batch_size: int = 32,
                expected_country: str | None = None,
                expected_content_mode: str | None = None) -> int:
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    excluded_chunks: list[dict] = []
    for raw in read_jsonl(corpus_path):
        document = Document(**raw)
        if expected_country is not None and document.country != expected_country:
            raise ValueError(
                f"Corpus country mismatch: expected {expected_country!r}, found {document.country!r}"
            )
        if expected_content_mode is not None and document.content_mode != expected_content_mode:
            raise ValueError(
                "Corpus content mode mismatch: "
                f"expected {expected_content_mode!r}, found {document.content_mode!r}"
            )
        for chunk in chunk_document(document, target_words, overlap_words):
            if document.source in {"loc_chronicling_america", "chronicling_america"}:
                decision = classify_chronicling_chunk(chunk.text)
                if decision["status"] == "literary":
                    excluded_chunks.append({**chunk.asdict(), "filter_decision": decision})
                    continue
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No documents/chunks found")
    query_prefix, document_prefix = _model_prefixes(model_name)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{index_dir.name}.staging.", dir=index_dir.parent))
    try:
        write_jsonl(staging_dir / "chunks.jsonl", (chunk.asdict() for chunk in chunks))
        write_jsonl(staging_dir / "excluded_chunks.jsonl", iter(excluded_chunks))
        embedder = Embedder(model_name, query_prefix=query_prefix, document_prefix=document_prefix)
        vectors = embedder.documents([chunk.text for chunk in chunks], batch_size=batch_size)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding shape mismatch: expected {len(chunks)} rows, got {vectors.shape}"
            )
        np.save(staging_dir / "embeddings.npy", vectors)
        lexical_rows = _write_lexical_index(staging_dir / "lexical.sqlite", chunks)
        countries = sorted({chunk.country for chunk in chunks})
        languages = sorted({chunk.language for chunk in chunks})
        content_modes = sorted({chunk.content_mode for chunk in chunks})
        (staging_dir / "config.json").write_text(json.dumps({
            "schema_version": INDEX_SCHEMA_VERSION,
            "model": model_name,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "lexical_tokenizer": "unicode61",
            "lexical_expansion": "remove_diacritics 2 with CJK/Hangul 1-2 grams",
            "chunks": len(chunks),
            "target_words": target_words,
            "overlap_words": overlap_words,
            "excluded_chronicling_chunks": len(excluded_chunks),
            "country": countries[0] if len(countries) == 1 else None,
            "content_mode": content_modes[0] if len(content_modes) == 1 else None,
            "countries": countries,
            "languages": languages,
            "content_modes": content_modes,
            "integrity": {
                "chunk_rows": len(chunks),
                "embedding_rows": int(vectors.shape[0]),
                "embedding_dimensions": int(vectors.shape[1]),
                "lexical_rows": lexical_rows,
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        _promote_index(staging_dir, index_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return len(chunks)


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def lexical_terms(query: str) -> list[str]:
    """Unicode-aware terms for FTS5; accented and CJK text must not disappear."""
    return re.findall(r"[^\W_]+", query, flags=re.UNICODE)


def search(index_dir: Path, query: str, *, top_k: int = 8, source: str = "",
           date_from: int | None = None, date_to: int | None = None,
           max_per_document: int = 2,
           lexical_tokenizer: Callable[[str], list[str]] = lexical_terms,
           expected_country: str | None = None,
           expected_content_mode: str | None = None) -> list[dict]:
    chunks = list(read_jsonl(index_dir / "chunks.jsonl"))
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    indexed_countries = set(config.get("countries") or (chunk.get("country", "america") for chunk in chunks))
    indexed_modes = set(config.get("content_modes") or (chunk.get("content_mode", "native") for chunk in chunks))
    if expected_country is not None and indexed_countries != {expected_country}:
        raise ValueError(
            f"Index country mismatch: expected {expected_country!r}, found {sorted(indexed_countries)!r}"
        )
    if expected_content_mode is not None and indexed_modes != {expected_content_mode}:
        raise ValueError(
            "Index content mode mismatch: "
            f"expected {expected_content_mode!r}, found {sorted(indexed_modes)!r}"
        )
    vectors = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    query_prefix = config.get("query_prefix")
    document_prefix = config.get("document_prefix")
    query_vector = Embedder(
        config["model"], local_files_only=True,
        query_prefix=query_prefix if isinstance(query_prefix, str) else None,
        document_prefix=document_prefix if isinstance(document_prefix, str) else None,
    ).query(query)
    dense_scores = np.asarray(vectors @ query_vector)
    eligible: list[int] = []
    for idx, chunk in enumerate(chunks):
        year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", chunk.get("date", ""))
        year = int(year_match.group(1)) if year_match else None
        if source and chunk["source"] != source:
            continue
        if date_from is not None and year is not None and year < date_from:
            continue
        if date_to is not None and year is not None and year > date_to:
            continue
        eligible.append(idx)
    dense_rank = sorted(eligible, key=lambda i: float(dense_scores[i]), reverse=True)[:max(50, top_k * 5)]
    id_to_idx = {chunk["id"]: idx for idx, chunk in enumerate(chunks)}
    eligible_set = set(eligible)
    terms = []
    for term in lexical_tokenizer(query):
        terms.extend(_fts_terms(term))
    terms = list(dict.fromkeys(terms))
    lexical_rank: list[int] = []
    if terms:
        fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        database = sqlite3.connect(index_dir / "lexical.sqlite")
        try:
            rows = database.execute(
                "SELECT id FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT 100",
                (fts_query,),
            ).fetchall()
            lexical_rank = [id_to_idx[row[0]] for row in rows
                            if row[0] in id_to_idx and id_to_idx[row[0]] in eligible_set]
        finally:
            database.close()
    fused = _rrf([dense_rank, lexical_rank])
    output = []
    document_counts: dict[str, int] = {}
    for idx in sorted(fused, key=fused.get, reverse=True):
        row = dict(chunks[idx])
        document_id = row["document_id"]
        if document_counts.get(document_id, 0) >= max_per_document:
            continue
        document_counts[document_id] = document_counts.get(document_id, 0) + 1
        row["score"] = fused[idx]
        row["dense_score"] = float(dense_scores[idx])
        output.append(row)
        if len(output) >= top_k:
            break
    return output


def context_prompt(question: str, results: list[dict]) -> str:
    evidence = []
    for i, row in enumerate(results, 1):
        mode = row.get("content_mode", "native")
        details = (
            f"Country: {row.get('country', 'unknown')} | Language: {row.get('language', 'unknown')} "
            f"| Content mode: {mode} | Document ID: {row.get('document_id', 'unknown')}"
        )
        warning = ""
        if mode == "english_translation":
            provenance = row.get("metadata", {}).get("translation_provenance", {})
            native_id = provenance.get("source_document_id", "unknown")
            warning = (
                f"\nMACHINE TRANSLATION: verify wording and quotations against native document {native_id}."
            )
        evidence.append(
            f"[S{i}] {row['title']} ({row.get('date') or 'undated'})\n{details}"
            f"{warning}\nURL: {row['url']}\n{row['text']}"
        )
    return (
        "Answer the historical question using only the evidence below. Cite claims with [S1], [S2], etc. "
        "Distinguish contemporaneous evidence from recollection, report disagreement, and say when the "
        "evidence is insufficient. Answer in the language of the question and do not generalize from a small sample.\n\n"
        f"QUESTION:\n{question}\n\nEVIDENCE:\n" + "\n\n".join(evidence)
    )
