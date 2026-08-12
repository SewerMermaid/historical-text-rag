from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from .filtering import classify_chronicling_chunk
from .schema import Chunk, Document
from .util import read_jsonl, stable_id, write_jsonl


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chunk_document(doc: Document, target_words: int = 350, overlap_words: int = 50) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    units: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= target_words:
            units.append(paragraph)
            continue
        unit_size = max(1, target_words - overlap_words)
        units.extend(" ".join(words[start:start + unit_size])
                     for start in range(0, len(words), unit_size)
                     if words[start:start + unit_size])
    for paragraph in units:
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
        place=doc.place, genre=doc.genre, rights=doc.rights, ordinal=i,
    ) for i, text in enumerate(chunks)]


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, *, local_files_only: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies before building the index") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True)[0], dtype=np.float32)


def build_index(corpus_path: Path, index_dir: Path, *, model_name: str = DEFAULT_EMBED_MODEL,
                target_words: int = 350, overlap_words: int = 50, batch_size: int = 32) -> int:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    excluded_chunks: list[dict] = []
    for raw in read_jsonl(corpus_path):
        document = Document(**raw)
        for chunk in chunk_document(document, target_words, overlap_words):
            if document.source in {"loc_chronicling_america", "chronicling_america"}:
                decision = classify_chronicling_chunk(chunk.text)
                if decision["status"] == "literary":
                    excluded_chunks.append({**chunk.asdict(), "filter_decision": decision})
                    continue
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No documents/chunks found")
    write_jsonl(index_dir / "chunks.jsonl", (chunk.asdict() for chunk in chunks))
    write_jsonl(index_dir / "excluded_chunks.jsonl", iter(excluded_chunks))
    embedder = Embedder(model_name)
    vectors = embedder.documents([chunk.text for chunk in chunks], batch_size=batch_size)
    np.save(index_dir / "embeddings.npy", vectors)
    database = sqlite3.connect(index_dir / "lexical.sqlite")
    try:
        database.execute("DROP TABLE IF EXISTS chunks")
        database.execute("CREATE VIRTUAL TABLE chunks USING fts5(id UNINDEXED, text, title, source UNINDEXED, date UNINDEXED)")
        database.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
            ((c.id, c.text, c.title, c.source, c.date) for c in chunks))
        database.commit()
    finally:
        database.close()
    (index_dir / "config.json").write_text(json.dumps({
        "model": model_name, "query_prefix": QUERY_PREFIX, "chunks": len(chunks),
        "target_words": target_words, "overlap_words": overlap_words,
        "excluded_chronicling_chunks": len(excluded_chunks),
    }, indent=2), encoding="utf-8")
    return len(chunks)


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def search(index_dir: Path, query: str, *, top_k: int = 8, source: str = "",
           date_from: int | None = None, date_to: int | None = None,
           max_per_document: int = 2) -> list[dict]:
    chunks = list(read_jsonl(index_dir / "chunks.jsonl"))
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    vectors = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    query_vector = Embedder(config["model"], local_files_only=True).query(query)
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
    terms = re.findall(r"[A-Za-z0-9]+", query)
    lexical_rank: list[int] = []
    if terms:
        fts_query = " OR ".join(f'"{term}"' for term in terms)
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
        evidence.append(f"[S{i}] {row['title']} ({row.get('date') or 'undated'})\nURL: {row['url']}\n{row['text']}")
    return (
        "Answer the historical question using only the evidence below. Cite claims with [S1], [S2], etc. "
        "Distinguish contemporaneous evidence from recollection, report disagreement, and say when the "
        "evidence is insufficient. Do not generalize to all Americans from a small sample.\n\n"
        f"QUESTION:\n{question}\n\nEVIDENCE:\n" + "\n\n".join(evidence)
    )
