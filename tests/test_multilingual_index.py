from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from american_history_rag import index as index_module
from american_history_rag.index import (
    Embedder,
    build_index,
    chunk_document,
    context_prompt,
    search,
)
from american_history_rag.schema import Document
from american_history_rag.util import write_jsonl


class FakeEmbedder:
    calls: list[dict] = []

    def __init__(self, model_name, **kwargs):
        self.calls.append({"model": model_name, **kwargs})

    def documents(self, texts, batch_size=32):
        return np.zeros((len(texts), 2), dtype=np.float32)

    def query(self, text):
        return np.zeros(2, dtype=np.float32)


def _document(identifier: str, text: str, *, country="china", language="zh",
              content_mode="native") -> dict:
    return Document(
        id=identifier, source="fixture", title=f"Title {identifier}", text=text,
        url=f"https://example.test/{identifier}", country=country,
        language=language, content_mode=content_mode,
    ).asdict()


@pytest.mark.parametrize("text", [
    "工人生活与社会变迁历史记录工人生活与社会变迁历史记录",
    "労働者の日常生活と社会変化の歴史記録労働者の日常生活と社会変化の歴史記録",
    "노동자의일상생활과사회변화에관한역사기록노동자의일상생활과사회변화에관한역사기록",
])
def test_unspaced_cjk_and_hangul_chunking_is_bounded_overlapping_and_lossless(text):
    document = Document(id="cjk", source="fixture", title="Record", text=text,
                        url="https://example.test/cjk")
    chunks = chunk_document(document, target_words=16, overlap_words=4)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 16 for chunk in chunks)
    assert all(left.text[-4:] == right.text[:4] for left, right in zip(chunks, chunks[1:]))
    reconstructed = chunks[0].text + "".join(chunk.text[4:] for chunk in chunks[1:])
    assert reconstructed == text


def test_mixed_whitespace_cjk_still_obeys_character_bound():
    text = " ".join(["工人生活與社會變遷" * 12, "物價與家庭記錄" * 12] * 8)
    chunks = chunk_document(
        Document(id="mixed", source="fixture", title="記錄", text=text,
                 url="https://example.test/mixed"),
        target_words=80, overlap_words=10,
    )
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 80 for chunk in chunks)


@pytest.mark.parametrize(("indexed_text", "query"), [
    ("O preço do açúcar subiu no mercado.", "acucar"),
    ("帝国工人生活史料汇编", "工人生活"),
    ("조선노동자생활사기록", "노동자"),
])
def test_search_gets_sqlite_fts_candidates_for_accents_and_internal_terms(
        tmp_path, monkeypatch, indexed_text, query):
    monkeypatch.setattr(index_module, "Embedder", FakeEmbedder)
    corpus = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    write_jsonl(corpus, [
        _document("distractor", "An unrelated first record."),
        _document("match", indexed_text),
    ])

    build_index(corpus, index_dir)
    results = search(index_dir, query, top_k=1)

    assert results[0]["document_id"] == "match"


def test_embedder_prefixes_are_model_specific(monkeypatch):
    encoded: list[list[str]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, local_files_only=False):
            pass

        def encode(self, texts, **kwargs):
            encoded.append(texts)
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    e5 = Embedder("intfloat/multilingual-e5-small")
    e5.documents(["document"])
    e5.query("question")
    arbitrary = Embedder("example/arbitrary-model")
    arbitrary.documents(["document"])
    arbitrary.query("question")
    arbitrary_ending_in_e5 = Embedder("example/example5")
    arbitrary_ending_in_e5.documents(["document"])
    arbitrary_ending_in_e5.query("question")

    assert encoded == [
        ["passage: document"], ["query: question"],
        ["document"], ["question"],
        ["document"], ["question"],
    ]


def test_search_honors_legacy_config_query_prefix_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "Embedder", FakeEmbedder)
    FakeEmbedder.calls.clear()
    corpus = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    write_jsonl(corpus, [_document("legacy", "A historical record.", country="america", language="en")])
    build_index(corpus, index_dir, model_name="BAAI/bge-small-en-v1.5")
    config_path = index_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["query_prefix"] = "EXACT LEGACY PREFIX: "
    config_path.write_text(json.dumps(config), encoding="utf-8")

    search(index_dir, "question")

    assert FakeEmbedder.calls[-1]["query_prefix"] == "EXACT LEGACY PREFIX: "


def test_expected_country_and_mode_reject_wrong_corpus_and_index(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "Embedder", FakeEmbedder)
    corpus = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    write_jsonl(corpus, [_document("china", "史料")])

    with pytest.raises(ValueError, match="country mismatch"):
        build_index(corpus, index_dir, expected_country="japan")

    build_index(
        corpus, index_dir, expected_country="china", expected_content_mode="native"
    )
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == 2
    assert config["country"] == "china"
    assert config["content_mode"] == "native"
    assert config["integrity"] == {
        "chunk_rows": 1,
        "embedding_rows": 1,
        "embedding_dimensions": 2,
        "lexical_rows": 1,
    }
    with pytest.raises(ValueError, match="content mode mismatch"):
        search(index_dir, "史料", expected_country="china",
               expected_content_mode="english_translation")


def test_failed_promotion_restores_the_previous_complete_index(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "Embedder", FakeEmbedder)
    corpus = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    write_jsonl(corpus, [_document("old", "旧记录")])
    build_index(corpus, index_dir)
    old_config = (index_dir / "config.json").read_bytes()
    old_chunks = (index_dir / "chunks.jsonl").read_bytes()

    write_jsonl(corpus, [_document("new", "新记录")])
    real_replace = os.replace
    failed = False

    def fail_staging_promotion(source, destination):
        nonlocal failed
        source_path = os.fspath(source)
        if not failed and ".staging." in source_path and os.fspath(destination) == os.fspath(index_dir):
            failed = True
            raise OSError("injected promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(index_module.os, "replace", fail_staging_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        build_index(corpus, index_dir)

    assert (index_dir / "config.json").read_bytes() == old_config
    assert (index_dir / "chunks.jsonl").read_bytes() == old_chunks
    assert search(index_dir, "记录", top_k=1)[0]["document_id"] == "old"


def test_context_prompt_labels_machine_translation_and_native_identity():
    prompt = context_prompt("What changed?", [{
        "title": "Diary", "date": "1890", "url": "https://example.test/native",
        "text": "Translated evidence", "country": "spain", "language": "en",
        "content_mode": "english_translation", "document_id": "translation:1",
        "metadata": {"translation_provenance": {"source_document_id": "native:1"}},
    }])
    assert "MACHINE TRANSLATION" in prompt
    assert "native document native:1" in prompt
    assert "Country: spain" in prompt
    assert "Content mode: english_translation" in prompt
