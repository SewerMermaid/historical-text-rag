from american_history_rag.index import _rrf, chunk_document
from american_history_rag.filtering import (classify_chronicling_chunk,
                                             classify_chronicling_metadata,
                                             classify_loc_nonfiction,
                                             evans_exclusion_reasons)
from american_history_rag.schema import Document
from american_history_rag.sources import _year
from american_history_rag.util import write_jsonl


def test_year():
    assert _year("January 17, 1898") == 1898
    assert _year("undated") is None


def test_chunk_preserves_citation():
    doc = Document(id="d1", source="test", title="Diary",
                   text="One two three.\n\nFour five six.", url="https://example.test/d1")
    chunks = chunk_document(doc, target_words=4, overlap_words=1)
    assert len(chunks) == 2
    assert all(chunk.url == doc.url for chunk in chunks)


def test_rrf_rewards_agreement():
    scores = _rrf([[1, 2], [1, 3]])
    assert scores[1] > scores[2]


def test_oversized_ocr_page_is_split():
    doc = Document(id="ocr", source="test", title="Page",
                   text=" ".join(["word"] * 1_000), url="https://example.test/page")
    chunks = chunk_document(doc, target_words=350, overlap_words=50)
    assert len(chunks) == 4
    assert max(len(chunk.text.split()) for chunk in chunks) <= 350


def test_loc_filter_excludes_explicit_fiction():
    result = classify_loc_nonfiction({
        "title": "A story",
        "subject": ["Domestic fiction"],
        "item": {"call_number": ["PZ3 .A1"]},
    })
    assert result["status"] == "fiction"


def test_loc_filter_keeps_documentary_history():
    result = classify_loc_nonfiction({
        "title": "Annual report of factory conditions",
        "subject": ["Labor", "Social conditions"],
        "item": {"call_number": ["HD8051 .A2"]},
    })
    assert result["status"] == "nonfiction"


def test_loc_filter_quarantines_ambiguous_literature():
    result = classify_loc_nonfiction({"title": "Collected works", "item": {"call_number": ["PS1000 .A1"]}})
    assert result["status"] == "uncertain"


def test_evans_review_flag_is_non_destructive_signal():
    reasons = evans_exclusion_reasons("A book of Psalms", ["Psalmody"])
    assert reasons


def test_chronicling_metadata_requires_newspaper_signal():
    kept = classify_chronicling_metadata({"title": "Daily Tribune", "genre": ["Newspapers"]})
    rejected = classify_chronicling_metadata({"title": "A Literary Magazine", "genre": ["Periodicals"]})
    assert kept["status"] == "newspaper"
    assert rejected["status"] == "literary"


def test_chronicling_chunk_rejects_serialized_fiction_marker():
    assert classify_chronicling_chunk("CHAPTER XII\nThe stranger entered.\nTo be continued.")["status"] == "literary"
    assert classify_chronicling_chunk("The market price of flour rose yesterday.")["status"] == "retain"


def test_overwrite_jsonl_is_atomic_on_generator_failure(tmp_path):
    output = tmp_path / "corpus.jsonl"
    output.write_text('{"old": true}\n', encoding="utf-8")

    def broken_rows():
        yield {"new": 1}
        raise RuntimeError("remote API failed")

    try:
        write_jsonl(output, broken_rows())
    except RuntimeError:
        pass

    assert output.read_text(encoding="utf-8") == '{"old": true}\n'
