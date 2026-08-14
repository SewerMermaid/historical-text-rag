import json

from american_history_rag.countries import country_manifest
from american_history_rag.index import DEFAULT_EMBED_MODEL, chunk_document, lexical_terms
from american_history_rag.multicountry_sources import parse_oai_page
from american_history_rag.schema import Document
from american_history_rag.translation import TRANSLATION_MODEL, translate_corpus
from american_history_rag.util import read_jsonl, write_jsonl


def _native_spanish() -> Document:
    return Document(
        id="es:1", source="fixture", title="Diario",
        text="La familia describió el precio del azúcar y el trabajo cotidiano.",
        url="https://example.test/es/1", date="1898", country="spain",
        language="es", content_mode="native", rights="Public domain fixture",
    )


def test_country_manifests_are_native_and_rights_cautious():
    spain = country_manifest("spain")
    portugal = country_manifest("portugal")
    assert (spain.language, portugal.language) == ("es", "pt")
    assert "Como" in portugal.native_query_default
    assert spain.publication_cutoff == portugal.publication_cutoff == 1929
    assert "not a copyright determination" in spain.cutoff_note
    assert spain.source("bne-bdh").protocol == "oai-pmh"
    assert portugal.source("bnp-digital-public-domain").options["set"] == "bndlivre_digitalizadas"


def test_multilingual_e5_and_chunk_language_provenance():
    assert DEFAULT_EMBED_MODEL == "intfloat/multilingual-e5-small"
    chunk = chunk_document(_native_spanish(), target_words=100)[0]
    assert (chunk.country, chunk.language, chunk.content_mode) == ("spain", "es", "native")


def test_lexical_terms_keep_spanish_portuguese_accents_and_cjk():
    assert lexical_terms("azúcar, coração e revolução") == ["azúcar", "coração", "e", "revolução"]
    assert lexical_terms("工人生活") == ["工人生活"]


def test_oai_parser_retains_dublin_core_and_resumption():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
      xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <ListRecords><record><header><identifier>oai:test:1</identifier></header>
      <metadata><oai_dc:dc><dc:title>Memórias</dc:title><dc:date>1891</dc:date>
      <dc:rights>Domínio público</dc:rights></oai_dc:dc></metadata></record>
      <resumptionToken>next-token</resumptionToken></ListRecords>
    </OAI-PMH>"""
    records, token = parse_oai_page(xml)
    assert records[0]["fields"]["title"] == ["Memórias"]
    assert records[0]["fields"]["rights"] == ["Domínio público"]
    assert token == "next-token"


class FakeTranslationClient:
    def __init__(self):
        self.calls = []

    def translate(self, documents):
        self.calls.append([document.id for document in documents])
        return [{
            "id": document.id, "title": "Diary", "text": "The family described work.",
            "place": "", "genre": "",
        } for document in documents]


def test_translation_is_bounded_resumable_and_never_reuses_native_id(tmp_path):
    corpus = tmp_path / "native.jsonl"
    output = tmp_path / "english.jsonl"
    state = tmp_path / "state.json"
    write_jsonl(corpus, [_native_spanish().asdict()])
    client = FakeTranslationClient()

    first = translate_corpus(corpus, output, state, country="spain", batch_size=1, client=client)
    second = translate_corpus(corpus, output, state, country="spain", batch_size=1, client=client)

    translated = list(read_jsonl(output))
    assert first["translated_documents"] == 1
    assert second["translated_documents"] == 0
    assert client.calls == [["es:1"]]
    assert translated[0]["id"] != "es:1"
    assert translated[0]["content_mode"] == "english_translation"
    provenance = translated[0]["metadata"]["translation_provenance"]
    assert provenance["source_document_id"] == "es:1"
    assert provenance["model"] == TRANSLATION_MODEL == "gpt-5.6-luna"
    assert json.loads(state.read_text(encoding="utf-8"))["completed"]["es:1"]


def test_translation_dry_run_needs_no_client_or_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    corpus = tmp_path / "native.jsonl"
    write_jsonl(corpus, [_native_spanish().asdict()])
    report = translate_corpus(corpus, tmp_path / "english.jsonl", tmp_path / "state.json",
                              country="spain", batch_size=1, dry_run=True)
    assert report["planned_batches"] == 1
    assert report["translated_documents"] == 0
    assert not (tmp_path / "state.json").exists()
