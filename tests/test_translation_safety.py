import json

import pytest

from american_history_rag.schema import Document
from american_history_rag.translation import (
    MAX_OUTPUT_TOKENS,
    TRANSLATION_MODEL,
    OpenAIResponsesTranslationClient,
    translate_corpus,
    translation_plan,
)
from american_history_rag.util import read_jsonl, write_jsonl


def native_document(text="Texto original", *, document_id="es:safety"):
    return Document(
        id=document_id,
        source="fixture",
        title="Diario",
        text=text,
        url="https://example.test/source",
        country="spain",
        language="es",
        content_mode="native",
        rights="Public domain test fixture",
    )


def translation_for(document, text="Translated text"):
    return {
        "id": document.id,
        "title": "Diary",
        "text": text,
        "place": "",
        "genre": "",
    }


class RecordingClient:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def translate(self, documents):
        self.calls.append([document.id for document in documents])
        if self.fail:
            raise RuntimeError("connection outcome unknown")
        return [translation_for(document, f"Translated: {document.text}") for document in documents]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def completed_payload(document, **overrides):
    payload = {
        "id": "resp_translation_1",
        "model": TRANSLATION_MODEL,
        "status": "completed",
        "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": json.dumps({"translations": [translation_for(document)]}),
            }],
        }],
    }
    payload.update(overrides)
    return payload


def test_real_responses_request_is_explicit_and_safety_bounded(monkeypatch):
    document = native_document()
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(completed_payload(document))

    monkeypatch.setattr("american_history_rag.translation.requests.post", fake_post)
    client = OpenAIResponsesTranslationClient(api_key="test-key")
    with pytest.raises(RuntimeError, match="execute=True"):
        client.translate([document])
    assert captured == {}

    result = client.translate([document], execute=True)
    request = captured["json"]
    assert request["model"] == TRANSLATION_MODEL == "gpt-5.6-luna"
    assert request["store"] is False
    assert request["tools"] == []
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert result.response_id == "resp_translation_1"
    assert result.returned_model == TRANSLATION_MODEL
    assert result.usage["total_tokens"] == 19
    assert len(result.source_input_sha256) == len(result.output_sha256) == 64


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"status": "incomplete", "model": TRANSLATION_MODEL}, "did not complete"),
        ({"status": "completed", "model": "wrong", "id": "r", "usage": {}},
         "unexpected model"),
        ({"status": "completed", "model": TRANSLATION_MODEL, "error": {"message": "bad"}},
         "error response"),
        ({"status": "completed", "model": TRANSLATION_MODEL, "id": "r", "usage": {},
          "output": [{"content": [{"type": "refusal", "refusal": "no"}]}]}, "refused"),
        ({"status": "completed", "model": TRANSLATION_MODEL, "id": "r", "usage": {},
          "output": []}, "no translation text"),
        ({"status": "completed", "model": TRANSLATION_MODEL, "id": "r", "usage": {},
          "output_text": "not json"}, "malformed translation JSON"),
        ({"status": "completed", "model": TRANSLATION_MODEL, "id": "r", "usage": {},
          "output_text": json.dumps({"translations": []})}, "IDs do not exactly match"),
    ],
)
def test_responses_rejects_unsafe_or_malformed_results(monkeypatch, payload, message):
    monkeypatch.setattr(
        "american_history_rag.translation.requests.post",
        lambda *args, **kwargs: FakeResponse(payload),
    )
    client = OpenAIResponsesTranslationClient(api_key="test-key")
    with pytest.raises(RuntimeError, match=message):
        client.translate([native_document()], execute=True)


def test_dry_run_is_network_and_key_free_and_reports_character_ceiling(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "american_history_rag.translation.requests.post",
        lambda *args, **kwargs: pytest.fail("dry run attempted network access"),
    )
    corpus = tmp_path / "native.jsonl"
    write_jsonl(corpus, [native_document().asdict()])
    report = translate_corpus(
        corpus,
        tmp_path / "english.jsonl",
        tmp_path / "state.json",
        country="spain",
        dry_run=True,
        max_input_characters=17,
    )
    assert report["estimated_input_characters"] > 0
    assert report["max_input_characters"] == 17
    assert report["within_input_character_ceiling"] is False
    assert not (tmp_path / "state.json").exists()


def test_execution_ceiling_and_oversized_document_fail_before_client_call(tmp_path):
    corpus = tmp_path / "native.jsonl"
    write_jsonl(corpus, [native_document("x" * 200).asdict()])
    client = RecordingClient()
    with pytest.raises(ValueError, match="caller-supplied ceiling"):
        translate_corpus(
            corpus, tmp_path / "out.jsonl", tmp_path / "state.json",
            country="spain", client=client, max_input_characters=1,
        )
    assert client.calls == []

    with pytest.raises(ValueError, match="the maximum is"):
        translation_plan(
            corpus, tmp_path / "out.jsonl", tmp_path / "state.json",
            country="spain", max_document_characters=10,
        )
    assert client.calls == []


def test_ambiguous_in_flight_batch_requires_explicit_retry(tmp_path):
    corpus = tmp_path / "native.jsonl"
    output = tmp_path / "english.jsonl"
    state = tmp_path / "state.json"
    write_jsonl(corpus, [native_document().asdict()])
    failing = RecordingClient(fail=True)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        translate_corpus(corpus, output, state, country="spain", client=failing)

    marker = json.loads(state.read_text(encoding="utf-8"))["in_flight"]
    assert marker["status"] == "in_flight"
    assert marker["source_input_sha256"]
    retry = RecordingClient()
    with pytest.raises(RuntimeError, match="ambiguous in-flight"):
        translate_corpus(corpus, output, state, country="spain", client=retry)
    assert retry.calls == []

    result = translate_corpus(
        corpus, output, state, country="spain", client=retry, retry_ambiguous=True,
    )
    assert result["translated_documents"] == 1
    assert json.loads(state.read_text(encoding="utf-8"))["in_flight"] is None


def test_changed_source_replaces_stale_derived_row_atomically(tmp_path):
    corpus = tmp_path / "native.jsonl"
    output = tmp_path / "english.jsonl"
    state = tmp_path / "state.json"
    write_jsonl(corpus, [native_document("first version").asdict()])
    client = RecordingClient()
    translate_corpus(corpus, output, state, country="spain", client=client)
    old_row = list(read_jsonl(output))[0]

    write_jsonl(corpus, [native_document("corrected source").asdict()])
    translate_corpus(corpus, output, state, country="spain", client=client)
    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["text"] == "Translated: corrected source"
    assert rows[0]["id"] != old_row["id"]
    assert client.calls == [["es:safety"], ["es:safety"]]


def test_duplicate_or_partial_batch_never_writes_output(tmp_path):
    class DuplicateClient:
        def translate(self, documents):
            item = translation_for(documents[0])
            return [item, item]

    corpus = tmp_path / "native.jsonl"
    output = tmp_path / "english.jsonl"
    state = tmp_path / "state.json"
    write_jsonl(corpus, [native_document().asdict()])
    with pytest.raises(RuntimeError, match="IDs do not exactly match"):
        translate_corpus(corpus, output, state, country="spain", client=DuplicateClient())
    assert not output.exists()
    assert json.loads(state.read_text(encoding="utf-8"))["in_flight"]


def test_response_audit_fields_are_returned_and_persisted(tmp_path, monkeypatch):
    document = native_document()
    monkeypatch.setattr(
        "american_history_rag.translation.requests.post",
        lambda *args, **kwargs: FakeResponse(completed_payload(document)),
    )
    corpus = tmp_path / "native.jsonl"
    output = tmp_path / "english.jsonl"
    state_path = tmp_path / "state.json"
    write_jsonl(corpus, [document.asdict()])
    report = translate_corpus(
        corpus, output, state_path, country="spain",
        client=OpenAIResponsesTranslationClient(api_key="test-key"), execute=True,
    )
    response = report["responses"][0]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    provenance = list(read_jsonl(output))[0]["metadata"]["translation_provenance"]
    assert response["status"] == "completed"
    assert response["response_id"] == provenance["response_id"] == "resp_translation_1"
    assert response["returned_model"] == provenance["returned_model"] == TRANSLATION_MODEL
    assert response["usage"] == provenance["usage"]
    assert response["source_input_sha256"] == provenance["batch_source_input_sha256"]
    assert response["output_sha256"] == provenance["batch_output_sha256"]
    assert response["request_started_at"] and response["completed_at"]
    assert provenance["prompt_sha256"] and provenance["schema_sha256"]
    assert provenance["requested_settings"]["store"] is False
    assert state["completed"][document.id]["response_id"] == "resp_translation_1"
