import json
from pathlib import Path

import pytest

from american_history_rag.countries import (
    country_manifest,
    data_dir,
    supported_countries,
)
from american_history_rag.multicountry_sources import (
    _oai_documents,
    collect_country_source,
    rights_are_eligible,
)


REQUESTED = {
    "spain", "portugal", "france", "germany", "uk", "china", "japan",
    "south-korea", "mexico", "brazil",
}


def test_all_requested_countries_are_registered_and_paths_are_repo_anchored(monkeypatch, tmp_path):
    assert REQUESTED <= set(supported_countries())
    before = data_dir("france")
    monkeypatch.chdir(tmp_path)
    assert data_dir("france") == before
    assert before.is_absolute()


def test_country_identifier_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid country"):
        country_manifest("../spain")


@pytest.mark.parametrize("rights", [
    "Not in the public domain",
    "Creative Commons licensing unavailable",
    "Creative Commons BY-ND 4.0 — NoDerivatives",
    "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "Public domain status unknown",
])
def test_negative_or_non_derivative_rights_fail_closed(rights):
    assert rights_are_eligible(rights) is False


@pytest.mark.parametrize("rights", [
    "Public domain",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/licenses/by-sa/4.0/",
    "Open Parliament Licence v3.0",
])
def test_affirmative_derivative_rights_are_eligible(rights):
    assert rights_are_eligible(rights) is True


@pytest.mark.parametrize("country", sorted(REQUESTED - {"mexico"}))
def test_enabled_country_sources_support_audited_local_exports(country, tmp_path):
    manifest = country_manifest(country)
    source = next(source for source in manifest.sources if source.status == "enabled")
    export = tmp_path / country
    export.mkdir()
    (export / "text.txt").write_text(
        (manifest.native_query_default + " ") * 20, encoding="utf-8"
    )
    (export / "records.jsonl").write_text(json.dumps({
        "id": "fixture-1", "source": source.id, "path": "text.txt",
        "title": "Fixture", "date": str(manifest.publication_cutoff),
        "url": f"https://example.test/{country}/fixture-1",
        "rights": "Public domain test fixture",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = list(collect_country_source(
        country, source.id, 1, input_root=export, max_bytes=100_000,
    ))
    assert len(rows) == 1
    assert (rows[0].country, rows[0].language, rows[0].content_mode) == (
        country, manifest.language, "native",
    )
    assert rows[0].metadata["rights_status"] == "source_record"


@pytest.mark.parametrize("source_id", ["uanl-digital", "hndm"])
def test_conditional_and_disabled_sources_fail_before_network(source_id):
    with pytest.raises(ValueError, match="conditional|disabled"):
        collect_country_source("mexico", source_id, 1)


def test_oai_missing_rights_is_quarantined():
    country = country_manifest("spain")
    source = country.source("bne-bdh")
    payload = """<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
      xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <ListRecords><record><header><identifier>oai:test:1</identifier></header>
      <metadata><oai_dc:dc><dc:title>Diario</dc:title><dc:date>1891</dc:date>
      <dc:identifier>https://example.test/diario.txt</dc:identifier>
      </oai_dc:dc></metadata></record></ListRecords>
    </OAI-PMH>"""
    text_fetches = []

    def metadata_fetch(url: str, cache: Path | None) -> str:
        return payload

    def text_fetch(url: str, cache: Path | None) -> str:
        text_fetches.append(url)
        return "historical text " * 20

    assert list(_oai_documents(
        country, source, 1, cache_dir=None, max_bytes=100_000,
        fetch_metadata=metadata_fetch, fetch_text=text_fetch,
    )) == []
    # The record is rejected before downloading the full text.
    assert text_fetches == []
