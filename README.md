# Multicountry Historical-Text Hybrid RAG

A bounded, resumable corpus builder and citation-preserving hybrid retriever for
historical texts from the United States, Spain, Portugal, France, Germany, the
United Kingdom, China, Japan, South Korea, Mexico, and Brazil. Each configured
country has isolated native-language and English-rendering corpus/index modes.
Mexico is intentionally source-blocked until a reusable bulk full-text route is validated.

Country-specific state and source manifests are separated under `databases/<country>`.
The backward-compatible Python package remains under `databases/america/scripts/`;
indexing, translation, and retrieval are country-aware.

It supports:

- Chronicling America through the supported `loc.gov` API
- Library of Congress Selected Digitized Books through its data-package manifest
- Evans-TCP from its bulk TEI/XML repository
- Founders Online through its metadata and document APIs
- Dense retrieval with `intfloat/multilingual-e5-small` and its query/passage prefixes
- SQLite FTS5 lexical retrieval, combined with dense rankings using reciprocal-rank fusion
- Unicode-aware lexical query terms, including accented Spanish/Portuguese and CJK
- Date/source filtering and grounded prompts with primary-source URLs
- Result diversification so one long book or newspaper page cannot monopolize the evidence
- Conservative LOC nonfiction filtering with retained reasons and an uncertainty quarantine
- Non-binding Evans and Chronicling metadata audit commands

## Why hybrid retrieval?

Historical queries need semantic retrieval, while names, prices, occupations, archaic spelling, and imperfect OCR often benefit from exact lexical matching. The pipeline runs both and fuses their rankings.

## Install

Python 3.10-3.12 is recommended because ML-library support for very new Python releases can lag.

```powershell
cd C:\path\to\american-history-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The first index build downloads the multilingual embedding model. After that,
document embeddings are computed once and persisted. Each search embeds only the query.
No API key is needed for harvesting, native indexing, or native querying.

## Countries and source status

The checked-in manifests record language, a native pilot query, publication cutoff,
protocol endpoint, availability, and a source-level rights caution. Inspect them with:

```powershell
history-rag list-sources --country spain
history-rag list-sources --country portugal
history-rag list-sources --country japan
```

| Country | Native language | Current ingestion route |
|---|---|---|
| Spain | Spanish | BNE/Prensa Histórica OAI-PMH; Internet Archive OCR |
| Portugal | Portuguese | BNP public-domain OAI set; Internet Archive OCR |
| France | French | audited Gallica text/ALTO export |
| Germany | German | audited DTA TEI/plain-text export |
| United Kingdom | English | audited Historic Hansard or EEBO-TCP export |
| China | Chinese | audited Kanripo UTF-8 export; CText disabled |
| Japan | Japanese | audited NDL rights-expired OCR or Aozora export |
| South Korea | Korean/Classical Chinese | audited NIKH Annals XML export |
| Brazil | Portuguese | audited Senate speeches export; Hemeroteca disabled |
| Mexico | Spanish | blocked: UANL conditional; HNDM bulk use disabled |

An “audited export” is a local `records.jsonl` plus source text/XML. This avoids
inventing undocumented scrapers where a supported bulk endpoint or reuse permission
has not been established. Conditional and disabled sources fail before networking.

Spain uses the official OAI-PMH repositories for Biblioteca Digital Hispánica and
Biblioteca Virtual de Prensa Histórica. Portugal uses the Biblioteca Nacional de
Portugal OAI-PMH public-domain digital set. Optional Internet Archive source entries
provide a practical path to plain OCR. The generic OAI collector follows resumption
tokens but stops at the requested limit; it only ingests dated records at or before
the cutoff that expose plain text and an explicit eligible rights statement.
Metadata-only and rights-unknown records are quarantined.

For a repository-supplied export, put a record like this in `records.jsonl`:

```json
{"id":"record-1","source":"dta","path":"texts/record-1.xml","title":"Titel","date":"1890","url":"https://example.invalid/persistent-record","rights":"Public domain"}
```

Then ingest the audited export. Paths cannot escape the input root.

```powershell
history-rag harvest dta --country germany --input-root C:\exports\dta --limit 50
history-rag build --country germany --mode native
history-rag query "Wie wurde über Brotpreise geschrieben?" --country germany --mode native
```

```powershell
# Native corpora default to databases/<country>/data/corpus.native.jsonl
history-rag harvest internet-archive-es --country spain --limit 10
history-rag harvest internet-archive-pt --country portugal --limit 10

history-rag build --country spain --mode native
history-rag query --country spain --mode native
history-rag build --country portugal --mode native
history-rag query "Como mudou o preço do pão?" --country portugal --mode native
```

The query may be omitted to use the manifest's native-language pilot question.
Corpus rows and chunks retain `country`, `language`, `content_mode`, item URL,
rights text, and source metadata.

### English translation mode

Translation is an explicit corpus transformation, never part of native build/query.
It uses the OpenAI Responses API with fixed model `gpt-5.6-luna`. Document, batch,
and invocation character ceilings make requests genuinely bounded. First inspect a
key-free plan:

```powershell
history-rag translate --country spain --batch-size 10 --dry-run
```

For a real run, set `OPENAI_API_KEY`, pass `--execute`, and omit `--dry-run`.
Corpus and state replacements are atomic. A durable in-flight marker prevents an
uncertain crash from silently rebilling; `--retry-ambiguous` is an explicit override.
Rerunning skips only an identical source/prompt/schema/model/settings identity, while
a changed source replaces its stale derived row. Provenance records response ID,
returned model, usage, hashes, settings, and timestamps.

```powershell
$env:OPENAI_API_KEY = "..."
history-rag translate --country spain --batch-size 10 --limit 100 --execute
history-rag build --country spain --mode english
history-rag query "How did food prices affect families?" --country spain --mode english
```

Translation is model-generated derivative text. Use native evidence for quotation and
verification; both modes retain the original item URL for citations.

UK native retrieval is already English. Its Luna-rendered mode is available only
when an explicitly derived corpus is useful; it is not a more authoritative edition.

## Harvest a small, balanced pilot

Each command is intentionally bounded. Increase `--limit` only after inspecting the resulting metadata and OCR.
Downloads also skip individual OCR objects larger than 20 MB unless `--max-document-bytes` is changed.
Remote JSON metadata is cached under `databases/america/data/cache/<source>` and fetched with four bounded workers by default.
Use `--workers`, `--cache-dir`, or `--no-metadata-cache` to change that behavior. Evans-TCP is parsed
from local XML and therefore does not use the remote metadata cache or worker setting.

When the loc.gov JSON API is unavailable or a reproducible batch sample is preferred,
Chronicling America can be harvested from an official NDNP batch without consuming the
20-request-per-minute JSON allowance:

```powershell
american-rag harvest loc-chronicling-america --limit 50 \
  --chronicling-batch-url https://chroniclingamerica.loc.gov/data/batches/dlc_bigstone_ver02/data/
```

```powershell
# Start a new corpus with newspaper pages about domestic life.
american-rag harvest loc-chronicling-america --query "rent wages household" --date-from 1880 --date-to 1929 --limit 20

# Append public-domain LOC books.
american-rag harvest loc-selected-digitized-books --limit 20 --append

# Append Founders Online primary documents (less representative of ordinary life).
american-rag harvest founders-online --limit 20 --append

# Fetch and append keyed early-American imprints.
american-rag fetch-evans
american-rag harvest evans-tcp --limit 20 --append
```

LOC books are filtered before their OCR is downloaded. Explicit fiction signals are rejected;
decisive nonfiction subjects or LC classes are accepted; ambiguous records are skipped. Use
`--loc-include-uncertain` only when you intentionally want the review queue included. Accepted
records retain subjects, call numbers, descriptions, and classification reasons.

`fetch-evans` downloads one `N00` prefix submodule by default. Use `--prefix N23` for a different
partition or `--all` only when you intentionally want the complete multi-submodule corpus.

Generate a manual Evans review list without removing anything:

```powershell
american-rag audit-evans
```

Evans candidates from that rule are now skipped during normal harvests. The source XML and audit
rows remain untouched. Use `--evans-include-review-candidates` to restore them.

Sample Chronicling America metadata across four periods before designing article-level rules:

```powershell
american-rag audit-chronicling --sample-per-period 15 --delay 1.0
```

The LOC metadata is issue/page-level, so the pipeline first requires newspaper genre/format metadata
and rejects explicitly literary publication metadata. During index building it then removes chunks
with strong serialized-fiction or poetry markers. Every removed chunk is written to
`databases/america/data/index/excluded_chunks.jsonl` for audit; the original page OCR is never deleted.

If Founders Online's large metadata endpoint responds with HTTP 202 and no body, download
`founders-online-metadata.json` from its Metadata directory when the service is available, then use
`american-rag harvest founders --metadata-file PATH --limit 20 --append`. The document API is still
used for the actual primary text.

The default LOC book collector uses the official 1,000-item sample manifest. For a larger run, change `LOC_BOOK_MANIFEST` in `sources.py` to the full manifest URL. The full data package contains tens of thousands of books and should be partitioned rather than downloaded casually.

For a bounded harvest from the full official manifest without changing source code:

```powershell
american-rag harvest loc-selected-digitized-books --limit 50 \
  --manifest-url https://data.labs.loc.gov/digitized-books/manifest.json \
  --loc-metadata-url https://data.labs.loc.gov/digitized-books/metadata.json
```

## Build and query

```powershell
american-rag build
american-rag query "How did working families manage the cost of food and rent?" --date-from 1880 --date-to 1929
```

To produce a source-grounded prompt suitable for a chat model:

```powershell
american-rag query "What did ordinary people say about household debt?" --prompt > evidence-prompt.txt
```

The generated prompt instructs the answer model to cite `[S1]`, `[S2]`, and so on, preserve disagreement, and avoid generalizing from a small sample. No answer-model dependency or API key is imposed by this project.

## Scaling

The starter index memory-maps a NumPy matrix and uses exact cosine search. This is simple and auditable for a pilot corpus. At roughly one million chunks, replace the dense-search implementation with Qdrant, FAISS, or another approximate-nearest-neighbor index while retaining `chunks.jsonl` and the FTS5 index.

Partition downloads by source, year, state, and query. Keep the original files outside the normalized corpus and record checksums for reproducibility. Do not treat several reprints of one newspaper article as independent historical observations.

## Known source and rights cautions

- Chronicling America OCR is frequently noisy and page-level reading order may mix columns.
- Selected Digitized Books OCR quality varies; the collection is public domain and LOC requests credit.
- Evans-TCP transcription text is public domain, but third-party page images may have different terms.
- Founders Online disproportionately represents political elites and contains modern editorial work. Its historical text, metadata, and editorial layers should not be assumed to share one license.
- Spanish OAI records can expose metadata without machine-readable full text; those records are intentionally skipped.
- The Portuguese BNP public-domain set is a strong discovery filter, but item-level statements and editions or added material still need review.
- Internet Archive availability and metadata do not establish copyright status.
- Publication cutoffs are collection heuristics, not public-domain determinations.
- Missing or ambiguous rights are quarantined. A local-export override is explicit,
  recorded provenance and remains the operator's responsibility.
- France, Germany, UK, China, Japan, South Korea, and Brazil currently use audited
  bulk exports unless a protocol-specific remote collector is documented as supported.
- Mexico is configured but not harvest-enabled: HNDM terms do not support this bulk
  workflow and UANL remains conditional pending endpoint/rights validation.
- “Pre-1930” describes publication date, not necessarily the period described by a later edition or recollection.

## Verification boundary

The checked-in tests use fixtures and mocked embedding/Responses clients. This
iteration has not performed live national-library harvests, a paid Luna translation,
the multilingual model download/full corpus rebuild, or expert evaluation of
retrieval and translation fidelity. Those are operational and scholarly acceptance
steps, not results implied by the passing unit suite.

## Tests

```powershell
pytest
```
