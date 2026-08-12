# American History Hybrid RAG

A bounded, resumable corpus builder and citation-preserving hybrid retriever for American texts before 1930.

Country-specific state is separated under `databases/`: the populated American
database is in `databases/america/`, with reserved empty directories for Spain
and Portugal. The American API/harvest/index package is in
`databases/america/scripts/`, and its corpus, caches, raw downloads, and indexes
are in `databases/america/data/`.

It supports:

- Chronicling America through the supported `loc.gov` API
- Library of Congress Selected Digitized Books through its data-package manifest
- Evans-TCP from its bulk TEI/XML repository
- Founders Online through its metadata and document APIs
- Dense retrieval with `BAAI/bge-small-en-v1.5`
- SQLite FTS5 lexical retrieval, combined with dense rankings using reciprocal-rank fusion
- Date/source filtering and grounded prompts with primary-source URLs
- Result diversification so one long book or newspaper page cannot monopolize the evidence
- Conservative LOC nonfiction filtering with retained reasons and an uncertainty quarantine
- Non-binding Evans and Chronicling metadata audit commands

## Why hybrid retrieval?

Historical queries need semantic retrieval, while names, prices, occupations, archaic spelling, and imperfect OCR often benefit from exact lexical matching. The pipeline runs both and fuses their rankings.

## Install

Python 3.10-3.12 is recommended because ML-library support for very new Python releases can lag.

```powershell
cd C:\Users\cheng\american-history-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The first index build downloads the embedding model (about 130 MB). After that, document embeddings are computed once and persisted. Each search embeds only the query.

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

## Known source cautions

- Chronicling America OCR is frequently noisy and page-level reading order may mix columns.
- Selected Digitized Books OCR quality varies; the collection is public domain and LOC requests credit.
- Evans-TCP transcription text is public domain, but third-party page images may have different terms.
- Founders Online disproportionately represents political elites and contains modern editorial work. Its historical text, metadata, and editorial layers should not be assumed to share one license.
- “Pre-1930” describes publication date, not necessarily the period described by a later edition or recollection.

## Tests

```powershell
pytest
```
