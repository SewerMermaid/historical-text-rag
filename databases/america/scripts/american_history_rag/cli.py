from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .audit import audit_chronicling, audit_evans
from .index import DEFAULT_EMBED_MODEL, build_index, context_prompt, search
from .sources import (LOC_BOOK_MANIFEST, LOC_BOOK_METADATA, collect,
                      collect_chronicling_batch)
from .util import write_jsonl


AMERICA_DATA = Path("databases/america/data")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="american-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    harvest = sub.add_parser("harvest", help="Download and normalize a bounded source sample")
    harvest.add_argument("source", choices=[
        "loc-chronicling-america", "loc-selected-digitized-books",
        "evans-tcp", "founders-online",
        # Backward-compatible command aliases.
        "chronicling", "loc-books", "evans", "founders",
    ])
    harvest.add_argument("--limit", type=int, default=10)
    harvest.add_argument("--output", type=Path, default=AMERICA_DATA / "corpus.jsonl")
    harvest.add_argument("--append", action="store_true")
    harvest.add_argument("--query", default="")
    harvest.add_argument("--date-from", type=int, default=1800)
    harvest.add_argument("--date-to", type=int, default=1929)
    harvest.add_argument("--delay", type=float, default=1.0,
                         help="Seconds between Chronicling America item requests")
    harvest.add_argument("--workers", type=int, default=4,
                         help="Bounded concurrent remote fetches (default: 4; Evans remains local and sequential)")
    harvest.add_argument("--cache-dir", type=Path, default=AMERICA_DATA / "cache",
                         help="Persistent cache root for remote JSON metadata")
    harvest.add_argument("--no-metadata-cache", action="store_true",
                         help="Disable reading and writing the remote JSON metadata cache")
    harvest.add_argument("--max-document-bytes", type=int, default=20_000_000,
                         help="Skip unexpectedly large OCR objects (default: 20 MB)")
    harvest.add_argument("--repo", type=Path, default=AMERICA_DATA / "evans-tcp")
    harvest.add_argument("--metadata-file", type=Path,
                         help="Cached founders-online-metadata.json (use if its server is busy)")
    harvest.add_argument("--manifest-url", default=LOC_BOOK_MANIFEST,
                         help="LOC Selected Digitized Books manifest (sample by default; use the full manifest for larger harvests)")
    harvest.add_argument("--loc-metadata-url", default=LOC_BOOK_METADATA,
                         help="Bulk LOC Selected Digitized Books metadata matching the selected manifest")
    harvest.add_argument("--chronicling-batch-url", default="",
                         help="Official NDNP batch data/ URL; bypasses loc.gov JSON discovery")
    harvest.add_argument("--loc-include-uncertain", action="store_true",
                         help="Include LOC books lacking decisive nonfiction metadata")
    harvest.add_argument("--evans-include-review-candidates", action="store_true",
                         help="Include Evans items flagged for literary/manual review")
    clone = sub.add_parser("fetch-evans", help="Clone the Evans-TCP bulk XML corpus")
    clone.add_argument("--repo", type=Path, default=AMERICA_DATA / "evans-tcp")
    clone.add_argument("--prefix", default="N00", help="One Evans ID prefix to fetch by default")
    clone.add_argument("--all", action="store_true", help="Fetch all 38 text submodules")
    evans_audit = sub.add_parser("audit-evans", help="List possible literary Evans texts; excludes nothing")
    evans_audit.add_argument("--repo", type=Path, default=AMERICA_DATA / "evans-tcp")
    evans_audit.add_argument("--output", type=Path,
                             default=AMERICA_DATA / "audits/evans_exclusion_candidates.csv")
    chronicling_audit = sub.add_parser("audit-chronicling", help="Sample metadata across four periods")
    chronicling_audit.add_argument("--sample-per-period", type=int, default=15)
    chronicling_audit.add_argument("--delay", type=float, default=1.0,
                                   help="Seconds between LOC item requests")
    chronicling_audit.add_argument("--output", type=Path,
                                   default=AMERICA_DATA / "audits/chronicling_metadata_report.json")
    build = sub.add_parser("build", help="Chunk and build dense plus FTS5 indexes")
    build.add_argument("--corpus", type=Path, default=AMERICA_DATA / "corpus.jsonl")
    build.add_argument("--index", type=Path, default=AMERICA_DATA / "index")
    build.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    build.add_argument("--chunk-words", type=int, default=350)
    build.add_argument("--overlap-words", type=int, default=50)
    build.add_argument("--batch-size", type=int, default=32)
    query = sub.add_parser("query", help="Run hybrid retrieval")
    query.add_argument("question")
    query.add_argument("--index", type=Path, default=AMERICA_DATA / "index")
    query.add_argument("--top-k", type=int, default=8)
    query.add_argument("--source", default="")
    query.add_argument("--date-from", type=int)
    query.add_argument("--date-to", type=int)
    query.add_argument("--max-per-document", type=int, default=2,
                       help="Diversity cap to prevent one book/page dominating results")
    query.add_argument("--prompt", action="store_true", help="Print a grounded prompt for any chat model")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fetch-evans":
        args.repo.parent.mkdir(parents=True, exist_ok=True)
        if not (args.repo / ".git").exists():
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://bitbucket.org/eads004/evanstcp.git", str(args.repo)], check=True)
        command = ["git", "-c", "url.https://bitbucket.org/.insteadOf=git@bitbucket.org:",
                   "submodule", "update", "--init", "--depth", "1"]
        if not args.all:
            command.append(f"texts/{args.prefix}")
        subprocess.run(command, cwd=args.repo, check=True)
        return
    if args.command == "audit-evans":
        total, candidates = audit_evans(args.repo, args.output)
        print(f"Reviewed metadata for {total} Evans files; wrote {candidates} non-binding candidates to {args.output}")
        return
    if args.command == "audit-chronicling":
        report = audit_chronicling(args.sample_per_period, args.output, delay=args.delay)
        print(f"Audited {report['records']} records across four periods; wrote {args.output}")
        return
    if args.command == "harvest":
        kwargs = {}
        source_cache = None if args.no_metadata_cache else args.cache_dir / args.source
        if args.source in {"loc-chronicling-america", "chronicling"}:
            kwargs.update(query=args.query, date_from=args.date_from, date_to=args.date_to,
                          max_bytes=args.max_document_bytes, delay=args.delay,
                          cache_dir=source_cache, workers=args.workers)
        elif args.source in {"loc-selected-digitized-books", "loc-books"}:
            kwargs["max_bytes"] = args.max_document_bytes
            kwargs["include_uncertain"] = args.loc_include_uncertain
            kwargs["manifest_url"] = args.manifest_url
            kwargs["metadata_url"] = args.loc_metadata_url
            kwargs["cache_dir"] = source_cache
            kwargs["workers"] = args.workers
        elif args.source in {"evans-tcp", "evans"}:
            kwargs["repo"] = args.repo
            kwargs["include_review_candidates"] = args.evans_include_review_candidates
        elif args.source in {"founders-online", "founders"}:
            if args.metadata_file:
                kwargs["metadata_url"] = str(args.metadata_file)
            kwargs["cache_dir"] = source_cache
            kwargs["workers"] = args.workers
        if args.source in {"loc-chronicling-america", "chronicling"} and args.chronicling_batch_url:
            documents = collect_chronicling_batch(
                args.limit, args.chronicling_batch_url,
                cache_dir=source_cache / "bulk" if source_cache else None,
                workers=args.workers,
            )
        else:
            documents = collect(args.source, args.limit, **kwargs)
        count = write_jsonl(args.output, (doc.asdict() for doc in documents), append=args.append)
        print(f"Wrote {count} {args.source} documents to {args.output}")
        return
    if args.command == "build":
        count = build_index(args.corpus, args.index, model_name=args.model,
                            target_words=args.chunk_words, overlap_words=args.overlap_words,
                            batch_size=args.batch_size)
        print(f"Indexed {count} chunks in {args.index}")
        return
    if args.command == "query":
        results = search(args.index, args.question, top_k=args.top_k, source=args.source,
                         date_from=args.date_from, date_to=args.date_to,
                         max_per_document=args.max_per_document)
        if args.prompt:
            print(context_prompt(args.question, results))
        else:
            for i, row in enumerate(results, 1):
                print(f"\n[{i}] {row['title']} ({row.get('date') or 'undated'})")
                print(f"{row['source']} | {row['url']} | score={row['score']:.5f}")
                print(row["text"][:700].replace("\n", " "))


if __name__ == "__main__":
    main()
