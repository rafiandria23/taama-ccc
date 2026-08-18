from __future__ import annotations

import argparse
import time
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from taama_ccc.config import get_settings
from taama_ccc.corpus import parse_corpus
from taama_ccc.models import DocumentChunk
from taama_ccc.qdrant_store import QdrantStore, create_qdrant_client
from taama_ccc.retrieval import embed_texts

_console = Console()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corpus-path",
        type=Path,
        required=True,
        help="Path to a corpus .docx file, or a directory containing one or more .docx files",
    )


def _resolve_corpus_paths(path: Path) -> list[Path]:
    if path.is_dir():
        found = sorted(path.rglob("*.docx"))

        if not found:
            raise SystemExit(f"no .docx files found in {path}")

        return found

    if not path.exists():
        raise SystemExit(f"corpus not found at {path}")

    if path.suffix.lower() != ".docx":
        raise SystemExit(f"expected a .docx file, got {path}")

    return [path]


def _print_summary(chunks: list[DocumentChunk]) -> None:
    table_rows = sum(1 for c in chunks if c.metadata.get("chunk_type") == "table_row")
    stale = sum(1 for c in chunks if c.metadata.get("possible_stale_source") == "true")
    examples = sum(
        1
        for c in chunks
        if c.metadata.get("document_section") == "illustrative_example"
    )

    table = Table(title="Corpus parse summary")

    table.add_column("Metric")
    table.add_column("Count", justify="right")

    table.add_row("Total chunks", str(len(chunks)))
    table.add_row("Table-row chunks", str(table_rows))
    table.add_row("Prose chunks", str(len(chunks) - table_rows))
    table.add_row("Illustrative-example rows (Part 4)", str(examples))
    table.add_row("Flagged possibly-stale sources", str(stale))

    _console.print(table)


def run(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()

    corpus_paths = _resolve_corpus_paths(args.corpus_path)
    settings = get_settings()

    _console.print(
        f"[bold]Corpus source(s):[/bold] {', '.join(p.name for p in corpus_paths)}"
    )

    all_chunks: list[DocumentChunk] = []

    for path in corpus_paths:
        start = time.perf_counter()
        with _console.status(f"Parsing {path.name}..."):
            chunks = parse_corpus(path)
        elapsed = time.perf_counter() - start

        all_chunks.extend(chunks)

        _console.print(
            f"  [green]\u2713[/green] {path.name} \u2014 {len(chunks)} chunks "
            f"[dim]({elapsed:.1f}s)[/dim]"
        )

    _print_summary(all_chunks)

    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

    qdrant_client = create_qdrant_client(settings)
    store = QdrantStore(qdrant_client, settings)

    start = time.perf_counter()
    with _console.status("Recreating Qdrant collection..."):
        store.recreate_collection()
    elapsed = time.perf_counter() - start

    _console.print(
        f"[green]\u2713[/green] Collection '{settings.qdrant_collection}' ready "
        f"[dim]({elapsed:.1f}s)[/dim]"
    )

    start = time.perf_counter()
    with _console.status(f"Embedding {len(all_chunks)} chunks...", spinner="dots"):
        vectors = embed_texts(client, settings, [chunk.text for chunk in all_chunks])
    elapsed = time.perf_counter() - start

    _console.print(
        f"[green]\u2713[/green] Embedded {len(vectors)} chunks [dim]({elapsed:.1f}s)[/dim]"
    )

    start = time.perf_counter()
    with _console.status("Indexing into Qdrant..."):
        store.upsert(all_chunks, vectors)
    elapsed = time.perf_counter() - start

    run_elapsed = time.perf_counter() - run_start

    _console.print(
        f"[bold green]Indexed {len(all_chunks)} chunks into "
        f"'{settings.qdrant_collection}'[/bold green] "
        f"[dim](indexing: {elapsed:.1f}s, total: {run_elapsed:.1f}s)[/dim]"
    )
