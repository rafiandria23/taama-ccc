from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from taama_ccc.config import get_settings
from taama_ccc.extraction import SUPPORTED_SUFFIXES, extract_claims
from taama_ccc.pipeline import check_claim
from taama_ccc.qdrant_store import QdrantStore, create_qdrant_client
from taama_ccc.retrieval import Retriever

_stderr_console = Console(stderr=True)
_console = Console()


STATUS_STYLES = {
    "red": "bold white on red",
    "amber": "bold black on yellow",
    "green": "bold white on green",
    "needs_review": "bold white on grey42",
}
STATUS_LABELS = {
    "red": "RED",
    "amber": "AMBER",
    "green": "GREEN",
    "needs_review": "NEEDS REVIEW",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help=(
            "Path(s) to images/PDFs of the same product, or a single directory "
            "containing them — directories are auto-discovered, no renaming needed"
        ),
    )
    group.add_argument(
        "--text",
        type=str,
        help="Raw claim text",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Output raw JSON Lines instead of the pretty terminal view — use "
            "this when piping to a file or another program."
        ),
    )


def _resolve_input_paths(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []

    for path in paths:
        if path.is_dir():
            found = sorted(
                p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
            )

            if not found:
                raise SystemExit(f"no supported image/PDF files found in {path}")

            resolved.extend(found)
        else:
            resolved.append(path)

    return resolved


def _progress(message: str) -> None:
    _stderr_console.print(f"[dim]{message}[/dim]")


def _evidence_summary(evidence) -> list[dict]:
    return [
        {
            "chunk_id": item.chunk.id,
            "document": item.chunk.document_id,
            "section": item.chunk.section,
            "category": item.chunk.metadata.get("category"),
            "requirement_area": item.chunk.metadata.get("requirement_area"),
            "excerpt": item.chunk.text,
            "comment": item.chunk.metadata.get("comment"),
            "source_links": item.chunk.metadata.get("source_links"),
            "possible_stale_source": item.chunk.metadata.get("possible_stale_source"),
        }
        for item in evidence
    ]


def _render_claim(record: dict) -> None:
    status = record.get("status", "unknown")
    style = STATUS_STYLES.get(status, "bold white")
    label = STATUS_LABELS.get(status, status.upper())
    border = style.split(" on ")[-1] if " on " in style else "white"

    header = Text(f" {label} ", style=style)
    header.append(f"  {record.get('product_name') or '(unknown product)'}", style="dim")

    body = Text()
    body.append("Claim: ", style="bold")
    body.append(f"{record.get('claim', '')}\n\n")
    body.append("Reasoning: ", style="bold")
    body.append(f"{record.get('reasoning', '')}\n")

    confidence = record.get("confidence")

    if confidence is not None:
        body.append(f"\nConfidence: {confidence:.2f}", style="dim")

    _console.print(
        Panel(
            body,
            title=header,
            border_style=border,
            expand=True,
        )
    )

    evidence = record.get("evidence") or []

    if evidence:
        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            padding=(0, 1),
        )

        table.add_column("Section", overflow="fold", ratio=2)
        table.add_column("Requirement", overflow="fold", ratio=2)
        table.add_column("Excerpt", overflow="fold", ratio=4)
        table.add_column("Stale?", justify="center", ratio=1)

        for item in evidence:
            stale = item.get("possible_stale_source") == "true"
            excerpt = (item.get("excerpt") or "").replace("\n", " ")

            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "\u2026"

            table.add_row(
                item.get("section") or "\u2014",
                item.get("requirement_area") or "\u2014",
                excerpt,
                Text("\u26a0 STALE", style="bold yellow") if stale else "\u2014",
            )

        _console.print(table)

    _console.print()


def _render_summary(record: dict) -> None:
    table = Table(
        title=f"Summary \u2014 {record.get('product_name') or '(unknown product)'}",
        show_header=True,
    )

    table.add_column("Status")
    table.add_column("Count", justify="right")

    counts = record.get("status_counts", {})

    for status, count in counts.items():
        label = STATUS_LABELS.get(status, status.upper())

        table.add_row(
            Text(label, style=STATUS_STYLES.get(status, "")),
            str(count),
        )

    table.add_row(
        "Total claims",
        str(record.get("claim_count", sum(counts.values()))),
    )

    _console.print()


def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

    input_paths = _resolve_input_paths(args.input) if args.input else None

    if input_paths:
        _progress(f"Extracting claims from {len(input_paths)} file(s)...")
    else:
        _progress("Extracting claims from text input...")

    extracted = extract_claims(
        client,
        settings,
        text=args.text,
        file_paths=input_paths,
    )

    _progress(
        f"Extracted {len(extracted.claims)} claim(s) for "
        f"'{extracted.product_name or 'unknown product'}'"
    )

    if not extracted.claims:
        summary = {
            "type": "summary",
            "product_name": extracted.product_name,
            "claim_count": 0,
            "status_counts": {},
            "note": "No distinct claims were extracted from this input.",
        }

        if args.json:
            print(json.dumps(summary, ensure_ascii=False))
        else:
            _console.print(
                "[yellow]No distinct claims were extracted from this input.[/yellow]"
            )

        return

    qdrant_client = create_qdrant_client(settings)
    store = QdrantStore(qdrant_client, settings)
    retriever = Retriever(client, store, settings)

    status_counts: dict[str, int] = {}

    for i, claim_text in enumerate(extracted.claims, start=1):
        _progress(f"[{i}/{len(extracted.claims)}] Checking: {claim_text[:70]}")

        result = check_claim(client, settings, retriever, claim_text)

        if args.json:
            _progress(f"[{i}/{len(extracted.claims)}] -> {result.status.value.upper()}")

        status_counts[result.status.value] = (
            status_counts.get(result.status.value, 0) + 1
        )

        claim_result = {
            "type": "claim_result",
            "product_name": extracted.product_name,
            "claim": claim_text,
            "status": result.status.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "evidence": _evidence_summary(result.evidence),
        }

        if args.json:
            print(json.dumps(claim_result, ensure_ascii=False))
            sys.stdout.flush()
        else:
            _render_claim(claim_result)

    summary_record = {
        "type": "summary",
        "product_name": extracted.product_name,
        "claim_count": len(extracted.claims),
        "status_counts": status_counts,
    }

    if args.json:
        print(json.dumps(summary_record, ensure_ascii=False))
    else:
        _render_summary(summary_record)
