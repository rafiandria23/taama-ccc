# Taama Claim Compliance Checker

Checks product marketing/label claims against Singapore's HSA/SFA regulatory
framework and returns a Red / Amber / Green / Needs-Review verdict per claim,
with every verdict traceable to the specific regulatory text it was decided
from.

## How to run it

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/), Docker (for
Qdrant), an OpenAI API key.

1. **Qdrant** (local, unauthenticated):
   ```bash
   docker compose up -d
   ```

2. **Environment:**
   ```bash
   cp .env.example .env
   # fill in OPENAI_API_KEY. Leave QDRANT_API_KEY blank for the compose
   # setup above — auth is only needed against a hosted/secured instance.
   ```

3. **Corpus:** this repo does not include `Regulatory_source_bank.docx` —
   it's Taama's own proprietary consultant document, not something to put in
   a public repo (see `.gitignore` — this also covers Qdrant's on-disk
   storage under `docker/volumes/`, since every chunk's full text is stored
   verbatim in Qdrant's payload; committing that directory would leak the
   same content through a different door). You already have the file you
   gave us; place it at:
   ```
   data/raw/Regulatory_source_bank.docx
   ```

4. **Install:**
   ```bash
   uv sync
   ```

5. **Build the index:**
   ```bash
   uv run taama-ccc rebuild-index --corpus-path data/raw/Regulatory_source_bank.docx
   ```
   Accepts a folder too (`--corpus-path data/raw`), searched recursively —
   every `.docx` found anywhere under it gets parsed and indexed together,
   in case a future market's corpus ends up split across multiple files in
   subfolders.

6. **Check a product:**
   ```bash
   # pretty terminal output, default — a directory of images/PDFs for one
   # product, searched recursively, no renaming needed
   uv run taama-ccc check --input data/fixtures/bp_pro

   # raw JSON Lines, for piping to a file or another program
   uv run taama-ccc check --input data/fixtures/bp_pro --json > data/fixtures/results/bp_pro.jsonl

   # or raw text
   uv run taama-ccc check --text "Boosts immunity and helps prevent colds"
   ```

   **Expect this to take a while** — one extraction call, then a
   classify + justify call per distinct claim found, all against
   `gpt-5-mini`/`gpt-5-nano`, which are reasoning-tuned models with real
   per-call latency. A multi-claim product can take a minute or more end to
   end. Every step (extraction, and each claim's check) reports its own
   elapsed time once it finishes, and shows a live spinner while it's
   running if you're in an actual terminal — specifically so a slow run
   never looks like a hang. `rebuild-index` reports the same per-step
   timing, plus a running total.

## Architecture

```
src/taama_ccc/
├── config.py                Settings (pydantic-settings)
├── models.py                Shared domain models (Document, DocumentChunk,
│                            Evidence, ComplianceResult, ExtractedClaims)
├── corpus.py                Row-level docx table parsing + prose chunking
├── qdrant_store.py          Qdrant client, collection setup, upsert/query
├── retrieval.py             Embeddings + hybrid dense/sparse search + rerank
├── evaluator.py             Deterministic verdict logic — no LLM call inside
├── extraction.py            Multimodal claim extraction (text/image/PDF)
├── pipeline.py              check_claim(): retrieve → classify → decide → justify
├── main.py                  CLI dispatcher (argparse subcommands)
├── rebuild_index.py         `taama-ccc rebuild-index`
└── check.py                 `taama-ccc check`
```

One installed command, two subcommands, no separate scripts directory. No
LangGraph, no LangChain — the actual pipeline shape is a straight line
(`retrieve → classify → decide → justify`) with one conditional inside
`evaluator.py`'s decision function, not a graph; a framework built for
cycles and branching state had nothing to offer a linear pipeline. Neither
`langchain`, `langchain-openai`, nor `langchain-qdrant` were ever imported
anywhere in this codebase's history — cut rather than kept as unused
weight.

### The design decision that matters most: LLM classifies, code decides

`evaluator.py::decide_verdict()` is pure Python — no I/O, no LLM call,
fully unit-testable. The LLM's job (`pipeline.py::check_claim`) is scoped to
one question per retrieved rule: does this apply to the claim, and how
strongly (blocks / warns / not applicable)? Everything downstream of that —
whether a stale-flagged source caps the verdict at NEEDS_REVIEW regardless
of how well it matched, whether missing evidence means "I don't know"
rather than a guess, the final RED/AMBER/GREEN/NEEDS_REVIEW assignment — is
a plain `if`/`elif` chain in one reviewable function. This is the direct
answer to "run it twice, same verdict?" (see WRITEUP.md) and the reason a
matched-but-stale source (e.g. the corpus's own "Annex A... Removed in v20"
row) gets flagged instead of confidently cited.

### Confidence is disclosed, not just documented

Every verdict carries a `confidence` float *and* a `confidence_basis`
string explaining why that number, inline in the actual output — not just
in this README. The five fixed values in `decide_verdict()` (0.85 for a
Block match down to 0.3 for no corpus coverage at all) aren't calibrated
against any labeled set; there isn't one large enough to fit anything
meaningful against. Rather than let a bare float imply more rigor than it
has, every `ComplianceResult` says so itself, in both the pretty terminal
view and `--json` output. A reviewer checking a specific verdict sees the
caveat right there, not buried in documentation they'd have to go find.

### Row-level corpus parsing

The regulatory docx is a consultant review document, not a clean rule
list — citations get corrected or flagged stale in an adjacent Comment
cell. `corpus.py` parses tables row-by-row (not paragraph-by-paragraph),
preserving which cell is the source, which is the approve/disapprove flag,
and which is the free-text correction that can override both — a
paragraph-based chunker would lose that relationship entirely, since
`python-docx`'s `doc.tables` and `doc.paragraphs` are separate flat lists
with no positional relationship to each other.

## What was built

- Row-level regulatory corpus parsing with automatic staleness detection
  (`possible_stale_source` in chunk metadata, keyword-matched against each
  row's own Comment text).
- Hybrid dense + sparse retrieval (Qdrant RRF fusion of
  `text-embedding-3-small` + BM25) with LLM reranking, constrained by a
  structured-output schema so the reranker can only reorder supplied
  candidates, never invent new ones (with an index-bounds check catching
  the case where it hallucinates one outside range anyway).
- Deterministic verdict logic with disclosed, non-calibrated confidence —
  both detailed above.
- Multimodal claim extraction — text, image, or PDF, with multiple files
  for one product sent as a single call (not one call per image), since a
  claim on a front label and a contradicting disclaimer on the back label
  need to be read together to catch the contradiction.
- A single installed CLI (`taama-ccc`) with two subcommands, pretty
  terminal output by default (color-coded verdict panels, evidence tables,
  elapsed-time tracking on every step, a live spinner when running in an
  actual terminal that degrades to plain persistent text when
  redirected/piped rather than vanishing), `--json` for raw
  machine-readable output. Both subcommands accept a folder as input,
  searched recursively.

## What was deliberately skipped

- **Chinese Proprietary Medicine (CPM) and Traditional Medicine (TM)
  categories.** None of the three sample products need them, and the
  corpus's own consultant notes say these can be modeled as a separate
  effort or referred to a specialist.
- **Webpage URL as an input type.** Not needed for any sample product.
- **Live auto-fetching of the docx's linked HSA/SFA/ASEAN pages.** The
  corpus already reflects the consultant's reviewed reading of those
  sources. Auto-fetching by default would make ingestion
  network-dependent (breaks "run it twice, same result?"), and several
  links point straight at PDFs an HTML scraper would mangle. Not built at
  all in this version — citation URLs are still extracted and kept in
  chunk metadata (`source_links`), just never fetched.
- **Detailed SFA (food-side) claim-matching rules.** The corpus itself has
  thin coverage here — mostly "general food composition rules apply,
  confirm against SFA," with no detailed matching logic the way the HSA
  side has. Food-side claims without a specific corpus match correctly
  resolve to NEEDS_REVIEW rather than guessing rules the source material
  doesn't actually support.
- **A dedicated UI.** Terminal output is the interface — the brief is
  explicit that this is a completely acceptable way to show results, and
  UI polish doesn't move any of the five weighted eval criteria.

## Known failure modes

- **LLM classification isn't perfectly deterministic.** `gpt-5-mini` and
  `gpt-5-nano` don't expose a `temperature` parameter, so identical inputs
  aren't *guaranteed* to produce identical classifications run-to-run,
  though the constrained output schema (evidence IDs must be from the
  supplied set) bounds how much they can vary. The verdict *decision* is
  fully deterministic regardless — see "the design decision that matters
  most" above.
- **Vision extraction quality depends on image legibility** and on the
  model correctly attributing a claim to the right source when several
  images are submitted together for one product.
- **No caching.** Every run re-embeds, re-retrieves, and re-calls the LLM
  from scratch, even for an identical repeated input.
- **Approve/Disapprove checkbox parsing is a regex heuristic** against
  inconsistently authored source text (`x Yes`, `xYes`, `X Approve`) — it
  can miss on formatting it hasn't seen.

Confidence calibration is deliberately *not* listed here — it isn't a bug
or a gap that slipped through, it's a disclosed design constraint: see
"Confidence is disclosed, not just documented" above.

## License

MIT License ([LICENSE](LICENSE) or <http://opensource.org/licenses/MIT>)
