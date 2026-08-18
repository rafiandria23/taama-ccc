# Write-up

## Run it twice on the same input — identical verdicts? How does it decide Red/Amber/Green?

**The decision itself, yes, always. The path to it, not quite guaranteed —
and it's worth being precise about which part carries which guarantee.**

`evaluator.py::decide_verdict()` is the actual decision boundary: pure
Python, no LLM call, no randomness. Given the same evidence and the same
classifications, it always returns the same status *and* the same
confidence, via a fixed rule order:

1. No retrieved rule classified as applicable to the claim → **Needs
   Review**, confidence 0.3 (a corpus-coverage gap, not a judgment call).
2. An applicable rule's own source is flagged possibly stale/superseded
   (keyword-matched against that row's own Comment text — e.g. "Removed in
   v20") → **Needs Review**, confidence 0.4, regardless of how well the
   semantic match looked, because citing a source the corpus itself has
   flagged as questionable is the exact confident-wrong-answer failure
   mode this tool exists to avoid.
3. Any applicable rule is a hard Block trigger → **Red**, confidence 0.85.
4. Any applicable rule is a Warn trigger (and none Block) → **Amber**,
   confidence 0.6.
5. Applicable rules exist, none block or warn → **Green**, confidence 0.75.

Every verdict also carries a `confidence_basis` string stating plainly that
these five numbers are fixed constants, not calibrated against any labeled
set — three fixture products isn't a training set, and I'd rather the tool
say so on every result than let a bare float imply statistical rigor it
doesn't have. That disclosure is itself deterministic, for what it's worth:
the same branch always produces the same basis text, same as the status.

What isn't fully guaranteed is the step feeding that function: an LLM call
classifies whether each retrieved chunk applies to the claim and how
strongly. `gpt-5-mini`/`gpt-5-nano` don't expose a `temperature` parameter
at all, so I can't force that call to be deterministic the way I originally
planned to. The structured-output schema constrains it to only classify
against evidence IDs actually supplied — it can't invent new evidence —
which bounds how much a rerun could vary, but I won't claim byte-identical
output across runs. The honest framing: the decision boundary is
architecturally deterministic and isolated from the LLM entirely; the
classification step feeding it is constrained but not provably
deterministic. Separating those two was a deliberate design choice, not an
accident of how the code happened to end up — earlier in development the
LLM decided verdicts directly, and the "run it twice" question was
genuinely hard to answer honestly under that design. It isn't now.

## How would you extend this to a new market (e.g. Australia) or a new input type?

**New market:** every piece of Singapore-specific content lives in the
regulatory corpus itself (`data/raw/Regulatory_source_bank.docx`) — the
category definitions, trigger catalog, claims mechanics. None of it is
hardcoded in `corpus.py`, which parses generically: tables with a header
row and a Comment-style correction column. Swapping in a
consultant-reviewed Australian corpus (TGA framework instead of HSA/SFA)
and re-running `taama-ccc rebuild-index` would work as-is *if* the new
document follows a similar structure. The row parser matches on a fixed
set of likely header names ("Requirement area", "Criterion", "Trigger",
etc.); a structurally different source document would need that list
extended, but the core architecture — retrieve → classify → deterministic
decide → justify — doesn't know anything about Singapore specifically,
only about "rules with block/warn strength and a source." `rebuild-index`
already accepts a folder of `.docx` files, searched recursively, as one
corpus, in case a market's regulatory basis ends up split across several
documents in subfolders rather than one flat file.

**New input type:** `extraction.py` is deliberately the only place that
knows about images/PDFs/text — the rest of the pipeline only ever sees an
extracted claim string. Adding video or audio would mean writing a new
extraction function with the same `ExtractedClaims` output shape; nothing
in retrieval, classification, or decision logic would need to change.

## How would you deploy this to production, and how would you know it broke before a user told you?

**Deployment:** not as a synchronous request/response API. We measured
this directly during development — a multi-claim product check can take
over a minute, sequential LLM calls per claim (classify + justify), plus
the extraction call up front. That's a bad shape for a blocking HTTP
request. I'd deploy it as a submit-a-job / poll-or-webhook-for-result
pattern — a queue plus workers running the same `check_claim()` function —
not a call-and-wait endpoint. Qdrant already runs as its own service
(`docker-compose.yml`); the same containerization approach extends
naturally to the application layer. The CLI already reports per-step
elapsed time (extraction, and each claim's classify+justify round trip,
plus `rebuild-index`'s per-stage timing) — that instrumentation is a
starting point for the latency metrics a production deployment would need
to export, not something to build from scratch later.

**How I'd know it broke**, in order of how specific the signal is to this
design rather than generic:

- **Needs-Review rate over time.** A sudden spike is real signal — either
  retrieval degraded (embedding model change, index corruption) or the
  corpus needs re-review. This metric falls out of the design almost for
  free, because "I don't know" is a first-class output state here, not
  something buried in a low-confidence score somewhere.
- **`possible_stale_source` hit rate specifically.** If this creeps up on
  an unchanged corpus, nothing about the code changed — the source
  material is aging and needs a consultant pass, a different kind of
  alert than a code regression.
- **Structured-output parse failure rate** on the classify/rerank calls —
  an increase usually means an upstream model version changed behavior in
  a way that breaks the current prompts, visible before it breaks a
  verdict.
- **End-to-end latency**, given how much of it is sequential LLM calls —
  the per-step timing already surfaced in the CLI is the metric to export
  here, not a new one to invent.

## If you had another week, what would you build next?

In priority order:

1. **Build out real SFA/food-side claim rules**, properly sourced. Right
   now food claims without a strong corpus match correctly say "I don't
   know" rather than guess — the right behavior given what the corpus
   supports, but a real product needs the food side actually covered, not
   just honestly acknowledged as uncovered.
2. **Turn the three fixture products into an automated regression suite**
   instead of manual runs — cheap to add, and the value compounds every
   time the chunker, prompts, or decision thresholds change.
3. **Move from disclosed to actually calibrated confidence.** The current
   `confidence_basis` string is an honesty fix — every verdict says
   plainly that its number isn't statistically calibrated — not an
   accuracy fix. The next step is deriving confidence from actual
   retrieval/rerank scores and validating it against a labeled set large
   enough to mean something, so the field becomes something a downstream
   consumer could actually threshold on.
4. **Add caching** for repeated identical claim checks — meaningful cost
   and latency win at any real volume; this codebase currently has none.
5. **Make the Approve/Disapprove checkbox parsing more robust** — it's
   currently a regex heuristic against inconsistently authored source
   text and can miss on formatting variants it hasn't seen.
6. **CPM/TM category support**, if there's ever real demand for it — cut
   this round because none of the sample products needed it, not because
   it's out of scope forever.
