from __future__ import annotations

from openai import OpenAI

from taama_ccc.config import Settings
from taama_ccc.evaluator import (
    EvidenceClassificationBatch,
    build_result,
    decide_verdict,
)
from taama_ccc.models import ComplianceResult
from taama_ccc.retrieval import Retriever

_CLASSIFY_SYSTEM_PROMPT = (
    "You classify whether each piece of regulatory evidence applies to a "
    "product claim, and how strongly.\n"
    "\n"
    "applies=true ONLY if the evidence text directly addresses this specific "
    "claim — not just the same general topic. When in doubt, applies=false; "
    "a false negative here just means less evidence considered, a false "
    "positive can misdirect the verdict.\n"
    "\n"
    "trigger_strength must reflect ONLY what the evidence text itself says "
    "about blocking vs warning ('Blocks' / 'Warns' in the source, or the "
    "equivalent prohibition language). Do not infer a stricter or looser "
    "reading than the text supports.\n"
    "\n"
    "Every evidence_id in your output must be exactly one of the supplied "
    "Evidence IDs. Do not invent IDs or evidence."
)


def _justify_system_prompt(status: str, reason: str) -> str:
    return (
        "Write a short, plain-English justification for a compliance verdict "
        f"that has ALREADY been decided as: {status.upper()}.\n"
        "\n"
        "Do not change, hedge on, or second-guess the verdict — your job is "
        "to explain how the cited evidence supports it, in language a "
        "non-lawyer can read and trust. Reference the specific evidence IDs.\n"
        "\n"
        f"The engine's own reason for this verdict: {reason}"
    )


def check_claim(
    client: OpenAI,
    settings: Settings,
    retriever: Retriever,
    claim: str,
) -> ComplianceResult:
    evidence = retriever.search(claim, retrieval_limit=20, top_k=5)

    if not evidence:
        classifications = []
    else:
        evidence_block = "\n\n".join(
            f"[Evidence ID: {item.chunk.id}]\n{item.chunk.text}" for item in evidence
        )

        response = client.responses.parse(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": _CLASSIFY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Claim:\n{claim}\n\nEvidence:\n{evidence_block}",
                },
            ],
            text_format=EvidenceClassificationBatch,
        )

        classifications = response.output_parsed.classifications

    verdict = decide_verdict(evidence, classifications)

    if verdict.cited_evidence:
        cited_block = "\n\n".join(
            f"[{e.chunk.id}]\n{e.chunk.text}" for e in verdict.cited_evidence
        )

        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": _justify_system_prompt(
                        verdict.status.value,
                        verdict.engine_reason,
                    ),
                },
                {
                    "role": "user",
                    "content": f"Claim:\n{claim}\n\nCited evidence:\n{cited_block}",
                },
            ],
        )

        justification = response.output_text
    else:
        justification = (
            "No regulatory evidence in the corpus was found to directly "
            "address this claim."
        )

    return build_result(verdict, justification)
