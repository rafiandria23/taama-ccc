from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from taama_ccc.models import ComplianceResult, ComplianceStatus, Evidence


class TriggerStrength(StrEnum):
    BLOCKS = "blocks"
    WARNS = "warns"
    NOT_APPLICABLE = "not_applicable"


class EvidenceClassification(BaseModel):
    evidence_id: str
    applies: bool
    trigger_strength: TriggerStrength
    rationale: str = Field(
        description="One sentence: why this evidence does or doesn't apply to the claim."
    )


class EvidenceClassificationBatch(BaseModel):
    classifications: list[EvidenceClassification]


class Verdict:
    status: ComplianceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    cited_evidence: list[Evidence]
    engine_reason: str


def decide_verdict(
    evidence: list[Evidence],
    classifications: list[EvidenceClassification],
) -> Verdict:
    by_id = {e.chunk.id: e for e in evidence}

    applicable = [c for c in classifications if c.applies and c.evidence_id in by_id]

    if not applicable:
        return Verdict(
            status=ComplianceStatus.NEEDS_REVIEW,
            confidence=0.3,
            cited_evidence=[],
            engine_reason=(
                "No retrieved rule was classified as applicable to this claim — "
                "this is a corpus-coverage gap, not a compliance judgment. "
                "Saying so is preferable to guessing."
            ),
        )

    cited = [by_id[c.evidence_id] for c in applicable]
    stale = [
        c
        for c in applicable
        if by_id[c.evidence_id].chunk.metadata.get("possible_stale_source") == "true"
    ]

    if stale:
        marker = by_id[stale[0].evidence_id].chunk.metadata.get("stale_marker", "")

        return Verdict(
            status=ComplianceStatus.NEEDS_REVIEW,
            confidence=0.4,
            cited_evidence=cited,
            engine_reason=(
                f"The applicable rule's own source is flagged as possibly stale or "
                f"superseded (matched marker: '{marker}'). Citing it as current "
                f"would be exactly the confident-wrong-answer failure mode this "
                f"tool exists to avoid — flagging for human review instead."
            ),
        )

    if any(c.trigger_strength == TriggerStrength.BLOCKS for c in applicable):
        return Verdict(
            status=ComplianceStatus.RED,
            confidence=0.85,
            cited_evidence=cited,
            engine_reason="At least one applicable rule is a hard Block trigger.",
        )

    if any(c.trigger_strength == TriggerStrength.WARNS for c in applicable):
        return Verdict(
            status=ComplianceStatus.AMBER,
            confidence=0.6,
            cited_evidence=cited,
            engine_reason=(
                "At least one applicable rule is a Warn trigger — conditionally "
                "compliant, not a clean pass."
            ),
        )

    return Verdict(
        status=ComplianceStatus.GREEN,
        confidence=0.75,
        cited_evidence=cited,
        engine_reason="Applicable rules found; none block or warn.",
    )


def build_result(verdict: Verdict, justification: str) -> ComplianceResult:
    return ComplianceResult(
        status=verdict.status,
        reasoning=f"{justification}\n\n[Engine: {verdict.engine_reason}]",
        evidence=verdict.cited_evidence,
        confidence=verdict.confidence,
    )
