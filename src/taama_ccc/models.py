from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class DocumentSource(StrEnum):
    DOCX = "docx"


class Document(BaseModel):
    id: str
    source: DocumentSource
    title: str
    content: str


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    text: str

    section: str | None = None
    source_url: HttpUrl | None = None

    metadata: dict[str, str] = Field(default_factory=dict)


class Evidence(BaseModel):
    chunk: DocumentChunk
    relevance_score: float = Field(ge=0.0)
    rerank_score: float | None = Field(default=None, ge=0.0)


class ComplianceStatus(StrEnum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    NEEDS_REVIEW = "needs_review"


class ComplianceResult(BaseModel):
    status: ComplianceStatus
    reasoning: str
    evidence: list[Evidence]
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedClaims(BaseModel):
    product_name: str | None = None
    claims: list[str] = Field(default_factory=list)
