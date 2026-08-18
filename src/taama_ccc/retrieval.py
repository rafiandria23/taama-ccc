from __future__ import annotations

import sys

from openai import OpenAI
from pydantic import BaseModel, Field

from taama_ccc.config import Settings
from taama_ccc.models import DocumentChunk, Evidence
from taama_ccc.qdrant_store import QdrantStore


def embed_texts(
    client: OpenAI,
    settings: Settings,
    texts: list[str],
    *,
    batch_size: int = 100,
) -> list[list[float]]:
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
            dimensions=settings.openai_embedding_dimensions,
        )

        vectors.extend(item.embedding for item in response.data)

    return vectors


def embed_text(
    client: OpenAI,
    settings: Settings,
    text: str,
) -> list[float]:
    return embed_texts(client, settings, [text])[0]


class RerankItem(BaseModel):
    index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)


class RerankResponse(BaseModel):
    results: list[RerankItem]


class Retriever:
    def __init__(
        self,
        client: OpenAI,
        store: QdrantStore,
        settings: Settings,
    ) -> None:
        self._client = client
        self._store = store
        self._settings = settings

    def search(
        self,
        query: str,
        *,
        retrieval_limit: int = 20,
        top_k: int = 5,
    ) -> list[Evidence]:
        candidates = self._hybrid_search(query, limit=retrieval_limit)

        if not candidates:
            return []

        return self._rerank(query, candidates, top_k=top_k)

    def _hybrid_search(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[Evidence]:
        dense_vector = embed_text(self._client, self._settings, query)
        points = self._store.query(
            dense_vector=dense_vector,
            query_text=query,
            limit=limit,
            prefetch_limit=limit,
        )

        return [
            Evidence(
                chunk=DocumentChunk.model_validate(point.payload),
                relevance_score=point.score,
            )
            for point in points
        ]

    def _rerank(
        self, query: str, candidates: list[Evidence], *, top_k: int
    ) -> list[Evidence]:
        chunks = [c.chunk for c in candidates]
        documents = "\n\n".join(
            f"[{i}]\n{chunk.text}" for i, chunk in enumerate(chunks)
        )

        response = self._client.responses.parse(
            model=self._settings.openai_reranker_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a regulatory retrieval reranker. Score each "
                        "document according to how directly it helps answer the "
                        "user's query. Only use the information present in each "
                        "document."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query:\n{query}\n\nDocuments:\n{documents}\n\n"
                        f"Return the most relevant documents, up to {top_k}."
                    ),
                },
            ],
            text_format=RerankResponse,
        )

        raw_results = response.output_parsed.results

        valid_results = [item for item in raw_results if item.index < len(chunks)]
        dropped = len(raw_results) - len(valid_results)

        if dropped:
            print(
                f"[retrieval] dropped {dropped} out-of-range rerank index(es) "
                f"({len(chunks)} candidates supplied)",
                file=sys.stderr,
            )

        ranked = sorted(
            valid_results,
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

        return [
            Evidence(
                chunk=chunks[item.index],
                relevance_score=candidates[item.index].relevance_score,
                rerank_score=item.score,
            )
            for item in ranked
        ]
