from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from taama_ccc.config import Settings
from taama_ccc.models import DocumentChunk

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def create_qdrant_client(settings: Settings) -> QdrantClient:
    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    )

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
    )


def _ensure_collection(client: QdrantClient, settings: Settings) -> None:
    if client.collection_exists(settings.qdrant_collection):
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=settings.openai_embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(),
        },
    )


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
    ) -> None:
        self._client = client
        self._settings = settings
        self._collection = settings.qdrant_collection

    def recreate_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)

        _ensure_collection(self._client, self._settings)

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        dense_vectors: Sequence[list[float]],
    ) -> None:
        if len(chunks) != len(dense_vectors):
            raise ValueError("chunks and dense_vectors must have the same length")

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                vector={
                    DENSE_VECTOR_NAME: vector,
                    SPARSE_VECTOR_NAME: models.Document(
                        text=chunk.text,
                        model="qdrant/bm25",
                    ),
                },
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, dense_vectors, strict=True)
        ]

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    def query(
        self,
        *,
        dense_vector: list[float],
        query_text: str,
        limit: int,
        prefetch_limit: int,
    ) -> list[models.ScoredPoint]:
        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query_text,
                        model="qdrant/bm25",
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return response.points
