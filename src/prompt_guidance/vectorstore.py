"""
Qdrant Vector Store
───────────────────
Wraps qdrant-client with a clean interface for:
  • Collection management (auto-create with HNSW cosine index)
  • Upsert documents with embeddings + metadata payload
  • Cosine similarity search
  • Delete by source file (for re-ingestion)
  • Listing ingested sources and chunk counts

Usage:
    store = QdrantStore()
    store.initialize()          # creates collection if needed
    store.upsert(chunks)
    results = store.search(vector, top_k=5)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from prompt_guidance.config import settings


# ─────────────────────────────────────────────────────────────────
# Data model shared across the whole app
# ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    score: Optional[float] = None


# ─────────────────────────────────────────────────────────────────
# Qdrant implementation
# ─────────────────────────────────────────────────────────────────

class QdrantStore:
    def __init__(
        self,
        collection: Optional[str] = None,
        dimensions: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        self.collection = collection or settings.qdrant_collection
        self.dimensions = dimensions or settings.embed_dimensions

        self._client = QdrantClient(
            host=host or settings.qdrant_host,
            port=port or settings.qdrant_port,
            api_key=api_key or settings.qdrant_api_key or None,
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the collection if it does not exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=Distance.COSINE,
                ),
            )

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection)

    # ── Write ─────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.embedding or [],
                payload={
                    "content": chunk.content,
                    "source": chunk.source,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks belonging to a given source file."""
        self._client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                )
            ),
        )

    # ── Read ──────────────────────────────────────────────────────

    def search(self, query_vector: list[float], top_k: Optional[int] = None) -> list[Chunk]:
        results = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k or settings.top_k,
            with_payload=True,
        )
        chunks = []
        for hit in results.points:
            payload = dict(hit.payload or {})
            content = payload.pop("content", "")
            source = payload.pop("source", "")
            chunks.append(
                Chunk(
                    id=str(hit.id),
                    content=content,
                    source=source,
                    metadata=payload,
                    score=hit.score,
                )
            )
        return chunks

    def list_sources(self) -> list[str]:
        """Return distinct source file paths stored in the collection."""
        sources: set[str] = set()
        offset = None
        while True:
            page, offset = self._client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for point in page:
                if point.payload and "source" in point.payload:
                    sources.add(point.payload["source"])
            if offset is None:
                break
        return sorted(sources)

    def count(self) -> int:
        info = self._client.get_collection(self.collection)
        return info.points_count or 0
