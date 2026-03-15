"""
RLHF Feedback Store
────────────────────
Stores human ratings on rewritten prompts in Qdrant.
High-rated rewrites become few-shot examples for future enhancement requests.

This implements a lightweight in-context RLHF loop:
  • User rates a rewrite  → stored as a preference record
  • Similar future prompts  → retrieve high-rated examples
  • Those examples are injected as few-shot context  → LLM "learns" the preference

No model fine-tuning needed — the preference signal is delivered through
retrieval-augmented context rather than gradient updates.

Usage:
    store = FeedbackStore()
    store.initialize()
    store.save(record, embedding)

    examples = store.get_good_examples(query_vector, min_rating=4)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    VectorParams,
)

from prompt_guidance.config import settings


FEEDBACK_COLLECTION = "prompt_feedback"


@dataclass
class FeedbackRecord:
    original_prompt: str
    enhanced_prompt: str
    rating: int                    # 1 (bad) → 5 (excellent)
    comment: str = ""
    techniques_used: str = ""
    provider: str = ""
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class FeedbackStore:
    """Qdrant-backed store for human preference signals on prompt rewrites."""

    def __init__(
        self,
        dimensions: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        self.dimensions = dimensions or settings.embed_dimensions
        self._client = QdrantClient(
            host=host or settings.qdrant_host,
            port=port or settings.qdrant_port,
            api_key=api_key or settings.qdrant_api_key or None,
        )

    def initialize(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if FEEDBACK_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=FEEDBACK_COLLECTION,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )

    def save(self, record: FeedbackRecord, embedding: list[float]) -> None:
        """Persist a rated rewrite example."""
        self._client.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[
                PointStruct(
                    id=record.id,
                    vector=embedding,
                    payload={
                        "original_prompt": record.original_prompt,
                        "enhanced_prompt": record.enhanced_prompt,
                        "rating": record.rating,
                        "comment": record.comment,
                        "techniques_used": record.techniques_used,
                        "provider": record.provider,
                        "model": record.model,
                        "timestamp": record.timestamp,
                    },
                )
            ],
        )

    def get_good_examples(
        self,
        query_vector: list[float],
        min_rating: int = 4,
        top_k: int = 3,
    ) -> list[FeedbackRecord]:
        """
        Retrieve semantically similar prompts that received high ratings.
        These become few-shot RLHF examples injected into future rewrites.
        """
        results = self._client.query_points(
            collection_name=FEEDBACK_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=Filter(
                must=[FieldCondition(key="rating", range=Range(gte=min_rating))]
            ),
        )
        records: list[FeedbackRecord] = []
        for hit in results.points:
            p = hit.payload or {}
            records.append(
                FeedbackRecord(
                    id=str(hit.id),
                    original_prompt=p.get("original_prompt", ""),
                    enhanced_prompt=p.get("enhanced_prompt", ""),
                    rating=p.get("rating", 0),
                    comment=p.get("comment", ""),
                    techniques_used=p.get("techniques_used", ""),
                    provider=p.get("provider", ""),
                    model=p.get("model", ""),
                    timestamp=p.get("timestamp", ""),
                )
            )
        return records

    def get_stats(self) -> dict:
        try:
            info = self._client.get_collection(FEEDBACK_COLLECTION)
            total = info.points_count or 0
        except Exception:
            return {"total": 0, "avg_rating": None, "distribution": {}}

        if total == 0:
            return {"total": 0, "avg_rating": None, "distribution": {}}

        # Scroll all points to compute rating distribution
        ratings: list[int] = []
        offset = None
        while True:
            page, offset = self._client.scroll(
                collection_name=FEEDBACK_COLLECTION,
                limit=200,
                offset=offset,
                with_payload=["rating"],
                with_vectors=False,
            )
            for pt in page:
                if pt.payload and "rating" in pt.payload:
                    ratings.append(int(pt.payload["rating"]))
            if offset is None:
                break

        dist = {str(i): ratings.count(i) for i in range(1, 6)}
        avg = sum(ratings) / len(ratings) if ratings else None
        return {"total": total, "avg_rating": round(avg, 2) if avg else None, "distribution": dist}

    def format_as_few_shot(self, examples: list[FeedbackRecord]) -> str:
        """
        Format high-rated examples as few-shot context for the rewriter LLM.
        This is the bridge between the feedback store and the prompt construction layer.
        """
        if not examples:
            return ""
        parts = ["## HIGHLY-RATED REWRITE EXAMPLES (learn from these)\n"]
        for i, ex in enumerate(examples, 1):
            stars = "★" * ex.rating + "☆" * (5 - ex.rating)
            parts.append(
                f"### Example {i}  {stars}  (rating {ex.rating}/5)\n"
                f"**Original:** {ex.original_prompt}\n\n"
                f"**Rewritten:** {ex.enhanced_prompt}\n"
                + (f"**Why it worked:** {ex.comment}\n" if ex.comment else "")
                + (f"**Techniques:** {ex.techniques_used}\n" if ex.techniques_used else "")
            )
        return "\n".join(parts)
