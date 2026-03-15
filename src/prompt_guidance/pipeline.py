"""
Pipeline
────────
Two pipelines:

1. IngestPipeline  — file → chunks → embeddings → Qdrant
2. RAGPipeline     — query → embed → search Qdrant → formatted context
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from prompt_guidance.config import settings
from prompt_guidance.embeddings import BaseEmbeddings, get_embeddings
from prompt_guidance.ingestors import Document, load as ingestor_load
from prompt_guidance.vectorstore import Chunk, QdrantStore

console = Console()


# ─────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for i, text in enumerate(self._split(doc.content)):
                chunks.append(
                    Chunk(
                        content=text,
                        source=doc.source,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        results: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                for sep in ("\n\n", "\n", ". ", " "):
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            piece = text[start:end].strip()
            if piece:
                results.append(piece)
            start = end - self.chunk_overlap
        return results


# ─────────────────────────────────────────────────────────────────
# Ingest Pipeline
# ─────────────────────────────────────────────────────────────────

class IngestPipeline:
    """Load → chunk → embed → store in Qdrant."""

    def __init__(
        self,
        embeddings: Optional[BaseEmbeddings] = None,
        store: Optional[QdrantStore] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.embeddings = embeddings or get_embeddings()
        self.store = store or QdrantStore(dimensions=self.embeddings.dimensions)
        self.chunker = TextChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

    def run(self, file_path: str | Path) -> dict:
        path = Path(file_path).resolve()
        source_key = str(path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:

            t1 = progress.add_task("Loading file…", total=None)
            docs = ingestor_load(path)
            progress.update(t1, description=f"Loaded {len(docs)} document(s)", completed=1, total=1)

            t2 = progress.add_task("Chunking…", total=None)
            chunks = self.chunker.chunk(docs)
            progress.update(t2, description=f"Created {len(chunks)} chunk(s)", completed=1, total=1)

            t3 = progress.add_task("Clearing previous data for this source…", total=None)
            self.store.delete_by_source(source_key)
            progress.update(t3, description="Previous data cleared", completed=1, total=1)

            t4 = progress.add_task("Embedding…", total=len(chunks))
            embedded: list[Chunk] = []
            for chunk in chunks:
                chunk.embedding = self.embeddings.embed(chunk.content)
                embedded.append(chunk)
                progress.advance(t4)

            t5 = progress.add_task("Storing in Qdrant…", total=None)
            stored = self.store.upsert(embedded)
            progress.update(t5, description=f"Stored {stored} vector(s)", completed=1, total=1)

        return {
            "file": str(path),
            "documents": len(docs),
            "chunks": len(chunks),
            "stored": stored,
        }


# ─────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """Embed query → search Qdrant → return chunks + formatted context string."""

    def __init__(
        self,
        embeddings: Optional[BaseEmbeddings] = None,
        store: Optional[QdrantStore] = None,
    ):
        self.embeddings = embeddings or get_embeddings()
        self.store = store or QdrantStore(dimensions=self.embeddings.dimensions)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Chunk]:
        vector = self.embeddings.embed(query)
        return self.store.search(vector, top_k=top_k or settings.top_k)

    def format_context(self, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant knowledge found in the knowledge base."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            score_str = f" [relevance: {chunk.score:.2f}]" if chunk.score is not None else ""
            basename = Path(chunk.source).name
            meta_extras: list[str] = []
            if "sheet" in chunk.metadata:
                meta_extras.append(f"sheet={chunk.metadata['sheet']}")
            if "page" in chunk.metadata:
                meta_extras.append(f"page={chunk.metadata['page']}")
            location = f" ({', '.join(meta_extras)})" if meta_extras else ""

            parts.append(
                f"[Excerpt {i}]{score_str} — {basename}{location}\n{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)
