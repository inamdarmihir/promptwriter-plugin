"""
Embedding Providers
───────────────────
Supported: Ollama (local) · OpenAI · Sentence Transformers (fully local)

All share the same BaseEmbeddings interface:
    embed(text: str) -> list[float]
    embed_batch(texts: list[str]) -> list[list[float]]

Factory:
    get_embeddings(provider="ollama", model=None, **kwargs) -> BaseEmbeddings
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from prompt_guidance.config import settings


# ─────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────

class BaseEmbeddings(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


# ─────────────────────────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────────────────────────

class OllamaEmbeddings(BaseEmbeddings):
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        import ollama
        self._model = model or settings.ollama_embed_model
        self._client = ollama.Client(host=host or settings.ollama_host)
        self._dims = settings.embed_dimensions

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings(model=self._model, prompt=text)
        return resp.embedding


# ─────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────

class OpenAIEmbeddings(BaseEmbeddings):
    # text-embedding-3-small → 1536 dims by default
    _DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI
        self._model = model or settings.openai_embed_model
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def dimensions(self) -> int:
        return self._DIM_MAP.get(self._model, 1536)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


# ─────────────────────────────────────────────────────────────────
# Sentence Transformers  (100 % local, no API key)
# ─────────────────────────────────────────────────────────────────

class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(self, model: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self._model_name = model or settings.st_model
        self._model = SentenceTransformer(self._model_name)
        self._dims = self._model.get_sentence_embedding_dimension()

    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[BaseEmbeddings]] = {
    "ollama": OllamaEmbeddings,
    "openai": OpenAIEmbeddings,
    "sentence-transformers": SentenceTransformerEmbeddings,
}


def get_embeddings(provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> BaseEmbeddings:
    """
    Return an embeddings instance for the given provider.

    Args:
        provider: "ollama" | "openai" | "sentence-transformers"
                  Defaults to settings.embed_provider.
        model:    Override the default embedding model.
        **kwargs: Passed directly to the provider constructor.
    """
    key = (provider or settings.embed_provider).lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(f"Unknown embeddings provider '{key}'. Supported: {list(_PROVIDERS)}")
    if model:
        kwargs["model"] = model
    return cls(**kwargs)
