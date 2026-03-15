"""
LLM Providers
─────────────
Supported: Ollama (local) · OpenAI · Anthropic/Claude · Mistral

All share the same BaseLLM interface:
    generate(system: str, user: str) -> str

Factory:
    get_llm(provider="ollama", model=None, **kwargs) -> BaseLLM
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from prompt_guidance.config import settings


# ─────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Call the model and return the full text response."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ─────────────────────────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        import ollama
        self._model = model or settings.ollama_llm_model
        self._client = ollama.Client(host=host or settings.ollama_host)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.message.content


# ─────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI
        self._model = model or settings.openai_llm_model
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
# Anthropic / Claude
# ─────────────────────────────────────────────────────────────────

class AnthropicLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import anthropic
        self._model = model or settings.anthropic_model
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key or None)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


# ─────────────────────────────────────────────────────────────────
# Mistral
# ─────────────────────────────────────────────────────────────────

class MistralLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from mistralai import Mistral
        self._model = model or settings.mistral_llm_model
        self._client = Mistral(api_key=api_key or settings.mistral_api_key or None)

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[BaseLLM]] = {
    "ollama": OllamaLLM,
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "mistral": MistralLLM,
}


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> BaseLLM:
    """
    Return an LLM instance for the given provider.

    Args:
        provider: "ollama" | "openai" | "anthropic" | "mistral"
                  Defaults to settings.llm_provider.
        model:    Override the default model for that provider.
        **kwargs: Passed directly to the provider constructor (e.g. api_key).
    """
    key = (provider or settings.llm_provider).lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{key}'. Supported: {list(_PROVIDERS)}")
    if model:
        kwargs["model"] = model
    return cls(**kwargs)
