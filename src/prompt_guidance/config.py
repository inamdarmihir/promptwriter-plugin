from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Provider selection ────────────────────────────────────────
    llm_provider: str = Field("ollama", description="ollama | openai | anthropic | mistral")
    embed_provider: str = Field("ollama", description="ollama | openai | sentence-transformers")

    # ── Ollama ────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"

    # ── OpenAI ────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # ── Anthropic ─────────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"

    # ── Mistral ───────────────────────────────────────────────────
    mistral_api_key: str = ""
    mistral_llm_model: str = "mistral-small-latest"
    mistral_embed_model: str = "mistral-embed"

    # ── Sentence Transformers ─────────────────────────────────────
    st_model: str = "all-MiniLM-L6-v2"

    # ── Embeddings ────────────────────────────────────────────────
    embed_dimensions: int = 768  # nomic-embed-text; change per model

    # ── Qdrant ────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "prompt_knowledge"

    # ── RAG / Chunking ────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5


settings = Settings()
