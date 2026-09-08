"""Runtime configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # LLM (OpenRouter, OpenAI-compatible)
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password123"))

    # Embeddings
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_dim: int = field(default_factory=lambda: _int("EMBED_DIM", 384))

    # Ingestion tuning
    chunk_tokens: int = field(default_factory=lambda: _int("CHUNK_TOKENS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 100))
    max_gleanings: int = field(default_factory=lambda: _int("MAX_GLEANINGS", 1))
    max_cluster_size: int = field(default_factory=lambda: _int("MAX_CLUSTER_SIZE", 10))
    llm_concurrency: int = field(default_factory=lambda: _int("LLM_CONCURRENCY", 4))

    # Paths
    input_dir: Path = ROOT / "input"
    output_dir: Path = ROOT / "output"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    def validate(self) -> None:
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (copy .env.example to .env).")

    def ensure_dirs(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
