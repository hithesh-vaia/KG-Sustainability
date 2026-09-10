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
    # cap completion length: keeps cost down and avoids OpenRouter pre-authorizing
    # the model's full max_tokens (which triggers 402s on a low balance)
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 4096))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password123"))

    # Embeddings -- EMBED_PROVIDER is "fastembed" (local, no API key) or "gemini"
    embed_provider: str = field(
        default_factory=lambda: os.getenv("EMBED_PROVIDER", "fastembed").strip().lower()
    )
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_dim: int = field(default_factory=lambda: _int("EMBED_DIM", 384))
    embed_batch: int = field(default_factory=lambda: _int("EMBED_BATCH", 100))

    # Gemini (embeddings) -- either an AI Studio API key, or Vertex AI via a
    # service account (GOOGLE_APPLICATION_CREDENTIALS + project/location).
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_embed_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )
    gcp_project: str = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID", ""))
    gcp_location: str = field(default_factory=lambda: os.getenv("GCP_LOCATION", "us-central1"))
    google_credentials: str = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    )

    @property
    def credentials_path(self) -> Path | None:
        """Absolute path to the service-account JSON, if it exists."""
        if not self.google_credentials:
            return None
        p = Path(self.google_credentials)
        if not p.is_absolute():
            p = ROOT / p
        return p if p.is_file() else None

    @property
    def use_vertex(self) -> bool:
        """Prefer Vertex AI when a service account and project are available.

        An AI Studio key starts with 'AIza'; anything else (e.g. an OAuth-style
        'AQ.' token) will be rejected by generativelanguage.googleapis.com, so
        Vertex is the correct route whenever it is configured.
        """
        return bool(self.gcp_project and self.credentials_path)

    # Ingestion tuning.
    # NOTE: raising CHUNK_TOKENS to 2400 + the owner-edge mandate in the
    # extraction prompt makes per-chunk output exceed max_tokens (40-90 items per
    # chunk), truncating the JSON. Keep at 1200 unless the model has a very large
    # output budget and is not a reasoning model.
    chunk_tokens: int = field(default_factory=lambda: _int("CHUNK_TOKENS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 100))
    # the organisation that owns the ingested documents; anchors orphan figures
    reporting_entity: str = field(default_factory=lambda: os.getenv("REPORTING_ENTITY", ""))
    max_gleanings: int = field(default_factory=lambda: _int("MAX_GLEANINGS", 1))
    max_cluster_size: int = field(default_factory=lambda: _int("MAX_CLUSTER_SIZE", 10))
    llm_concurrency: int = field(default_factory=lambda: _int("LLM_CONCURRENCY", 4))
    # Derive SAME_METRIC_PRIOR_PERIOD edges chaining a metric across reporting
    # periods -- connects the many table-only measurement nodes to their
    # other-year twins. Neutral-to-slightly-negative on the local eval, positive
    # for graph structure / global search. Excluded from local-search neighbour
    # expansion in graphdb.py so it does not bloat the answer context.
    link_period_siblings: bool = field(
        default_factory=lambda: os.getenv("LINK_PERIOD_SIBLINGS", "1") == "1")

    # Paths
    input_dir: Path = ROOT / "input"
    output_dir: Path = ROOT / "output"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    def validate(self) -> None:
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (copy .env.example to .env).")

    def validate_embeddings(self) -> None:
        if self.embed_provider not in ("fastembed", "gemini"):
            raise RuntimeError(
                f"EMBED_PROVIDER must be 'fastembed' or 'gemini', got {self.embed_provider!r}"
            )
        if self.embed_provider == "gemini" and not (self.use_vertex or self.gemini_api_key):
            raise RuntimeError(
                "EMBED_PROVIDER=gemini needs either GOOGLE_APPLICATION_CREDENTIALS + "
                "GCP_PROJECT_ID (Vertex AI) or a GEMINI_API_KEY (AI Studio)."
            )

    def ensure_dirs(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
