"""Embeddings: local fastembed (no API key) or Gemini (fast, batched).

Selected with EMBED_PROVIDER in .env. Gemini is materially faster on large
corpora because the work happens server-side and in batches, but it costs an
API call per batch and EMBED_DIM must match the Neo4j vector index dimension.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import CONFIG
from .log import get_logger

log = get_logger()

# Gemini embeds documents and queries differently; using the right task type is
# worth a few points of retrieval accuracy.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


# ---------------------------------------------------------------- fastembed --
@lru_cache(maxsize=2)
def _fastembed_model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name)


def _embed_fastembed(texts: list[str], model: str) -> np.ndarray:
    vectors = list(_fastembed_model(model).embed(texts))
    return np.asarray(vectors, dtype=np.float32)


# ------------------------------------------------------------------- gemini --
@lru_cache(maxsize=1)
def _gemini_client(vertex: bool, project: str, location: str, api_key: str):
    import os

    from google import genai

    if vertex:
        # the SDK reads ADC from this env var; make the path absolute so the
        # client works regardless of the process working directory
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CONFIG.credentials_path)
        log.info("gemini embeddings via Vertex AI (project=%s, location=%s)", project, location)
        return genai.Client(vertexai=True, project=project, location=location)
    log.info("gemini embeddings via AI Studio API key")
    return genai.Client(api_key=api_key)


def _client():
    return _gemini_client(
        CONFIG.use_vertex, CONFIG.gcp_project, CONFIG.gcp_location, CONFIG.gemini_api_key
    )


@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
def _gemini_batch(batch: list[str], model: str, dim: int, task: str) -> list[list[float]]:
    from google.genai import types

    resp = _client().models.embed_content(
        model=model,
        contents=batch,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=dim),
    )
    return [e.values for e in resp.embeddings]


def _embed_gemini(texts: list[str], model: str, task: str) -> np.ndarray:
    dim = CONFIG.embed_dim
    size = max(1, CONFIG.embed_batch)
    out: list[list[float]] = []
    total = (len(texts) + size - 1) // size
    for i in range(0, len(texts), size):
        batch = [t if t.strip() else " " for t in texts[i : i + size]]
        out.extend(_gemini_batch(batch, model, dim, task))
        if total > 1:
            log.info("  embedded %d/%d texts (gemini)", min(i + size, len(texts)), len(texts))
    arr = np.asarray(out, dtype=np.float32)
    # gemini-embedding-001 is not unit-normalised below 3072 dims; Neo4j cosine
    # tolerates that, but normalising keeps scores comparable across providers.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


# -------------------------------------------------------------------- public --
def embed(texts: list[str], model: str | None = None, task: str = TASK_DOCUMENT) -> np.ndarray:
    """Return an (n, EMBED_DIM) float32 array of embeddings."""
    if not texts:
        return np.zeros((0, CONFIG.embed_dim), dtype=np.float32)
    CONFIG.validate_embeddings()
    if CONFIG.embed_provider == "gemini":
        return _embed_gemini(texts, model or CONFIG.gemini_embed_model, task)
    return _embed_fastembed(texts, model or CONFIG.embed_model)


def embed_one(text: str, model: str | None = None) -> list[float]:
    """Embed a single search query (Gemini uses the query task type)."""
    return embed([text], model, task=TASK_QUERY)[0].tolist()
