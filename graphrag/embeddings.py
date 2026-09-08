"""Local embeddings via fastembed (no external API needed)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import CONFIG


@lru_cache(maxsize=2)
def _model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name)


def embed(texts: list[str], model: str | None = None) -> np.ndarray:
    """Return an (n, dim) float32 array of embeddings."""
    if not texts:
        return np.zeros((0, CONFIG.embed_dim), dtype=np.float32)
    name = model or CONFIG.embed_model
    vectors = list(_model(name).embed(texts))
    return np.asarray(vectors, dtype=np.float32)


def embed_one(text: str, model: str | None = None) -> list[float]:
    return embed([text], model)[0].tolist()
