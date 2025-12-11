"""
Embedding utilities for llama-cpp-py-sync.

Provides convenient functions for generating embeddings from text using
llama.cpp models that support embedding generation.
"""

from __future__ import annotations

import numpy as np

from llama_cpp_py_sync.llama import Llama


def normalize_embedding(embedding: list[float]) -> list[float]:
    """
    Normalize an embedding vector to unit length.

    Args:
        embedding: Raw embedding vector.

    Returns:
        Normalized embedding vector.
    """
    arr = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def get_embeddings(
    model: str | Llama,
    text: str,
    normalize: bool = True,
    n_ctx: int = 512,
    n_gpu_layers: int = 0,
) -> list[float]:
    """
    Get embeddings for a single text string.

    Args:
        model: Either a path to a GGUF model file or an existing Llama instance.
        text: Text to embed.
        normalize: Whether to normalize the embedding to unit length.
        n_ctx: Context size (only used if model is a path).
        n_gpu_layers: GPU layers (only used if model is a path).

    Returns:
        Embedding vector as a list of floats.

    Example:
        >>> emb = get_embeddings("model.gguf", "Hello, world!")
        >>> print(len(emb))
        4096
    """
    if isinstance(model, str):
        with Llama(
            model,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True
        ) as llm:
            embedding = llm.get_embeddings(text)
    else:
        embedding = model.get_embeddings(text)

    if normalize:
        embedding = normalize_embedding(embedding)

    return embedding


def get_embeddings_batch(
    model: str | Llama,
    texts: list[str],
    normalize: bool = True,
    n_ctx: int = 512,
    n_gpu_layers: int = 0,
) -> list[list[float]]:
    """
    Get embeddings for multiple text strings.

    Args:
        model: Either a path to a GGUF model file or an existing Llama instance.
        texts: List of texts to embed.
        normalize: Whether to normalize embeddings to unit length.
        n_ctx: Context size (only used if model is a path).
        n_gpu_layers: GPU layers (only used if model is a path).

    Returns:
        List of embedding vectors.

    Example:
        >>> embs = get_embeddings_batch("model.gguf", ["Hello", "World"])
        >>> print(len(embs))
        2
    """
    if isinstance(model, str):
        with Llama(
            model,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True
        ) as llm:
            embeddings = [llm.get_embeddings(text) for text in texts]
    else:
        embeddings = [model.get_embeddings(text) for text in texts]

    if normalize:
        embeddings = [normalize_embedding(emb) for emb in embeddings]

    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)

    dot_product = np.dot(arr_a, arr_b)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """
    Compute Euclidean distance between two embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Euclidean distance (lower = more similar).
    """
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(arr_a - arr_b))
