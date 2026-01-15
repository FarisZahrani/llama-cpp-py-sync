#!/usr/bin/env python3
"""
Embeddings example.

This example demonstrates how to generate embeddings and compute similarity.
"""

import sys

from llama_cpp_py_sync import Llama
from llama_cpp_py_sync.embeddings import cosine_similarity


def main():
    if len(sys.argv) < 2:
        print("Usage: python embeddings_example.py <embedding_model_path>")
        print("Note: Use an embedding model (e.g., nomic-embed-text)")
        sys.exit(1)

    model_path = sys.argv[1]

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Python is a popular programming language.",
        "Machine learning models require training data.",
    ]

    print(f"Loading embedding model: {model_path}")

    with Llama(model_path, n_ctx=512, n_gpu_layers=35, embedding=True) as llm:
        print("\nGenerating embeddings...")
        embeddings = [llm.get_embeddings(s) for s in sentences]

        print(f"\nEmbedding dimension: {len(embeddings[0])}")

        print("\nSimilarity Matrix:")
        print("-" * 60)

        for i, _s1 in enumerate(sentences):
            for j, _s2 in enumerate(sentences):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if i <= j:
                    print(f"[{i}] vs [{j}]: {sim:.4f}")

        print("\nSentences:")
        for i, s in enumerate(sentences):
            print(f"  [{i}] {s}")


if __name__ == "__main__":
    main()
