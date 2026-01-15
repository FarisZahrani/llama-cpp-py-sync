#!/usr/bin/env python3
"""
Streaming text generation example.

This example demonstrates real-time token streaming during generation.
"""

import sys

from llama_cpp_py_sync import Llama


def main():
    if len(sys.argv) < 2:
        print("Usage: python streaming_generation.py <model_path> [prompt]")
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a short poem about coding:"

    print(f"Loading model: {model_path}")

    with Llama(model_path, n_ctx=2048, n_gpu_layers=35) as llm:
        print(f"\nPrompt: {prompt}")
        print("\nStreaming response:")
        print("-" * 40)

        for token in llm.generate(prompt, max_tokens=256, stream=True):
            print(token, end="", flush=True)

        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()
