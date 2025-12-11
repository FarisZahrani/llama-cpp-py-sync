#!/usr/bin/env python3
"""
Basic text generation example.

This example demonstrates how to load a GGUF model and generate text.
"""

import sys
from llama_cpp_py_sync import Llama


def main():
    if len(sys.argv) < 2:
        print("Usage: python basic_generation.py <model_path> [prompt]")
        print("Example: python basic_generation.py model.gguf 'Hello, world!'")
        sys.exit(1)
    
    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    
    print(f"Loading model: {model_path}")
    
    with Llama(
        model_path,
        n_ctx=2048,
        n_gpu_layers=35,
        verbose=True
    ) as llm:
        print(f"\nPrompt: {prompt}")
        print("\nGenerating...")
        print("-" * 40)
        
        response = llm.generate(
            prompt,
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
        )
        
        print(response)
        print("-" * 40)


if __name__ == "__main__":
    main()
