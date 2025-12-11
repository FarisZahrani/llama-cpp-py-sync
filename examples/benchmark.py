#!/usr/bin/env python3
"""
Benchmark example.

This example benchmarks token throughput for prompt processing and generation.
"""

import sys
import time
from llama_cpp_py_sync import Llama


def benchmark_generation(llm: Llama, prompt: str, n_tokens: int = 128) -> dict:
    """Benchmark generation speed."""
    tokens = llm.tokenize(prompt)
    prompt_tokens = len(tokens)
    
    start = time.perf_counter()
    response = llm.generate(prompt, max_tokens=n_tokens)
    end = time.perf_counter()
    
    output_tokens = len(llm.tokenize(response, add_special=False))
    total_time = end - start
    
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_time": total_time,
        "tokens_per_second": output_tokens / total_time if total_time > 0 else 0,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <model_path> [n_tokens]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    
    prompts = [
        "Write a detailed explanation of how neural networks work:",
        "The history of artificial intelligence began",
        "In computer science, algorithms are",
    ]
    
    print(f"Loading model: {model_path}")
    print(f"Benchmark: {n_tokens} tokens per prompt")
    print("=" * 60)
    
    with Llama(model_path, n_ctx=2048, n_gpu_layers=35) as llm:
        print(f"\nModel: {llm.get_model_desc()}")
        print(f"Context size: {llm.n_ctx}")
        print(f"Vocab size: {llm.n_vocab}")
        print("=" * 60)
        
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(prompts):
            print(f"\nRun {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            result = benchmark_generation(llm, prompt, n_tokens)
            
            print(f"  Prompt tokens:  {result['prompt_tokens']}")
            print(f"  Output tokens:  {result['output_tokens']}")
            print(f"  Time:           {result['total_time']:.2f}s")
            print(f"  Speed:          {result['tokens_per_second']:.2f} tok/s")
            
            total_tokens += result['output_tokens']
            total_time += result['total_time']
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total tokens generated: {total_tokens}")
        print(f"Total time:             {total_time:.2f}s")
        print(f"Average speed:          {total_tokens/total_time:.2f} tok/s")


if __name__ == "__main__":
    main()
