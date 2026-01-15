#!/usr/bin/env python3
"""
Backend information example.

This example shows how to check available backends and system capabilities.
"""

from llama_cpp_py_sync import (
    get_available_backends,
    is_blas_available,
    is_cuda_available,
    is_metal_available,
    is_rocm_available,
    is_vulkan_available,
)
from llama_cpp_py_sync.backends import print_backend_info


def main():
    print("=" * 60)
    print("llama-cpp-py-sync Backend Detection")
    print("=" * 60)

    print("\nAvailable backends:", get_available_backends())

    print("\nIndividual backend checks:")
    print(f"  CUDA:   {is_cuda_available()}")
    print(f"  Metal:  {is_metal_available()}")
    print(f"  Vulkan: {is_vulkan_available()}")
    print(f"  ROCm:   {is_rocm_available()}")
    print(f"  BLAS:   {is_blas_available()}")

    print("\n")
    print_backend_info()


if __name__ == "__main__":
    main()
