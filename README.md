# llama-cpp-py-sync

**Auto-synchronized Python bindings for llama.cpp**

[![Build Wheels](https://github.com/llama-cpp-py-sync/llama-cpp-py-sync/actions/workflows/build.yml/badge.svg)](https://github.com/llama-cpp-py-sync/llama-cpp-py-sync/actions/workflows/build.yml)
[![Sync Upstream](https://github.com/llama-cpp-py-sync/llama-cpp-py-sync/actions/workflows/sync.yml/badge.svg)](https://github.com/llama-cpp-py-sync/llama-cpp-py-sync/actions/workflows/sync.yml)
[![PyPI version](https://badge.fury.io/py/llama-cpp-py-sync.svg)](https://badge.fury.io/py/llama-cpp-py-sync)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**llama-cpp-py-sync** solves the long-standing update lag problem in Python bindings for llama.cpp. Instead of manually maintaining bindings that fall behind upstream changes, this project uses **CFFI ABI mode** to automatically generate Python bindings directly from the upstream `llama.h` header file.

### Key Features

- 🔄 **Automatic Synchronization**: Bindings regenerate automatically when upstream llama.cpp changes
- 🚀 **Zero Manual Updates**: No ctypes, no pybind11 maintenance - pure automation
- 📦 **Pre-built Wheels**: Multi-platform wheels built automatically via CI
- 🎯 **GPU Support**: CUDA, ROCm, Vulkan, Metal, and BLAS backends
- 🐍 **Simple API**: High-level Pythonic interface for common operations

> **Note**: This is NOT a fork of llama-cpp-python. It's a completely different system designed for full automation.

## Installation

### From PyPI (Recommended)

```bash
pip install llama-cpp-py-sync
```

### From GitHub Releases

Download the appropriate wheel for your platform from [Releases](https://github.com/llama-cpp-py-sync/llama-cpp-py-sync/releases):

- `*-linux-x86_64-cpu.whl` - Linux x64 CPU-only
- `*-linux-x86_64-cuda.whl` - Linux x64 with CUDA
- `*-macos-arm64-metal.whl` - macOS Apple Silicon with Metal
- `*-macos-x86_64.whl` - macOS Intel
- `*-windows-x64-cpu.whl` - Windows x64 CPU-only

### From Source

```bash
git clone https://github.com/llama-cpp-py-sync/llama-cpp-py-sync.git
cd llama-cpp-py-sync

# Sync upstream llama.cpp
python scripts/sync_upstream.py

# Build the shared library
python scripts/build_llama_cpp.py

# Install the package
pip install -e .
```

## Quick Start

```python
import llama_cpp_py_sync as llama

# Load a model
llm = llama.Llama("path/to/model.gguf", n_ctx=2048, n_gpu_layers=35)

# Generate text
response = llm.generate("Hello, world!", max_tokens=100)
print(response)

# Streaming generation
for token in llm.generate("Write a poem:", max_tokens=100, stream=True):
    print(token, end="", flush=True)

# Clean up
llm.close()
```

### Using Context Manager

```python
with llama.Llama("model.gguf", n_gpu_layers=35) as llm:
    print(llm.generate("Once upon a time"))
```

### Embeddings

```python
# Load an embedding model
with llama.Llama("embed-model.gguf", embedding=True) as llm:
    emb = llm.get_embeddings("Hello, world!")
    print(f"Embedding dimension: {len(emb)}")
```

### Check Available Backends

```python
from llama_cpp_py_sync import get_available_backends, get_backend_info

print(get_available_backends())  # ['cuda', 'blas'] or similar

info = get_backend_info()
print(f"CUDA available: {info.cuda}")
print(f"Metal available: {info.metal}")
```

## How It Works

### Automatic Synchronization

1. **Scheduled Checks**: GitHub Actions checks upstream llama.cpp every 6 hours
2. **SHA Comparison**: Compares upstream HEAD with last synced commit
3. **Auto-Sync**: If changes detected, pulls latest code automatically
4. **Binding Regeneration**: `gen_bindings.py` regenerates CFFI bindings from headers
5. **Wheel Building**: CI builds wheels for all platforms
6. **Auto-Release**: New wheels published to GitHub Releases (and PyPI if configured)

### CFFI ABI Mode

Unlike pybind11 or manual ctypes, CFFI ABI mode:

- Reads C declarations directly (no compilation needed for bindings)
- Loads the shared library at runtime via `ffi.dlopen()`
- Automatically handles type conversions
- Works across platforms without modification

### Version Tracking

Check which llama.cpp version you're running:

```python
import llama_cpp_py_sync as llama

print(f"Package version: {llama.__version__}")
print(f"llama.cpp commit: {llama.__llama_cpp_commit__}")
```

## GPU Backend Selection

### Build-time Detection

The build system automatically detects available backends:

| Backend | Platform | Detection |
|---------|----------|-----------|
| CUDA | Linux, Windows | `CUDA_HOME` or `/usr/local/cuda` |
| ROCm | Linux | `ROCM_PATH` or `/opt/rocm` |
| Metal | macOS | Xcode SDK |
| Vulkan | All | `VULKAN_SDK` environment variable |
| BLAS | All | OpenBLAS, MKL, or Accelerate |

### Runtime Configuration

```python
# Use GPU acceleration
llm = llama.Llama("model.gguf", n_gpu_layers=35)

# CPU only (no GPU offload)
llm = llama.Llama("model.gguf", n_gpu_layers=0)

# Full GPU offload (all layers)
llm = llama.Llama("model.gguf", n_gpu_layers=-1)
```

## API Reference

### Llama Class

```python
class Llama:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,           # Context window size
        n_batch: int = 512,         # Batch size for prompt processing
        n_threads: int = None,      # CPU threads (auto-detect if None)
        n_gpu_layers: int = 0,      # Layers to offload to GPU
        seed: int = -1,             # Random seed (-1 for random)
        use_mmap: bool = True,      # Memory map model file
        use_mlock: bool = False,    # Lock model in RAM
        verbose: bool = False,      # Print loading info
        embedding: bool = False,    # Enable embedding mode
    ): ...
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        stop_sequences: List[str] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]: ...
    
    def tokenize(self, text: str, add_special: bool = True) -> List[int]: ...
    def detokenize(self, tokens: List[int]) -> str: ...
    def get_embeddings(self, text: str) -> List[float]: ...
    def close(self): ...
```

### Backend Functions

```python
def get_available_backends() -> List[str]: ...
def get_backend_info() -> BackendInfo: ...
def is_cuda_available() -> bool: ...
def is_metal_available() -> bool: ...
def is_vulkan_available() -> bool: ...
def is_rocm_available() -> bool: ...
def is_blas_available() -> bool: ...
```

### Embedding Functions

```python
def get_embeddings(model: Union[str, Llama], text: str) -> List[float]: ...
def get_embeddings_batch(model: Union[str, Llama], texts: List[str]) -> List[List[float]]: ...
def cosine_similarity(a: List[float], b: List[float]) -> float: ...
```

## Examples

See the `examples/` directory:

- `basic_generation.py` - Simple text generation
- `streaming_generation.py` - Real-time token streaming
- `embeddings_example.py` - Generate and compare embeddings
- `backend_info.py` - Check available GPU backends
- `benchmark.py` - Measure token throughput

## Building from Source

### Prerequisites

- Python 3.8+
- CMake 3.14+
- C/C++ compiler (GCC, Clang, MSVC)
- Git

### Build Commands

```bash
# Clone repository
git clone https://github.com/llama-cpp-py-sync/llama-cpp-py-sync.git
cd llama-cpp-py-sync

# Sync upstream llama.cpp
python scripts/sync_upstream.py

# Build with auto-detected backends
python scripts/build_llama_cpp.py

# Build with specific backends disabled
python scripts/build_llama_cpp.py --no-cuda --no-vulkan

# Detect available backends without building
python scripts/build_llama_cpp.py --detect-only

# Generate bindings
python scripts/gen_bindings.py

# Build wheel
pip install build
python -m build --wheel
```

## Project Structure

```
llama-cpp-py-sync/
├── src/llama_cpp_py_sync/      # Python package
│   ├── __init__.py             # Public API
│   ├── _cffi_bindings.py       # Auto-generated CFFI bindings
│   ├── _version.py             # Version info
│   ├── llama.py                # High-level Llama class
│   ├── embeddings.py           # Embedding utilities
│   └── backends.py             # Backend detection
├── scripts/                     # Build and sync scripts
│   ├── sync_upstream.py        # Sync upstream llama.cpp
│   ├── gen_bindings.py         # Generate CFFI bindings
│   ├── build_llama_cpp.py      # Build shared library
│   └── auto_version.py         # Version generation
├── examples/                    # Example scripts
├── vendor/llama.cpp/           # Upstream source (git ignored)
├── .github/workflows/          # CI/CD pipelines
├── pyproject.toml              # Package metadata
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

This project uses llama.cpp which is also MIT licensed.

## Acknowledgments

- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - The upstream C/C++ implementation
- [CFFI](https://cffi.readthedocs.io/) - C Foreign Function Interface for Python
