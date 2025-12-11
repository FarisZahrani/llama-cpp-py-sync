"""Test that the package can be imported."""

import pytest


def test_import():
    """Test basic import."""
    import llama_cpp_py_sync
    assert hasattr(llama_cpp_py_sync, "__version__")
    assert hasattr(llama_cpp_py_sync, "__llama_cpp_commit__")


def test_import_llama_class():
    """Test Llama class import."""
    from llama_cpp_py_sync import Llama
    assert Llama is not None


def test_import_backend_functions():
    """Test backend function imports."""
    from llama_cpp_py_sync import (
        get_available_backends,
        is_cuda_available,
        is_metal_available,
        is_vulkan_available,
        is_rocm_available,
        is_blas_available,
        get_backend_info,
    )
    assert callable(get_available_backends)
    assert callable(is_cuda_available)


def test_import_embedding_functions():
    """Test embedding function imports."""
    from llama_cpp_py_sync import get_embeddings, get_embeddings_batch
    assert callable(get_embeddings)
    assert callable(get_embeddings_batch)


def test_version_format():
    """Test version string format."""
    from llama_cpp_py_sync import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0
