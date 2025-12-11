"""Test that the package can be imported."""

import pytest


def test_version_import():
    """Test version can be imported without library."""
    from llama_cpp_py_sync._version import __version__, __llama_cpp_commit__
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    assert isinstance(__llama_cpp_commit__, str)


def test_cffi_bindings_module():
    """Test CFFI bindings module structure."""
    from llama_cpp_py_sync._cffi_bindings import ffi, get_ffi, get_lib
    assert ffi is not None
    assert callable(get_ffi)
    assert callable(get_lib)


def test_llama_class_definition():
    """Test Llama class can be imported."""
    from llama_cpp_py_sync.llama import Llama
    assert Llama is not None
    assert hasattr(Llama, '__init__')
    assert hasattr(Llama, 'generate')
    assert hasattr(Llama, 'tokenize')


def test_backends_module_structure():
    """Test backends module structure without loading library."""
    from llama_cpp_py_sync.backends import BackendInfo
    assert BackendInfo is not None
    
    info = BackendInfo()
    assert info.cuda == False
    assert info.metal == False


def test_embeddings_module_structure():
    """Test embeddings module structure."""
    from llama_cpp_py_sync.embeddings import (
        normalize_embedding,
        cosine_similarity,
        euclidean_distance,
    )
    assert callable(normalize_embedding)
    assert callable(cosine_similarity)
    assert callable(euclidean_distance)


def test_cosine_similarity():
    """Test cosine similarity function."""
    from llama_cpp_py_sync.embeddings import cosine_similarity
    
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, b) - 1.0) < 0.001
    
    c = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(a, c) - 0.0) < 0.001


def test_normalize_embedding():
    """Test embedding normalization."""
    from llama_cpp_py_sync.embeddings import normalize_embedding
    import math
    
    emb = [3.0, 4.0]
    normalized = normalize_embedding(emb)
    
    length = math.sqrt(sum(x*x for x in normalized))
    assert abs(length - 1.0) < 0.001
