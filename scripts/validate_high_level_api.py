#!/usr/bin/env python3
"""Validate that high-level Python uses only names that exist in the CFFI cdef.

After llama.cpp sync, struct fields or function names in ``llama.h`` can change.
``validate_cffi_surface.py`` checks header vs cdef; this script checks that
``llama.py`` (and optionally other modules) only references:

- ``ctx_params.<field>`` where ``<field>`` exists on ``struct llama_context_params``
  in the cdef
- ``model_params.<field>`` on ``struct llama_model_params``
- ``self._lib.<fn>`` and string literals like ``getattr(..., "llama_...")`` where
  ``<fn>`` appears as a function in the cdef

This does not require a vendored ``llama.h`` — the cdef is the ABI contract for
the Python package. Run with the same tree you ship (committed ``_cffi_bindings.py``).

Usage:
    python scripts/validate_high_level_api.py
    python scripts/validate_high_level_api.py --module src/llama_cpp_py_sync/embeddings.py
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_validate_cffi():
    path = Path(__file__).resolve().parent / "validate_cffi_surface.py"
    spec = importlib.util.spec_from_file_location("validate_cffi_surface", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_lib_api_names(py_source: str) -> set[str]:
    """Collect llama_* API names referenced from Python source."""
    names: set[str] = set()

    for m in re.finditer(r"\bself\._lib\.(llama_[a-zA-Z0-9_]+)\b", py_source):
        names.add(m.group(1))

    for m in re.finditer(r"\blib\.(llama_[a-zA-Z0-9_]+)\b", py_source):
        names.add(m.group(1))

    for m in re.finditer(
        r"\b(?:getattr|hasattr)\s*\(\s*[^,]+,\s*[\"'](llama_[a-zA-Z0-9_]+)[\"']\s*\)",
        py_source,
    ):
        names.add(m.group(1))

    for m in re.finditer(
        r"\bget_\w+\s*=\s*getattr\s*\(\s*[^,]+,\s*[\"'](llama_[a-zA-Z0-9_]+)[\"']",
        py_source,
    ):
        names.add(m.group(1))

    return names


def _extract_ctx_param_fields(py_source: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\bctx_params\.([a-zA-Z0-9_]+)\b", py_source)}


def _extract_model_param_fields(py_source: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\bmodel_params\.([a-zA-Z0-9_]+)\b", py_source)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate high-level Python against CFFI cdef struct fields and functions"
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=None,
        help="Path to _cffi_bindings.py (default: src/.../_cffi_bindings.py)",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help="Extra Python file to scan (repeatable). Default: llama.py",
    )
    args = parser.parse_args()

    root = _project_root()
    vcs = _load_validate_cffi()
    bindings_path = args.bindings or (root / "src" / "llama_cpp_py_sync" / "_cffi_bindings.py")
    bindings_text = bindings_path.read_text(encoding="utf-8", errors="ignore")
    cdef_text = vcs._extract_cdef_text(bindings_text)
    cdef_funcs = vcs._extract_cdef_functions(cdef_text)
    cdef_structs = vcs._extract_structs(cdef_text)

    modules = [root / "src" / "llama_cpp_py_sync" / "llama.py"]
    if args.module:
        for p in args.module:
            modules.append(Path(p) if Path(p).is_absolute() else root / p)

    combined = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in modules)

    ctx_used = _extract_ctx_param_fields(combined)
    model_used = _extract_model_param_fields(combined)
    lib_used = _extract_lib_api_names(combined)

    ctx_fields = cdef_structs.get("llama_context_params", set())
    model_fields = cdef_structs.get("llama_model_params", set())

    bad_ctx = sorted(ctx_used - ctx_fields)
    bad_model = sorted(model_used - model_fields)
    bad_lib = sorted(lib_used - cdef_funcs)

    print(f"Scanned {len(modules)} module(s); cdef functions: {len(cdef_funcs)}")
    print(f"ctx_params.* fields used: {sorted(ctx_used)}")
    print(f"model_params.* fields used: {sorted(model_used)}")
    print(f"lib API symbols used: {len(lib_used)}")

    ok = True
    if bad_ctx:
        ok = False
        print("\nERROR: ctx_params fields not in cdef struct llama_context_params:")
        for x in bad_ctx:
            print(f"  - {x}")
    if bad_model:
        ok = False
        print("\nERROR: model_params fields not in cdef struct llama_model_params:")
        for x in bad_model:
            print(f"  - {x}")
    if bad_lib:
        ok = False
        print("\nERROR: llama_* calls not found as cdef functions:")
        for x in bad_lib:
            print(f"  - {x}")

    if ok:
        print("\nOK: high-level references match cdef.")
        return 0
    print(
        "\nFix: sync/regenerate _cffi_bindings.py or update llama.py to match the cdef.\n"
        "Also run: python scripts/validate_cffi_surface.py --check-structs ... against vendor llama.h",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
