#!/usr/bin/env python3

import argparse
import ctypes
import platform
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_library_path(project_root: Path) -> Path:
    pkg_dir = project_root / "src" / "llama_cpp_py_sync"
    system = platform.system().lower()

    if system == "windows":
        names = ["llama.dll", "libllama.dll"]
    elif system == "darwin":
        names = ["libllama.dylib", "libllama.so"]
    else:
        names = ["libllama.so"]

    for name in names:
        p = pkg_dir / name
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not locate built llama.cpp shared library in the package directory. "
        "Expected it next to the Python package under src/llama_cpp_py_sync/."
    )


def _load_library(lib_path: Path) -> ctypes.CDLL:
    system = platform.system().lower()

    if system == "windows":
        try:
            if hasattr(ctypes, "WinDLL"):
                return ctypes.WinDLL(str(lib_path))
        except Exception:
            pass

    return ctypes.CDLL(str(lib_path))


def _require_symbol(lib: ctypes.CDLL, name: str) -> None:
    try:
        getattr(lib, name)
    except AttributeError as e:
        raise RuntimeError(f"Missing required export: {name}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate llama.cpp shared library exports")
    parser.add_argument(
        "--lib",
        type=Path,
        default=None,
        help="Path to the built llama shared library (defaults to src/llama_cpp_py_sync/*llama*)",
    )

    args = parser.parse_args()

    root = _project_root()
    lib_path = args.lib or _default_library_path(root)

    lib = _load_library(lib_path)

    required = [
        "llama_backend_init",
        "llama_model_default_params",
        "llama_context_default_params",
        "llama_print_system_info",
    ]

    for sym in required:
        _require_symbol(lib, sym)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
