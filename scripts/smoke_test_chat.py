import os
import subprocess
import sys
import importlib.util
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_model_path() -> Path:
    return _repo_root() / "google_gemma-3-4b-it-Q8_0.gguf"


def _ensure_llama_library() -> None:
    if os.environ.get("LLAMA_CPP_LIB"):
        return

    repo_root = _repo_root()
    package_dir = repo_root / "src" / "llama_cpp_py_sync"

    spec = importlib.util.find_spec("llama_cpp_py_sync")
    if spec and spec.origin:
        installed_package_dir = Path(spec.origin).resolve().parent
    else:
        installed_package_dir = None

    dll_candidates = [
        package_dir / "llama.dll",
        package_dir / "libllama.dll",
        package_dir / "libllama.so",
        package_dir / "libllama.dylib",
    ]

    if installed_package_dir:
        dll_candidates.extend(
            [
                installed_package_dir / "llama.dll",
                installed_package_dir / "libllama.dll",
                installed_package_dir / "libllama.so",
                installed_package_dir / "libllama.dylib",
            ]
        )

    if any(p.exists() for p in dll_candidates):
        return

    vendor_dir = repo_root / "vendor" / "llama.cpp"
    if not vendor_dir.exists():
        raise RuntimeError(
            "vendor/llama.cpp not found. Run: git clone https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp"
        )

    if shutil.which("cmake") is None:
        raise RuntimeError(
            "CMake was not found in PATH, so the local llama.cpp build cannot run. "
            "Either install CMake + C++ build tools, or install a prebuilt wheel that bundles llama.dll."
        )

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "build_llama_cpp.py"),
        "--no-cuda",
        "--no-rocm",
        "--no-vulkan",
        "--no-metal",
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))


def main() -> int:
    model_path = Path(os.environ.get("LLAMA_MODEL", str(_default_model_path())))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Set LLAMA_MODEL to a valid .gguf path."
        )

    _ensure_llama_library()

    from llama_cpp_py_sync.llama import Llama

    llm = Llama(
        model_path=str(model_path),
        n_ctx=int(os.environ.get("LLAMA_N_CTX", "2048")),
        n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", "0")),
        verbose=os.environ.get("LLAMA_VERBOSE", "0") == "1",
    )

    prompt = os.environ.get(
        "LLAMA_PROMPT",
        "User: Tell me one fun fact about space.\nAssistant:",
    )

    out = llm.generate(
        prompt,
        max_tokens=int(os.environ.get("LLAMA_MAX_TOKENS", "128")),
        temperature=float(os.environ.get("LLAMA_TEMPERATURE", "0.7")),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
