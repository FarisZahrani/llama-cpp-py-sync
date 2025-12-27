import argparse
import importlib.util
import os
import subprocess
import sys
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

    dll_candidates: list[Path] = []
    if installed_package_dir:
        dll_candidates.extend(
            [
                installed_package_dir / "llama.dll",
                installed_package_dir / "libllama.dll",
                installed_package_dir / "libllama.so",
                installed_package_dir / "libllama.dylib",
            ]
        )
    dll_candidates.extend(
        [
            package_dir / "llama.dll",
            package_dir / "libllama.dll",
            package_dir / "libllama.so",
            package_dir / "libllama.dylib",
        ]
    )

    existing = next((p for p in dll_candidates if p.exists()), None)
    if existing is not None:
        os.environ["LLAMA_CPP_LIB"] = str(existing)
        return

    vendor_dir = repo_root / "vendor" / "llama.cpp"
    if not vendor_dir.exists():
        raise RuntimeError(
            "vendor/llama.cpp not found. Run: git clone https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp"
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

    existing = next((p for p in dll_candidates if p.exists()), None)
    if existing is None:
        raise RuntimeError(
            "llama.cpp build completed, but no shared library was found in expected locations. "
            "Set LLAMA_CPP_LIB to the built library path."
        )
    os.environ["LLAMA_CPP_LIB"] = str(existing)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a GGUF model and chat.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to GGUF model (default: env LLAMA_MODEL or repo default)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="One-shot prompt; if omitted, starts interactive chat.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("LLAMA_MAX_TOKENS", "128")),
        help="Max tokens per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LLAMA_TEMPERATURE", "0.7")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=int(os.environ.get("LLAMA_N_CTX", "2048")),
        help="Context length.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=int(os.environ.get("LLAMA_N_GPU_LAYERS", "0")),
        help="GPU layers (0 = CPU).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose model load/logs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set llama_cpp_py_sync log level (default: INFO)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose step-by-step logging for troubleshooting.",
    )
    return parser.parse_args()


def _resolve_model_path(args_model: Path | None) -> Path:
    if args_model:
        return args_model
    env_model = os.environ.get("LLAMA_MODEL")
    if env_model:
        return Path(env_model)
    return _default_model_path()


def main() -> int:
    try:
        return _run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


def _run() -> int:
    args = _parse_args()

    if args.prompt is None and not sys.stdin.isatty():
        args.prompt = "Say 'ok'."

    def log(msg: str) -> None:
        if args.debug:
            print(msg)
            sys.stdout.flush()

    model_path = _resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Set --model or LLAMA_MODEL to a valid .gguf path."
        )

    log(f"[debug] model_path={model_path}")

    _ensure_llama_library()
    log("[debug] ensured llama library")

    print(f"Loading model (this may take a few seconds): {model_path}")
    sys.stdout.flush()
    # set log level early
    os.environ["LLAMA_LOG_LEVEL"] = args.log_level
    log(f"[debug] LLAMA_LOG_LEVEL={args.log_level}")
    from llama_cpp_py_sync.llama import Llama
    log("[debug] imported Llama")

    print("Initializing Llama...")
    sys.stdout.flush()
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    print("Model loaded successfully.")
    sys.stdout.flush()
    log("[debug] Llama constructed")

    if args.prompt:
        print("Generating one-shot response...")
        sys.stdout.flush()
        try:
            out = llm.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print("Response:")
            print(out)
        except Exception as e:
            print(f"Generation failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
        return 0

    print(f"Model loaded: {model_path}")
    print("Interactive chat. Ctrl+C or blank line to exit.")
    sys.stdout.flush()

    history: list[str] = []
    try:
        while True:
            user_msg = input("You: ").strip()
            if not user_msg:
                break

            prompt = "".join(history) + f"User: {user_msg}\nAssistant:"
            try:
                out = llm.generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop_sequences=["\nUser:", "\nYou:", "\nAssistant:"],
                )

                out_text = str(out).strip()
                print(f"Assistant: {out_text}")
                history.append(f"User: {user_msg}\nAssistant: {out_text}\n")
            except ValueError as e:
                # Most commonly: prompt too long for the configured context.
                if "Prompt too long" in str(e):
                    history = []
                    try:
                        out = llm.generate(
                            f"User: {user_msg}\nAssistant:",
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            stop_sequences=["\nUser:", "\nYou:", "\nAssistant:"],
                        )
                        out_text = str(out).strip()
                        print(f"Assistant: {out_text}")
                        history.append(f"User: {user_msg}\nAssistant: {out_text}\n")
                        continue
                    except Exception as inner:
                        print(f"Generation failed: {inner}")
                        if args.debug:
                            import traceback
                            traceback.print_exc()
                        break

                print(f"Generation failed: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                break
            except Exception as e:
                print(f"Generation failed: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                break
    except KeyboardInterrupt:
        print("\nExiting.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
