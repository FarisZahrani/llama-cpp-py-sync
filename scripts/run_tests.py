import argparse
import os
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _require_module(module: str, install_hint: str) -> bool:
    if find_spec(module) is not None:
        return True
    print(f"Missing dependency: {module}")
    print("Install it with:")
    print(f"  {install_hint}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local checks before pushing changes.")
    parser.add_argument("--skip-wheel", action="store_true", help="Skip building a wheel.")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the isolated wheel-install smoke test.",
    )
    parser.add_argument(
        "--skip-bindings-check",
        action="store_true",
        help="Skip scripts/validate_cffi_surface.py checks.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()

    python_exe = sys.executable
    if os.environ.get("VIRTUAL_ENV") is None:
        print("Warning: VIRTUAL_ENV is not set. Consider activating venv before running this script.")
        print(f"Using python: {python_exe}")

    pip_install = f"{python_exe} -m pip install -r requirements.txt"

    if not _require_module("ruff", pip_install):
        return 1
    if not _require_module("pytest", pip_install):
        return 1
    if not args.skip_wheel and not _require_module("build", pip_install):
        return 1

    checks: list[list[str]] = [
        [python_exe, "-m", "ruff", "check", "."],
        [python_exe, "-m", "pytest"],
    ]

    if not args.skip_bindings_check:
        checks.append([python_exe, str(repo_root / "scripts" / "validate_cffi_surface.py"), "--check-structs", "--check-enums", "--check-signatures"])

    if not args.skip_wheel:
        checks.append([python_exe, "-m", "build", "--wheel"])

    for cmd in checks:
        rc = _run(cmd, cwd=repo_root)
        if rc != 0:
            print(f"\nFailed: {' '.join(cmd)}")
            return rc

    if not args.skip_smoke:
        dist_dir = repo_root / "dist"
        wheels = sorted(dist_dir.glob("*.whl"))
        if not wheels:
            print("\nNo wheels found in dist/. Run without --skip-wheel or build a wheel first.")
            return 1

        venv_root = Path(tempfile.mkdtemp(prefix="llama_cpp_py_sync_smoke_"))
        try:
            rc = _run([python_exe, "-m", "venv", str(venv_root)], cwd=repo_root)
            if rc != 0:
                return rc

            vpy = _venv_python(venv_root)
            if not vpy.exists():
                print(f"\nSmoke venv python not found: {vpy}")
                return 1

            wheel_path = str(wheels[-1])
            rc = _run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"], cwd=repo_root)
            if rc != 0:
                return rc
            rc = _run([str(vpy), "-m", "pip", "install", "--force-reinstall", wheel_path], cwd=repo_root)
            if rc != 0:
                return rc

            smoke = "from llama_cpp_py_sync._cffi_bindings import get_lib; get_lib(); print('smoke-ok')"
            rc = _run([str(vpy), "-c", smoke], cwd=repo_root)
            if rc != 0:
                print("\nWheel smoke test failed (native library load).")
                return rc
        finally:
            rmtree(venv_root, ignore_errors=True)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
