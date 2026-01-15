import argparse
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def _require_module(module: str, install_hint: str) -> bool:
    if find_spec(module) is not None:
        return True
    print(f"Missing dependency: {module}")
    print("Install it with:")
    print(f"  {install_hint}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local checks before pushing changes.")
    parser.add_argument(
        "--build-wheel",
        action="store_true",
        help="Also build a wheel (python -m build --wheel).",
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
    if args.build_wheel and not _require_module("build", pip_install):
        return 1

    checks: list[list[str]] = [
        [python_exe, "-m", "ruff", "check", "."],
        [python_exe, "-m", "pytest"],
    ]

    if args.build_wheel:
        checks.append([python_exe, "-m", "build", "--wheel"])

    for cmd in checks:
        rc = _run(cmd, cwd=repo_root)
        if rc != 0:
            print(f"\nFailed: {' '.join(cmd)}")
            return rc

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
