#!/usr/bin/env python3
"""
Run the same pipeline as GitHub Actions locally (pinned vendor → gen bindings →
optional native build → validate → lint/tests).

  1. Vendor llama.cpp at the pinned SHA (``.llama_cpp_sync_state.json``), same as
     the ``validate-bindings`` job (clone + fetch + checkout).
  2. Regenerate ``_cffi_bindings.py`` (``scripts/gen_bindings.py``).
  3. Optionally build the native library (``scripts/build_llama_cpp.py``).
  4. Validate CFFI vs header and high-level Python vs cdef.
  5. Ruff + pytest (+ optional wheel into ``test_temp/dist/``), same spirit as ``scripts/run_tests.py``.

Usage (from repo root, venv activated recommended)::

    python scripts/run_local_ci.py
    python scripts/run_local_ci.py --skip-build
    python scripts/run_local_ci.py --skip-vendor   # use existing vendor/ as-is
    python scripts/run_local_ci.py --pull-upstream # optional: sync_upstream.py first (master)

Output from each step is **streamed live** to the terminal. A **LOCAL CI REPORT**
is printed at the end (per-step status, duration, and failure tails). Use
``--json report.json`` for machine-readable step metadata and truncated failure logs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from subprocess import list2cmdline
from typing import Any

# Cap stored detail for JSON / failure summary (full build logs stay on console).
_MAX_STORED_DETAIL = 16000


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _cmdline_display(cmd: list[str]) -> str:
    try:
        return list2cmdline(cmd)
    except Exception:
        return " ".join(str(c) for c in cmd)


def _run_live(
    cmd: list[str],
    cwd: Path,
    label: str,
) -> tuple[bool, str]:
    """Stream stdout+stderr to the console; return (ok, detail string for report/JSON)."""
    print(f"\n{'─' * 72}", flush=True)
    print(f"▶ {label}", flush=True)
    print(f"$ {_cmdline_display(cmd)}", flush=True)
    print(f"{'─' * 72}\n", flush=True)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except OSError as e:
        return False, f"failed to start: {e}"

    assert proc.stdout is not None
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    parts: list[str] = []
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(enc, errors="replace")
            parts.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
    finally:
        proc.stdout.close()

    rc = proc.wait()
    full = "".join(parts)
    ok = rc == 0
    if ok:
        detail = f"exit=0, {len(full)} chars (streamed above)"
    else:
        tail = full[-_MAX_STORED_DETAIL:] if len(full) > _MAX_STORED_DETAIL else full
        detail = f"exit={rc}\n{tail}"
    return ok, detail


def _read_pinned_sha(root: Path) -> str:
    state_path = root / ".llama_cpp_sync_state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing {state_path.name}; cannot resolve pinned llama.cpp SHA."
        )
    data = json.loads(state_path.read_text(encoding="utf-8"))
    sha = (data.get("last_sync_sha") or "").strip()
    if not sha:
        raise ValueError(f"{state_path} has no last_sync_sha")
    return sha


def _bindings_timestamp(root: Path) -> str:
    p = root / "src" / "llama_cpp_py_sync" / "_cffi_bindings.py"
    if not p.exists():
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    m = re.search(r"Generated:\s*(.+)", p.read_text(encoding="utf-8", errors="ignore"))
    if m:
        return m.group(1).strip()
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ensure_vendor_at_pinned(root: Path, repo: str) -> tuple[bool, str, str]:
    """Clone/fetch/checkout vendor/llama.cpp to pinned SHA. Returns ok, sha, log."""
    sha = _read_pinned_sha(root)
    vendor = root / "vendor" / "llama.cpp"
    vendor.parent.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []

    if not vendor.exists():
        rc, out = _run_live(
            ["git", "clone", repo, str(vendor)],
            root,
            "git clone",
        )
        logs.append(out)
        if not rc:
            return False, sha, "\n".join(logs)

    rc, out = _run_live(
        ["git", "-C", str(vendor), "fetch", "--depth", "1", "origin", sha],
        root,
        "git fetch",
    )
    logs.append(out)
    if not rc:
        rc2, out2 = _run_live(
            ["git", "-C", str(vendor), "fetch", "origin", sha],
            root,
            "git fetch (full)",
        )
        logs.append(out2)
        if not rc2:
            return False, sha, "\n".join(logs)

    rc, out = _run_live(
        ["git", "-C", str(vendor), "checkout", "--detach", sha],
        root,
        "git checkout",
    )
    logs.append(out)
    if not rc:
        return False, sha, "\n".join(logs)

    proc = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    head = (proc.stdout or "").strip() if proc.returncode == 0 else sha
    return True, head, "\n".join(logs)


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local CI pipeline: vendor → gen bindings → build (opt) → validate → test"
    )
    parser.add_argument(
        "--skip-vendor",
        action="store_true",
        help="Do not clone/fetch vendor; use existing vendor/llama.cpp as-is",
    )
    parser.add_argument(
        "--pull-upstream",
        action="store_true",
        help="Run scripts/sync_upstream.py --force first (updates vendor toward upstream master; "
        "then still run pinned checkout unless --skip-vendor)",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip native build_llama_cpp.py")
    parser.add_argument("--skip-wheel", action="store_true", help="Skip python -m build --wheel")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write report JSON to this path",
    )
    parser.add_argument(
        "--repo",
        default="https://github.com/ggml-org/llama.cpp.git",
        help="llama.git URL (default: ggml-org/llama.cpp)",
    )
    args = parser.parse_args()

    root = _project_root()
    py = sys.executable
    print(
        "\nllama-cpp-py-sync — local CI (live logs below, summary at end)\n",
        flush=True,
    )
    started = datetime.now(timezone.utc).isoformat()
    steps: list[StepResult] = []
    pinned_sha = ""
    vendor_head = ""

    try:
        pinned_sha = _read_pinned_sha(root)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        steps.append(StepResult(name="read_pinned_sha", ok=False, detail=str(e)))
        _print_report(started, pinned_sha, vendor_head, steps, False)
        return 2

    if args.pull_upstream:
        rc, out = _run_live([py, str(root / "scripts" / "sync_upstream.py"), "--force"], root, "sync_upstream")
        steps.append(StepResult(name="sync_upstream (--force)", ok=rc, detail=out))

    if not args.skip_vendor:
        ok, head, log = _ensure_vendor_at_pinned(root, args.repo)
        vendor_head = head
        steps.append(StepResult(name="vendor_at_pinned_sha", ok=ok, detail=log))
        if not ok:
            _print_report(started, pinned_sha, vendor_head, steps, False)
            _write_json(args.json, started, pinned_sha, vendor_head, steps, False)
            return 2
    else:
        vendor = root / "vendor" / "llama.cpp"
        if vendor.exists():
            proc = subprocess.run(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            vendor_head = (proc.stdout or "").strip() if proc.returncode == 0 else "(unknown)"
        steps.append(
            StepResult(
                name="vendor_at_pinned_sha",
                ok=True,
                detail=f"skipped (--skip-vendor); HEAD={vendor_head or 'n/a'}",
            )
        )

    ts = _bindings_timestamp(root)
    rc, out = _run_live(
        [
            py,
            str(root / "scripts" / "gen_bindings.py"),
            "--commit-sha",
            vendor_head or pinned_sha,
            "--timestamp",
            ts,
        ],
        root,
        "gen_bindings",
    )
    steps.append(StepResult(name="gen_bindings", ok=rc, detail=out))
    if not rc:
        _print_report(started, pinned_sha, vendor_head, steps, False)
        _write_json(args.json, started, pinned_sha, vendor_head, steps, False)
        return 2

    if not args.skip_build:
        rc, out = _run_live(
            [py, str(root / "scripts" / "build_llama_cpp.py"), "--backend", "auto"],
            root,
            "build_llama_cpp",
        )
        steps.append(StepResult(name="build_llama_cpp", ok=rc, detail=out))
        if not rc:
            _print_report(started, pinned_sha, vendor_head, steps, False)
            _write_json(args.json, started, pinned_sha, vendor_head, steps, False)
            return 2
    else:
        steps.append(StepResult(name="build_llama_cpp", ok=True, detail="skipped (--skip-build)"))

    for script, label in (
        (
            [
                py,
                str(root / "scripts" / "validate_cffi_surface.py"),
                "--check-structs",
                "--check-enums",
                "--check-signatures",
            ],
            "validate_cffi_surface",
        ),
        (
            [
                py,
                str(root / "scripts" / "validate_high_level_api.py"),
                "--module",
                "src/llama_cpp_py_sync/embeddings.py",
            ],
            "validate_high_level_api",
        ),
    ):
        rc, out = _run_live(script, root, label)
        steps.append(StepResult(name=label, ok=rc, detail=out))
        if not rc:
            _print_report(started, pinned_sha, vendor_head, steps, False)
            _write_json(args.json, started, pinned_sha, vendor_head, steps, False)
            return 2

    rc, out = _run_live([py, "-m", "ruff", "check", "."], root, "ruff")
    steps.append(StepResult(name="ruff", ok=rc, detail=out))
    if not rc:
        _print_report(started, pinned_sha, vendor_head, steps, False)
        _write_json(args.json, started, pinned_sha, vendor_head, steps, False)
        return 2

    rc, out = _run_live([py, "-m", "pytest", "tests/", "-q"], root, "pytest")
    steps.append(StepResult(name="pytest", ok=rc, detail=out))
    if not rc:
        _print_report(started, pinned_sha, vendor_head, steps, False)
        _write_json(args.json, started, pinned_sha, vendor_head, steps, False)
        return 2

    if not args.skip_wheel:
        wheel_out = root / "test_temp" / "dist"
        wheel_out.mkdir(parents=True, exist_ok=True)
        rc, out = _run_live(
            [py, "-m", "build", "--wheel", "--outdir", str(wheel_out)],
            root,
            "build_wheel",
        )
        steps.append(StepResult(name="build_wheel", ok=rc, detail=out))
        if not rc:
            _print_report(started, pinned_sha, vendor_head, steps, False)
            _write_json(args.json, started, pinned_sha, vendor_head, steps, False)
            return 2
    else:
        steps.append(StepResult(name="build_wheel", ok=True, detail="skipped (--skip-wheel)"))

    _print_report(started, pinned_sha, vendor_head, steps, True)
    _write_json(args.json, started, pinned_sha, vendor_head, steps, True)
    return 0


def _print_report(
    started: str,
    pinned: str,
    vendor_head: str,
    steps: list[StepResult],
    overall_ok: bool,
) -> None:
    finished = datetime.now(timezone.utc).isoformat()
    print()
    print("=" * 72)
    print("LOCAL CI REPORT")
    print("=" * 72)
    print(f"Started:   {started}")
    print(f"Finished:  {finished}")
    try:
        t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        print(f"Duration:  {t1 - t0}")
    except Exception:
        pass
    print()
    if pinned:
        print(
            f"Pinned SHA (.llama_cpp_sync_state.json): {pinned[:12]}…"
            if len(pinned) > 12
            else f"Pinned SHA: {pinned}"
        )
    if vendor_head:
        print(
            f"Vendor HEAD: {vendor_head[:12]}…"
            if len(vendor_head) > 12
            else f"Vendor HEAD: {vendor_head}"
        )
    print()
    print("Steps (command output was streamed above):")
    print()
    for s in steps:
        status = "OK" if s.ok else "FAIL"
        print(f"  [{status}] {s.name}")
        if s.detail and not s.ok:
            print("       --- tail ---")
            for line in s.detail.strip().splitlines()[-24:]:
                print(f"       {line}")
        elif s.detail and s.ok and "skipped" in s.detail.lower():
            print(f"       {s.detail}")
    print()
    print("Overall:", "SUCCESS" if overall_ok else "FAILURE")
    print("=" * 72)
    print()
    if not overall_ok:
        print(
            "Note: If validate_cffi_surface failed, ensure vendor/llama.cpp matches the pinned SHA "
            "(re-run without --skip-vendor). CI regenerates bindings from that pin."
        )


def _write_json(
    path: Path | None,
    started: str,
    pinned: str,
    vendor_head: str,
    steps: list[StepResult],
    overall_ok: bool,
) -> None:
    if path is None:
        return
    finished = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "started_utc": started,
        "finished_utc": finished,
        "pinned_sha": pinned,
        "vendor_head": vendor_head,
        "overall_ok": overall_ok,
        "steps": [asdict(s) for s in steps],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JSON report: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
