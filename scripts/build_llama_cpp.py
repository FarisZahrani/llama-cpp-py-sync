#!/usr/bin/env python3
"""
Build llama.cpp shared library.

This script compiles llama.cpp into a shared library that can be used
by the Python bindings. It supports multiple backends including CPU,
CUDA, ROCm, Vulkan, Metal, and various BLAS implementations.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _copy_runtime_dll(src: Path, dst_dir: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception:
        return False
    return True


def _copy_msvc_openmp_runtimes(package_dir: Path) -> int:
    candidates: List[Path] = []

    vs_roots: List[Path] = []
    for env_key in ["VCToolsInstallDir", "VCINSTALLDIR", "VSINSTALLDIR", "VSCMD_ARG_VCVARS"]:
        val = os.environ.get(env_key)
        if val:
            vs_roots.append(Path(val))

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")
    try:
        vs_roots.extend(
            list(
                Path(program_files_x86).glob(
                    "Microsoft Visual Studio/*/*/VC/Redist/MSVC/*/x64/*"
                )
            )
        )
    except Exception:
        pass

    vc_patterns = [
        "Microsoft.VC*CRT/vcruntime140*.dll",
        "Microsoft.VC*CRT/msvcp140*.dll",
        "Microsoft.VC*CRT/concrt140*.dll",
        "Microsoft.VC*OpenMP/vcomp140*.dll",
    ]

    for root in vs_roots:
        try:
            if root.is_dir():
                for pattern in vc_patterns:
                    for p in root.glob(pattern):
                        if p.is_file():
                            candidates.append(p)
        except Exception:
            pass

    system32 = Path(r"C:\\Windows\\System32")
    for p in [
        system32 / "vcruntime140.dll",
        system32 / "vcruntime140_1.dll",
        system32 / "msvcp140.dll",
        system32 / "msvcp140_1.dll",
        system32 / "msvcp140_atomic_wait.dll",
        system32 / "concrt140.dll",
        system32 / "vcomp140.dll",
    ]:
        if p.exists():
            candidates.append(p)

    copied = 0
    seen = set()
    for src in candidates:
        key = src.name.lower()
        if key in seen:
            continue
        seen.add(key)
        if _copy_runtime_dll(src, package_dir):
            copied += 1
    return copied


def _copy_vulkan_runtime_dlls(package_dir: Path) -> int:
    copied = 0
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    vulkan_bin_dir = Path(vulkan_sdk) / "Bin" if vulkan_sdk else None

    if vulkan_bin_dir is not None and vulkan_bin_dir.exists():
        src = vulkan_bin_dir / "vulkan-1.dll"
        if _copy_runtime_dll(src, package_dir):
            copied += 1

    if not (package_dir / "vulkan-1.dll").exists():
        for sys_path in [
            Path(r"C:\\Windows\\System32\\vulkan-1.dll"),
            Path(r"C:\\Windows\\SysWOW64\\vulkan-1.dll"),
        ]:
            if _copy_runtime_dll(sys_path, package_dir):
                copied += 1
                break

    return copied


def _copy_cuda_runtime_dlls(package_dir: Path) -> int:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        return 0
    cuda_bin = Path(cuda_home) / "bin"
    if not cuda_bin.exists():
        return 0

    required = [
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "cusparse64_*.dll",
        "cusolver64_*.dll",
        "curand64_*.dll",
    ]

    copied = 0
    seen = set()
    for pattern in required:
        for src in cuda_bin.glob(pattern):
            key = src.name.lower()
            if key in seen:
                continue
            seen.add(key)
            if _copy_runtime_dll(src, package_dir):
                copied += 1
    return copied


def _bundle_windows_runtime_dlls(
    package_dir: Path,
    backends: Dict[str, Tuple[bool, Optional[str]]],
    enable_cuda: bool,
    enable_vulkan: bool,
) -> None:
    msvc_count = _copy_msvc_openmp_runtimes(package_dir)
    cuda_count = 0
    vulkan_count = 0

    if enable_cuda and backends.get("cuda", (False, None))[0]:
        cuda_count = _copy_cuda_runtime_dlls(package_dir)

    if enable_vulkan and backends.get("vulkan", (False, None))[0]:
        vulkan_count = _copy_vulkan_runtime_dlls(package_dir)

    if msvc_count == 0:
        print(
            "Warning: No MSVC/OpenMP runtime DLLs were bundled. If you see Windows error 0x7e on a clean machine, install VC++ 2015-2022 x64 redistributable."
        )
    if enable_cuda and backends.get("cuda", (False, None))[0] and cuda_count == 0:
        print(
            "Warning: CUDA backend was enabled but no CUDA runtime DLLs were bundled. If you see Windows error 0x7e on a clean machine, ensure CUDA runtime DLLs are present or set CUDA_PATH when building."
        )

    if enable_vulkan and backends.get("vulkan", (False, None))[0] and vulkan_count == 0:
        print(
            "Warning: Vulkan backend was enabled but vulkan-1.dll could not be bundled. Ensure Vulkan is installed (GPU drivers) or set VULKAN_SDK when building."
        )


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _ensure_vendor_llama_cpp(project_root: Path, vendor_path: Path) -> None:
    if vendor_path.exists():
        return

    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            f"Vendor directory not found: {vendor_path}. "
            "Git is required to auto-fetch llama.cpp (install git or provide --vendor-path)."
        )

    vendor_path.parent.mkdir(parents=True, exist_ok=True)

    # Otherwise clone from upstream.
    repo_url = os.environ.get("LLAMA_CPP_VENDOR_REPO", "https://github.com/ggml-org/llama.cpp")
    cmd = [git, "clone", "--depth", "1", repo_url, str(vendor_path)]
    print(f"Vendor llama.cpp missing; cloning: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(project_root))
    if res.returncode != 0 or not vendor_path.exists():
        raise RuntimeError(
            f"Failed to clone llama.cpp into {vendor_path}. "
            "You can clone it manually or set LLAMA_CPP_VENDOR_REPO / use --vendor-path."
        )


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def _run_and_capture_env(cmd: list[str], cwd: Path | None = None) -> dict[str, str]:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg = stderr or stdout or f"Command failed: {' '.join(cmd)}"
        raise RuntimeError(msg)

    env: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k] = v
    return env


def _try_import_windows_toolchain_env(project_root: Path) -> bool:
    if not _is_windows():
        return True

    setup_script = project_root / "scripts" / "setup_windows_toolchain.ps1"
    if not setup_script.exists():
        return False

    # Run the toolchain script in a child PowerShell, then print the resulting environment
    # so we can import it into this Python process.
    ps_cmd = (
        "& { "
        f". '{str(setup_script)}' -Quiet; "
        "Get-ChildItem Env:* | ForEach-Object { \"$($_.Name)=$($_.Value)\" } "
        "}"
    )

    try:
        env = _run_and_capture_env(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_cmd,
            ],
            cwd=project_root,
        )
    except Exception:
        return False

    for k, v in env.items():
        os.environ[k] = v

    return shutil.which("cl") is not None


def _vswhere_path() -> Optional[Path]:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", ""))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe",
        Path(os.environ.get("ProgramFiles", ""))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _prepend_path(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    p = str(dir_path)
    cur = os.environ.get("PATH", "")
    if cur.lower().startswith(p.lower() + os.pathsep):
        return
    os.environ["PATH"] = p + os.pathsep + cur


def _try_add_vs_cmake_ninja(install_path: str) -> None:
    # Visual Studio bundles CMake and Ninja under the IDE directory.
    # This keeps setup minimal for Windows users.
    root = Path(install_path)
    cmake_bin = root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin"
    ninja_bin = root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja"
    _prepend_path(cmake_bin)
    _prepend_path(ninja_bin)


def _try_load_msvc_env(project_root: Optional[Path] = None) -> bool:
    if not _is_windows():
        return True

    if shutil.which("cl") is not None:
        return True

    root = project_root or get_project_root()
    if _try_import_windows_toolchain_env(root):
        return True

    vswhere = _vswhere_path()
    if vswhere is None:
        return False

    try:
        install_path = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
    except Exception:
        return False

    if not install_path:
        return False

    _try_add_vs_cmake_ninja(install_path)

    vcvars64 = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars64.exists():
        return False

    try:
        env = _run_and_capture_env(["cmd", "/c", f'"{vcvars64}" && set'])
    except Exception:
        return False

    for k, v in env.items():
        os.environ[k] = v

    return shutil.which("cl") is not None


def _cmake_generator() -> Optional[str]:
    if not _is_windows():
        return "Ninja"

    if shutil.which("ninja") is not None:
        return "Ninja"
    if shutil.which("nmake") is not None:
        return "NMake Makefiles"
    if shutil.which("cl") is not None:
        return "NMake Makefiles"
    return None


def _require_build_tools() -> None:
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "CMake was not found in PATH. Install CMake and ensure `cmake` is available, "
            "or install a prebuilt wheel that bundles the llama shared library."
        )

    if _is_windows():
        if not _try_load_msvc_env(get_project_root()):
            raise RuntimeError(
                "No usable C/C++ toolchain was detected. Install 'Visual Studio Build Tools' (MSVC) with the C++ workload, "
                "or run scripts/setup_windows_toolchain.ps1, then retry."
            )
        gen = _cmake_generator()
        if gen is None:
            raise RuntimeError(
                "No usable C/C++ build toolchain was detected. "
                "Install 'Visual Studio Build Tools' (MSVC) or add a supported generator to PATH (Ninja/NMake)."
            )
    else:
        if shutil.which("ninja") is None:
            raise RuntimeError(
                "Ninja was not found in PATH. Install Ninja and ensure `ninja` is available. "
                "(This project uses Ninja for builds.)"
            )


def detect_cuda() -> Tuple[bool, Optional[str]]:
    """Detect if CUDA is available and return version."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if cuda_home and Path(cuda_home).exists():
        nvcc_path = Path(cuda_home) / "bin" / ("nvcc.exe" if platform.system() == "Windows" else "nvcc")
        if nvcc_path.exists():
            try:
                result = subprocess.run([str(nvcc_path), "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "release" in line.lower():
                            return True, line.strip()
            except Exception:
                pass
            return True, "unknown version"

    for path in ["/usr/local/cuda", "/usr/cuda", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"]:
        if Path(path).exists():
            return True, "detected"

    return False, None


def detect_rocm() -> Tuple[bool, Optional[str]]:
    """Detect if ROCm is available."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")

    if Path(rocm_path).exists():
        hipcc_path = Path(rocm_path) / "bin" / "hipcc"
        if hipcc_path.exists():
            try:
                result = subprocess.run([str(hipcc_path), "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return True, result.stdout.split("\n")[0]
            except Exception:
                pass
            return True, "detected"

    return False, None


def detect_vulkan() -> Tuple[bool, Optional[str]]:
    """Detect if Vulkan SDK is available."""
    vulkan_sdk = os.environ.get("VULKAN_SDK")

    if vulkan_sdk and Path(vulkan_sdk).exists():
        return True, vulkan_sdk

    if platform.system() == "Linux":
        if Path("/usr/include/vulkan/vulkan.h").exists():
            return True, "system"

    return False, None


def detect_metal() -> Tuple[bool, Optional[str]]:
    """Detect if Metal is available (macOS only)."""
    if platform.system() != "Darwin":
        return False, None

    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
    except Exception:
        pass

    return False, None


def detect_blas() -> Tuple[bool, str]:
    """Detect available BLAS implementation."""
    if platform.system() == "Darwin":
        return True, "accelerate"

    openblas_paths = [
        "/usr/include/openblas",
        "/usr/local/include/openblas",
        "/opt/OpenBLAS/include",
    ]
    for path in openblas_paths:
        if Path(path).exists():
            return True, "openblas"

    mkl_root = os.environ.get("MKLROOT")
    if mkl_root and Path(mkl_root).exists():
        return True, "mkl"

    return False, "none"


def detect_backends() -> Dict[str, Tuple[bool, Optional[str]]]:
    """Detect all available backends."""
    return {
        "cuda": detect_cuda(),
        "rocm": detect_rocm(),
        "vulkan": detect_vulkan(),
        "metal": detect_metal(),
        "blas": detect_blas(),
    }


def get_cmake_args(
    backends: Dict[str, Tuple[bool, Optional[str]]],
    enable_cuda: bool = True,
    enable_rocm: bool = True,
    enable_vulkan: bool = True,
    enable_metal: bool = True,
    enable_blas: bool = True,
) -> List[str]:
    """Get CMake configuration arguments based on detected backends."""
    args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
    ]

    # When producing distributable wheels in CI, never compile with -march=native
    # (GGML_NATIVE=ON). GitHub runners may support instructions (e.g. AVX512) that
    # are not available on end-user CPUs, leading to runtime "Illegal instruction"
    # crashes.
    if os.environ.get("GITHUB_ACTIONS") == "true" and not any(
        a.startswith("-DGGML_NATIVE=") for a in args
    ):
        args.append("-DGGML_NATIVE=OFF")

    if enable_cuda and backends["cuda"][0]:
        args.append("-DGGML_CUDA=ON")
        if not any(a.startswith("-DCMAKE_CUDA_ARCHITECTURES=") for a in args):
            cuda_archs = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
            if not cuda_archs:
                cuda_archs = "75;80;86"
            args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}")
        print(f"  CUDA: enabled ({backends['cuda'][1]})")
    else:
        args.append("-DGGML_CUDA=OFF")

    if enable_rocm and backends["rocm"][0]:
        args.append("-DGGML_HIP=ON")
        print(f"  ROCm: enabled ({backends['rocm'][1]})")
    else:
        args.append("-DGGML_HIP=OFF")

    if enable_vulkan and backends["vulkan"][0]:
        args.append("-DGGML_VULKAN=ON")
        print(f"  Vulkan: enabled ({backends['vulkan'][1]})")
    else:
        args.append("-DGGML_VULKAN=OFF")

    if enable_metal and backends["metal"][0]:
        args.append("-DGGML_METAL=ON")
        print(f"  Metal: enabled ({backends['metal'][1]})")
    else:
        args.append("-DGGML_METAL=OFF")

    if enable_blas and backends["blas"][0]:
        blas_type = backends["blas"][1]
        if blas_type == "accelerate":
            args.append("-DGGML_ACCELERATE=ON")
        elif blas_type == "openblas":
            args.append("-DGGML_BLAS=ON")
            args.append("-DGGML_BLAS_VENDOR=OpenBLAS")
        elif blas_type == "mkl":
            args.append("-DGGML_BLAS=ON")
            args.append("-DGGML_BLAS_VENDOR=Intel10_64lp")
        print(f"  BLAS: enabled ({blas_type})")
    else:
        args.append("-DGGML_BLAS=OFF")

    return args


def run_cmake_configure(
    source_dir: Path,
    build_dir: Path,
    cmake_args: List[str],
) -> bool:
    """Run CMake configuration."""
    _require_build_tools()
    build_dir.mkdir(parents=True, exist_ok=True)

    gen = _cmake_generator()
    if gen is None:
        raise RuntimeError("Unable to select a CMake generator for this platform.")
    cmd = ["cmake", "-G", gen, str(source_dir)] + cmake_args

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=build_dir)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to run CMake. Ensure CMake is installed and available on PATH."
        ) from e
    return result.returncode == 0


def run_cmake_build(build_dir: Path, parallel: int = 0, target: Optional[str] = None) -> bool:
    """Run CMake build."""
    _require_build_tools()
    cmd = ["cmake", "--build", str(build_dir)]

    if target:
        cmd.extend(["--target", target])

    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])
    else:
        cmd.extend(["--parallel"])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to run CMake build. Ensure CMake is installed and available on PATH."
        ) from e
    return result.returncode == 0


def find_built_library(build_dir: Path) -> Optional[Path]:
    """Find the built shared library."""
    system = platform.system().lower()

    if system == "windows":
        patterns = [
            "**/libllama.dll",
            "**/Release/libllama.dll",
            "**/bin/libllama.dll",
            "**/llama.dll",
            "**/Release/llama.dll",
            "**/bin/llama.dll",
        ]
    elif system == "darwin":
        patterns = ["**/libllama.dylib", "**/lib/libllama.dylib"]
    else:
        patterns = ["**/libllama.so", "**/lib/libllama.so"]

    for pattern in patterns:
        matches = list(build_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def _copy_windows_dependency_dlls(lib_path: Path, package_dir: Path) -> None:
    lib_dir = lib_path.parent
    patterns = ["ggml*.dll"]

    copied_any = False
    for pattern in patterns:
        for dep_path in lib_dir.glob(pattern):
            if dep_path.name.lower() == lib_path.name.lower():
                continue

            dest_path = package_dir / dep_path.name
            shutil.copy2(dep_path, dest_path)
            copied_any = True
            print(f"Copied {dep_path} to {dest_path}")

    if not copied_any:
        print(
            "Note: no ggml*.dll dependencies were found next to the built llama DLL. "
            "If you still see Windows error 0x7e at runtime, the missing dependency is likely a system/runtime DLL."
        )


def copy_library_to_package(lib_path: Path, package_dir: Path) -> Path:
    """Copy the built library to the package directory."""
    package_dir.mkdir(parents=True, exist_ok=True)

    dest_path = package_dir / lib_path.name
    shutil.copy2(lib_path, dest_path)

    print(f"Copied {lib_path} to {dest_path}")

    if platform.system().lower() == "windows" and lib_path.suffix.lower() == ".dll":
        _copy_windows_dependency_dlls(lib_path, package_dir)

    return dest_path


def build_llama_cpp(
    vendor_path: Path,
    output_dir: Path,
    enable_cuda: bool = True,
    enable_rocm: bool = True,
    enable_vulkan: bool = True,
    enable_metal: bool = True,
    enable_blas: bool = True,
    parallel: int = 0,
    clean: bool = False,
    fetch_vendor: bool = True,
    bundle_runtime_dlls: bool = True,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Build llama.cpp and return path to the built library.

    Args:
        vendor_path: Path to the llama.cpp source directory.
        output_dir: Directory to place the built library.
        enable_*: Enable specific backends if available.
        parallel: Number of parallel build jobs (0 for auto).
        clean: Clean build directory before building.

    Returns:
        Path to the built library, or None if build failed.
    """
    if not vendor_path.exists():
        if fetch_vendor:
            try:
                _ensure_vendor_llama_cpp(project_root or get_project_root(), vendor_path)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                return None
        else:
            print(f"Error: Vendor directory not found: {vendor_path}", file=sys.stderr)
            return None

    build_dir = vendor_path / "build"

    if clean and build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

    print("Detecting available backends...")
    backends = detect_backends()

    print("\nConfiguring build...")
    cmake_args = get_cmake_args(
        backends,
        enable_cuda=enable_cuda,
        enable_rocm=enable_rocm,
        enable_vulkan=enable_vulkan,
        enable_metal=enable_metal,
        enable_blas=enable_blas,
    )

    if not run_cmake_configure(vendor_path, build_dir, cmake_args):
        print("Error: CMake configuration failed", file=sys.stderr)
        return None

    print("\nBuilding...")
    if not run_cmake_build(build_dir, parallel=parallel, target="llama"):
        print("Error: Build failed", file=sys.stderr)
        return None

    print("\nLocating built library...")
    lib_path = find_built_library(build_dir)

    if lib_path is None:
        print("Error: Could not find built library", file=sys.stderr)
        return None

    print(f"Found library: {lib_path}")

    dest_path = copy_library_to_package(lib_path, output_dir)

    if _is_windows() and bundle_runtime_dlls:
        _bundle_windows_runtime_dlls(
            output_dir,
            backends,
            enable_cuda=enable_cuda,
            enable_vulkan=enable_vulkan,
        )

    return dest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build llama.cpp shared library"
    )
    parser.add_argument(
        "--vendor-path",
        type=Path,
        default=None,
        help="Path to vendor/llama.cpp directory"
    )
    parser.add_argument(
        "--no-fetch-vendor",
        action="store_true",
        help="Do not auto-fetch vendor/llama.cpp when missing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for built library"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory"
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "cuda", "vulkan", "rocm", "metal"],
        default="auto",
        help=(
            "Select a single backend to build (default: auto). "
            "This is a convenience flag that sets the --no-* toggles for you; "
            "explicit --no-* flags still override."
        ),
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--no-rocm",
        action="store_true",
        help="Disable ROCm even if available"
    )
    parser.add_argument(
        "--no-vulkan",
        action="store_true",
        help="Disable Vulkan even if available"
    )
    parser.add_argument(
        "--no-metal",
        action="store_true",
        help="Disable Metal even if available"
    )
    parser.add_argument(
        "--no-blas",
        action="store_true",
        help="Disable BLAS even if available"
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=0,
        help="Number of parallel build jobs (0 for auto)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect backends, don't build"
    )

    parser.add_argument(
        "--no-bundle-runtime-dlls",
        action="store_true",
        help="Do not bundle Windows runtime DLLs (CUDA/MSVC/OpenMP) next to the built library",
    )

    args = parser.parse_args()

    project_root = args.project_root or get_project_root()
    vendor_path = args.vendor_path or (project_root / "vendor" / "llama.cpp")
    output_dir = args.output_dir or (project_root / "src" / "llama_cpp_py_sync")

    if args.detect_only:
        print("Detecting available backends...")
        backends = detect_backends()
        print("\nBackend Detection Results:")
        print(f"  CUDA:   {'✓' if backends['cuda'][0] else '✗'} {backends['cuda'][1] or ''}")
        print(f"  ROCm:   {'✓' if backends['rocm'][0] else '✗'} {backends['rocm'][1] or ''}")
        print(f"  Vulkan: {'✓' if backends['vulkan'][0] else '✗'} {backends['vulkan'][1] or ''}")
        print(f"  Metal:  {'✓' if backends['metal'][0] else '✗'} {backends['metal'][1] or ''}")
        print(f"  BLAS:   {'✓' if backends['blas'][0] else '✗'} {backends['blas'][1] or ''}")
        return

    # Backend selection convenience. Defaults to existing behavior (auto).
    enable_cuda = not args.no_cuda
    enable_rocm = not args.no_rocm
    enable_vulkan = not args.no_vulkan
    enable_metal = not args.no_metal
    enable_blas = not args.no_blas

    if args.backend != "auto":
        enable_cuda = args.backend == "cuda"
        enable_rocm = args.backend == "rocm"
        enable_vulkan = args.backend == "vulkan"
        enable_metal = args.backend == "metal"
        # For a predictable single-backend build, keep BLAS off by default.
        enable_blas = False

    # Allow explicit no-* flags to override backend convenience.
    if args.no_cuda:
        enable_cuda = False
    if args.no_rocm:
        enable_rocm = False
    if args.no_vulkan:
        enable_vulkan = False
    if args.no_metal:
        enable_metal = False
    if args.no_blas:
        enable_blas = False

    lib_path = build_llama_cpp(
        vendor_path=vendor_path,
        output_dir=output_dir,
        enable_cuda=enable_cuda,
        enable_rocm=enable_rocm,
        enable_vulkan=enable_vulkan,
        enable_metal=enable_metal,
        enable_blas=enable_blas,
        parallel=args.parallel,
        clean=args.clean,
        fetch_vendor=not args.no_fetch_vendor,
        bundle_runtime_dlls=(not args.no_bundle_runtime_dlls) if _is_windows() else False,
        project_root=project_root,
    )

    if lib_path:
        print("\nBuild successful!")
        print(f"Library: {lib_path}")
        sys.exit(0)
    else:
        print("\nBuild failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
