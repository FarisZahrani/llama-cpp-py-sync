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


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def _require_build_tools() -> None:
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "CMake was not found in PATH. Install CMake and ensure `cmake` is available, "
            "or install a prebuilt wheel that bundles the llama shared library."
        )

    if platform.system().lower() == "windows":
        # CMake on Windows typically needs a compiler toolchain available.
        # We can't reliably validate every setup, but we can catch the common missing-tool case.
        if shutil.which("cl") is None and shutil.which("ninja") is None and shutil.which("mingw32-make") is None:
            raise RuntimeError(
                "No C/C++ build toolchain was detected (missing `cl`, `ninja`, and `mingw32-make`). "
                "Install 'Visual Studio Build Tools' (MSVC) or Ninja and try again."
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
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
    ]

    if enable_cuda and backends["cuda"][0]:
        args.append("-DGGML_CUDA=ON")
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

    cmd = ["cmake", str(source_dir)] + cmake_args

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
    cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]

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
        patterns = ["**/llama.dll", "**/Release/llama.dll", "**/bin/llama.dll"]
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
            "Note: no ggml*.dll dependencies were found next to the built llama.dll. "
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

    lib_path = build_llama_cpp(
        vendor_path=vendor_path,
        output_dir=output_dir,
        enable_cuda=not args.no_cuda,
        enable_rocm=not args.no_rocm,
        enable_vulkan=not args.no_vulkan,
        enable_metal=not args.no_metal,
        enable_blas=not args.no_blas,
        parallel=args.parallel,
        clean=args.clean,
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
