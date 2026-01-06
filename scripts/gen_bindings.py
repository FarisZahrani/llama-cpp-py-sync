#!/usr/bin/env python3
"""
Generate CFFI bindings from llama.cpp header files.

This script reads the llama.h header file and generates Python CFFI bindings
automatically. It parses the C declarations and creates a _cffi_bindings.py
file that can be used to interface with the llama.cpp shared library.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def find_header_files(vendor_path: Path) -> dict:
    """Find relevant header files in the vendor directory."""
    headers = {}

    include_path = vendor_path / "include"
    if include_path.exists():
        llama_h = include_path / "llama.h"
        if llama_h.exists():
            headers["llama.h"] = llama_h

    llama_h_root = vendor_path / "llama.h"
    if llama_h_root.exists() and "llama.h" not in headers:
        headers["llama.h"] = llama_h_root

    ggml_h = vendor_path / "ggml" / "include" / "ggml.h"
    if ggml_h.exists():
        headers["ggml.h"] = ggml_h

    return headers


def preprocess_header(content: str) -> str:
    """
    Preprocess the header content to make it CFFI-compatible.

    This removes compiler-specific attributes, macros, and other
    elements that CFFI cannot parse directly.
    """
    content = re.sub(r'#include\s*[<"][^>"]+[>"]', '', content)

    content = re.sub(r'#\s*if.*?#\s*endif', '', content, flags=re.DOTALL)
    content = re.sub(r'#\s*ifdef.*?#\s*endif', '', content, flags=re.DOTALL)
    content = re.sub(r'#\s*ifndef.*?#\s*endif', '', content, flags=re.DOTALL)
    content = re.sub(r'#\s*define\s+\w+\s*\n', '', content)
    content = re.sub(r'#\s*define\s+\w+\([^)]*\)[^\n]*\n', '', content)
    content = re.sub(r'#\s*undef\s+\w+', '', content)
    content = re.sub(r'#\s*pragma[^\n]*\n', '', content)
    content = re.sub(r'#\s*error[^\n]*\n', '', content)
    content = re.sub(r'#\s*warning[^\n]*\n', '', content)

    content = re.sub(r'LLAMA_API\s+', '', content)
    content = re.sub(r'GGML_API\s+', '', content)
    content = re.sub(r'__attribute__\s*\(\([^)]*\)\)', '', content)
    content = re.sub(r'__declspec\s*\([^)]*\)', '', content)
    content = re.sub(r'GGML_CALL\s*', '', content)
    content = re.sub(r'GGML_RESTRICT\s*', '', content)
    content = re.sub(r'LLAMA_DEPRECATED\s*', '', content)
    content = re.sub(r'GGML_DEPRECATED\s*', '', content)

    # ggml types referenced in llama.h but defined in other ggml headers. For CFFI ABI
    # bindings we don't need their concrete definitions; treat them as opaque/primitive.
    content = re.sub(r'\benum\s+ggml_numa_strategy\b', 'int', content)
    content = re.sub(r'\benum\s+ggml_opt_optimizer_type\b', 'int', content)
    content = re.sub(r'\bggml_opt_get_optimizer_params\b', 'void *', content)

    content = re.sub(r'\bextern\s+"C"\s*\{', '', content)
    content = re.sub(r'\}\s*//\s*extern\s+"C"', '', content)

    content = re.sub(r'//[^\n]*\n', '\n', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

    return content


def extract_enums(content: str) -> List[str]:
    """Extract enum definitions from header content."""
    enums = []

    enum_pattern = r'enum\s+(\w+)\s*\{([^}]+)\}'

    for match in re.finditer(enum_pattern, content):
        enum_name = match.group(1)
        enum_body = match.group(2)

        enum_def = f"enum {enum_name} {{\n"

        members = []
        for line in enum_body.split('\n'):
            line = line.strip()
            if line and not line.startswith('//'):
                line = re.sub(r'//.*$', '', line).strip()
                if line:
                    # CFFI only supports simple numeric constants in enums. Upstream headers
                    # sometimes assign enum values from other macros (e.g. GGML_*). For the
                    # Python bindings, we only need the enum names, so strip initializers.
                    line = re.sub(r'\s*=\s*[^,}]+', '', line).strip()
                    members.append(f"    {line}")

        enum_def += '\n'.join(members)
        enum_def += "\n};"
        enums.append(enum_def)

    return enums


def extract_structs(content: str) -> List[str]:
    """Extract struct definitions from header content."""
    structs: List[str] = []

    i = 0
    needle = "typedef struct"
    while True:
        start = content.find(needle, i)
        if start < 0:
            break

        brace_open = content.find("{", start)
        if brace_open < 0:
            break

        depth = 0
        j = brace_open
        while j < len(content):
            ch = content[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if j >= len(content):
            break

        brace_close = j
        semi = content.find(";", brace_close)
        if semi < 0:
            break

        after = content[brace_close + 1 : semi]
        m_name = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*$", after.strip())
        if not m_name:
            i = semi + 1
            continue
        struct_name = m_name.group(1)

        struct_body = content[brace_open + 1 : brace_close]
        struct_def = f"typedef struct {struct_name} {{\n"
        for line in struct_body.split("\n"):
            line = line.strip()
            if line and not line.startswith("//"):
                line = re.sub(r"//.*$", "", line).strip()
                if line:
                    struct_def += f"    {line}\n"
        struct_def += f"}} {struct_name};"
        structs.append(struct_def)

        i = semi + 1

    return structs


def extract_named_struct_decls(content: str) -> List[str]:
    """Extract `struct name { ... };` declarations from header content."""
    structs: List[str] = []

    for m in re.finditer(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", content):
        struct_name = m.group(1)
        brace_open = content.find("{", m.start())
        if brace_open < 0:
            continue

        depth = 0
        j = brace_open
        while j < len(content):
            ch = content[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if j >= len(content):
            continue

        brace_close = j
        tail = content[brace_close : brace_close + 4]
        if ";" not in tail:
            # require closing '};' right after the block (allow whitespace/newlines)
            post = content[brace_close : brace_close + 50]
            if not re.match(r"\}\s*;", post):
                continue

        struct_body = content[brace_open + 1 : brace_close]
        struct_def = f"struct {struct_name} {{\n"
        for line in struct_body.split("\n"):
            line = line.strip()
            if line and not line.startswith("//"):
                line = re.sub(r"//.*$", "", line).strip()
                if line:
                    struct_def += f"    {line}\n"
        struct_def += "};"
        structs.append(struct_def)

    return structs


def extract_typedefs(content: str) -> List[str]:
    """Extract non-struct typedef statements from header content."""
    typedefs: List[str] = []

    typedef_pattern = r"^\s*typedef\s+(?!struct\b)[^;]+;\s*$"
    for match in re.finditer(typedef_pattern, content, re.MULTILINE):
        typedefs.append(match.group(0).strip())

    return typedefs


def _typedef_name(typedef_stmt: str) -> Optional[str]:
    """Best-effort extraction of the declared typedef name."""
    m = re.match(r"^typedef\s+.+?\b([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$", typedef_stmt.strip())
    if not m:
        return None
    return m.group(1)


def extract_functions(content: str) -> List[str]:
    """Extract function declarations from header content."""
    functions = []

    func_pattern = r'^[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*;'

    for match in re.finditer(func_pattern, content, re.MULTILINE):
        func_decl = match.group(0).strip()
        if not func_decl.startswith('typedef'):
            functions.append(func_decl)

    return functions


def generate_cdef(headers: dict) -> str:
    """
    Generate CFFI cdef string from header files.

    This is a simplified parser - for production use, consider using
    pycparser or a more robust C parser.
    """
    cdef_parts = []

    prelude_typedef_names = {
        "llama_pos",
        "llama_token",
        "llama_seq_id",
        "llama_memory_t",
        "ggml_threadpool_t",
        "ggml_backend_dev_t",
        "ggml_backend_buffer_type_t",
        "ggml_backend_sched_eval_callback",
        "ggml_abort_callback",
        "ggml_log_callback",
        "ggml_opt_dataset_t",
        "ggml_opt_result_t",
        "ggml_opt_epoch_callback",
    }

    cdef_parts.append("""
// Basic types
typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;

// Opaque structs
typedef struct llama_model llama_model;
typedef struct llama_vocab llama_vocab;
typedef struct llama_context llama_context;
typedef struct llama_sampler llama_sampler;

// Opaque handles
struct llama_memory_i;
typedef struct llama_memory_i * llama_memory_t;

// Opaque ggml types referenced by the llama public API
typedef void * ggml_threadpool_t;
typedef void * ggml_backend_dev_t;
typedef void * ggml_backend_buffer_type_t;
typedef void * ggml_backend_sched_eval_callback;
typedef void * ggml_abort_callback;
typedef void * ggml_log_callback;
typedef void * ggml_opt_dataset_t;
typedef void * ggml_opt_result_t;
typedef void * ggml_opt_epoch_callback;
""")

    if "llama.h" in headers:
        with open(headers["llama.h"], encoding="utf-8", errors="ignore") as f:
            content = f.read()

        content = preprocess_header(content)

        typedefs = extract_typedefs(content)
        if typedefs:
            seen_typedef_names: set[str] = set()
            filtered_typedefs: list[str] = []
            for td in typedefs:
                name = _typedef_name(td)
                if name is None:
                    filtered_typedefs.append(td)
                    continue

                if name in prelude_typedef_names:
                    continue

                if name in seen_typedef_names:
                    continue
                seen_typedef_names.add(name)
                filtered_typedefs.append(td)

            if filtered_typedefs:
                cdef_parts.append("\n".join(filtered_typedefs))

        enums = extract_enums(content)
        if enums:
            cdef_parts.append("\n\n".join(enums))

        structs = extract_structs(content)
        if structs:
            cdef_parts.append("\n\n".join(structs))

        named_structs = extract_named_struct_decls(content)
        if named_structs:
            cdef_parts.append("\n\n".join(named_structs))

        functions = extract_functions(content)
        if functions:
            cdef_parts.append("\n".join(functions))

    return "\n".join(cdef_parts)


def generate_bindings_file(
    project_root: Path,
    vendor_path: Path,
    output_path: Path,
    commit_sha: Optional[str] = None
):
    """Generate the _cffi_bindings.py file."""
    headers = find_header_files(vendor_path)

    cdef_text = generate_cdef(headers) if headers else ""

    if not headers:
        print("Warning: No header files found in vendor directory")
        print("Using default bindings template")

    template = '''"""
CFFI ABI bindings for llama.cpp

This module is auto-generated by scripts/gen_bindings.py
DO NOT EDIT MANUALLY - changes will be overwritten on next sync.

Generated: {timestamp}
llama.cpp commit: {commit_sha}
"""

import os
import platform
from pathlib import Path

from cffi import FFI

ffi = FFI()

_LLAMA_H_CDEF = """
{cdef_text}
"""

ffi.cdef(_LLAMA_H_CDEF)


def _find_library():
    """Find the llama shared library."""
    lib_dir = Path(__file__).parent

    system = platform.system().lower()

    if system == "windows":
        lib_names = ["llama.dll", "libllama.dll"]
    elif system == "darwin":
        lib_names = ["libllama.dylib", "libllama.so"]
    else:
        lib_names = ["libllama.so"]

    for lib_name in lib_names:
        lib_path = lib_dir / lib_name
        if lib_path.exists():
            return str(lib_path)

    env_path = os.environ.get("LLAMA_CPP_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    search_paths = []
    if system == "linux":
        search_paths = [
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
        ]
    elif system == "darwin":
        search_paths = [
            "/usr/local/lib",
            "/opt/homebrew/lib",
        ]
    elif system == "windows":
        search_paths = [
            os.path.join(os.environ.get("ProgramFiles", ""), "llama.cpp", "bin"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "llama.cpp", "bin"),
        ]

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = os.path.join(search_path, lib_name)
            if os.path.exists(lib_path):
                return lib_path

    return None


def _load_library():
    """Load the llama shared library."""
    lib_path = _find_library()

    if lib_path is None:
        raise RuntimeError(
            "Could not find llama.cpp shared library. "
            "Please ensure the library is installed or set LLAMA_CPP_LIB environment variable."
        )

    try:
        if platform.system().lower() == "windows":
            lib_dir = str(Path(lib_path).parent)
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(lib_dir)
            except Exception:
                pass
            try:
                os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass

        return ffi.dlopen(lib_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load llama.cpp library from {{lib_path}}: {{e}}") from e


_lib = None


def get_lib():
    """Get the loaded llama library, loading it if necessary."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def get_ffi():
    """Get the CFFI FFI instance."""
    return ffi
'''

    timestamp = datetime.utcnow().isoformat()
    commit_sha = commit_sha or "unknown"

    output_content = template.format(
        timestamp=timestamp,
        commit_sha=commit_sha,
        cdef_text=cdef_text,
    )

    output_content = "\n".join(line.rstrip() for line in output_content.splitlines()) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"Generated bindings at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CFFI bindings from llama.cpp headers"
    )
    parser.add_argument(
        "--vendor-path",
        type=Path,
        default=None,
        help="Path to vendor/llama.cpp directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for generated bindings"
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default=None,
        help="Commit SHA to record in generated file"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory"
    )

    args = parser.parse_args()

    project_root = args.project_root or get_project_root()
    vendor_path = args.vendor_path or (project_root / "vendor" / "llama.cpp")
    output_path = args.output or (project_root / "src" / "llama_cpp_py_sync" / "_cffi_bindings.py")

    generate_bindings_file(
        project_root=project_root,
        vendor_path=vendor_path,
        output_path=output_path,
        commit_sha=args.commit_sha
    )


if __name__ == "__main__":
    main()
