"""Post-generate validation for test_input.py files.

Checks syntax, import availability, and mock.patch target existence.
Runs WITHOUT Docker — pure AST analysis + filesystem checks.
"""

import ast
import os
import re
import sys
from pathlib import Path

# ── Known available packages per framework (Docker image pre-installed) ──

# From Dockerfile.raganything.smolagents (Python 3.10-slim)
_RAGANYTHING_PACKAGES = frozenset({
    "torch", "torchvision", "torchaudio", "raganything", "pytest",
    "pytest_asyncio", "faiss_cpu", "faiss", "google_cloud_storage",
    "google.cloud", "google.cloud.storage", "sentence_transformers",
    "sentence_transformers.util", "langchain_core", "langchain",
    "langchain_community", "langchain_text_splitters", "langchain_chroma",
    "mcp", "frontmatter", "smolagents", "datasets", "langchain_huggingface",
    "litellm", "ddgs", "rank_bm25", "mcpadapt", "jsonref", "peft",
    "numpy", "scipy", "pandas", "PIL", "Pillow", "yaml", "tqdm",
    "requests", "urllib3", "certifi", "charset_normalizer", "idna",
    "typing_extensions", "pydantic", "pydantic_core", "attrs", "packaging",
})

_STDLIB_MODULES = frozenset(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else None


def _get_known_packages(framework: str) -> frozenset[str] | None:
    """Return known pre-installed packages for a framework, or None to skip checks."""
    if framework in ("raganything", "smolagents"):
        return _RAGANYTHING_PACKAGES
    # verl / open-r1 / tensorrt: too many deps to list; skip import check
    return None


def _file_exists_in_code(code_root: str, module_path: str) -> bool:
    """Check if a dotted module path exists as a .py file under code_root."""
    parts = module_path.split(".")
    for root, dirs, files in os.walk(code_root):
        # Skip excluded dirs
        dirs[:] = [d for d in dirs if d not in ("test_code", "__pycache__", ".pytest_cache", "tests")]
        # Try matching path parts
        rel = os.path.relpath(root, code_root).replace(os.sep, "/")
        if rel == ".":
            rel_parts = []
        else:
            rel_parts = rel.split("/")
        # Check if we can match this module
        target = os.path.join(root, parts[-1] + ".py")
        if os.path.isfile(target):
            return True
        # Also check __init__.py for packages
        if len(parts) >= 2:
            init_file = os.path.join(root, *parts[:-1], "__init__.py")
            leaf_file = os.path.join(root, *parts[:-1], parts[-1] + ".py")
            if os.path.isfile(init_file) or os.path.isfile(leaf_file):
                return True
    return False


def validate_test_input(
    source: str,
    code_root: str,
    framework: str = "",
) -> list[str]:
    """Validate a test_input.py source string.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    # 1. Syntax check
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    # 2. Extract mock.patch targets and verify existence
    mock_targets = _extract_mock_patch_targets(tree)
    for target in mock_targets:
        if not _mock_target_exists(target, code_root):
            errors.append(
                f"mock.patch target '{target}' not found in code directory. "
                f"Verify the module path and function/class name exist."
            )

    # 3. Import check (only for frameworks with known package lists)
    known = _get_known_packages(framework)
    if known is not None and _STDLIB_MODULES is not None:
        imports = _extract_top_level_imports(tree)
        for imp in imports:
            if not _is_import_available(imp, known):
                errors.append(
                    f"Import '{imp}' is not a stdlib module or known pre-installed package. "
                    f"Use only packages available in the Docker evaluation environment."
                )

    return errors


def _extract_mock_patch_targets(tree: ast.AST) -> list[str]:
    """Extract all mock.patch("target") string arguments from AST."""
    targets: list[str] = []

    class MockPatchVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # mock.patch("target") or mock.patch("target", ...)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "patch":
                    # Could be mock.patch, or unittest.mock.patch
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                            targets.append(first_arg.value)
            self.generic_visit(node)

    MockPatchVisitor().visit(tree)
    return targets


def _mock_target_exists(target: str, code_root: str) -> bool:
    """Check if a mock.patch target refers to a real module/function/class.

    The target is a dotted path like "lightrag.operate.clean_str".
    We check if the module part exists under code_root.
    """
    parts = target.split(".")
    if len(parts) < 2:
        return False

    # The module path is everything before the last dot
    module = parts[:-1]
    symbol = parts[-1]

    # Try as a .py file
    module_path = "/".join(module) + ".py"
    full = os.path.join(code_root, module_path)
    if os.path.isfile(full):
        try:
            with open(full, encoding="utf-8") as f:
                mod_source = f.read()
            mod_tree = ast.parse(mod_source)
            for node in ast.walk(mod_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == symbol:
                        return True
                elif isinstance(node, ast.ClassDef):
                    if node.name == symbol:
                        return True
        except (SyntaxError, UnicodeDecodeError):
            pass

    # Try as a package
    module_path = "/".join(module) + "/__init__.py"
    full = os.path.join(code_root, module_path)
    if os.path.isfile(full):
        return True  # Package exists, accept it

    return False


def _extract_top_level_imports(tree: ast.AST) -> list[str]:
    """Extract ALL import module names from the entire AST (not just top-level).

    Uses ast.walk to find imports inside functions like setup_environment()
    as well as at the module level.
    """
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])

    return list(set(imports))


def _is_import_available(module_top: str, known: frozenset[str]) -> bool:
    """Check if a top-level module is available."""
    if _STDLIB_MODULES and module_top in _STDLIB_MODULES:
        return True
    if module_top in known:
        return True
    # Common aliases
    aliases = {
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "bs4": "beautifulsoup4",
    }
    if module_top in aliases:
        return False  # Not guaranteed to be installed
    return False
