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
    "litellm", "openai", "ddgs", "rank_bm25", "mcpadapt", "jsonref", "peft",
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
    implementation_location: str = "",
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

    temporary_workspace_paths = sorted(set(
        re.findall(r"/tmp/tmp[^/'\"\s]+/workspace(?:/[^'\"\s]*)?", source)
    ))
    if temporary_workspace_paths:
        errors.append(
            "test_input contains agent-temporary workspace path(s): "
            f"{', '.join(temporary_workspace_paths)}. "
            "Do not hard-code /tmp/tmp.../workspace paths; derive portable paths "
            "from __file__ or rely on the coverage runner's import paths."
        )

    # 2. Extract mock.patch targets and verify project-local targets. External
    # packages are handled with the same framework policy as imports below.
    known = _get_known_packages(framework)
    mock_targets = _extract_mock_patch_targets(tree)
    for target in mock_targets:
        top_level = target.split(".", 1)[0]
        is_local = _top_level_module_exists(code_root, top_level)
        is_stdlib = _STDLIB_MODULES is not None and top_level in _STDLIB_MODULES
        is_known_external = (
            not is_local
            and (
                is_stdlib
                or known is None
                or _is_import_available(top_level, known)
            )
        )
        if not is_known_external and not _mock_target_exists(target, code_root):
            errors.append(
                f"mock.patch target '{target}' not found in code directory. "
                f"Verify the module path and function/class name exist."
            )

    # 3. Import check. For small framework images, use an explicit allowlist.
    # Large framework images have too many dependencies to enumerate reliably,
    # so their imports are verified later by coverage inside the actual image.
    if _STDLIB_MODULES is not None and known is not None:
        for module_name in _extract_import_modules(tree):
            top_level = module_name.split(".", 1)[0]
            if top_level in _STDLIB_MODULES:
                continue
            if _is_import_available(top_level, known):
                continue
            if _file_exists_in_code(code_root, module_name):
                continue
            errors.append(
                f"Import '{module_name}' is not a stdlib module, known "
                "pre-installed package, or project module under code/. "
                "Use only packages available in the Docker evaluation environment."
            )

    if implementation_location:
        signature = _target_signature(code_root, implementation_location)
        if signature is not None:
            errors.extend(_validate_test_cases(tree, signature))

    return errors


def _target_signature(
    code_root: str,
    implementation_location: str,
) -> tuple[set[str], bool] | None:
    """Return (accepted input names, accepts arbitrary keyword arguments)."""
    relative_path, line_start = _parse_implementation_location(implementation_location)
    if not relative_path or not line_start:
        return None

    source_path = Path(code_root) / relative_path
    if not source_path.is_file():
        return None

    try:
        target_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    candidates = [
        node
        for node in ast.walk(target_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line_start <= getattr(node, "end_lineno", node.lineno)
    ]
    if not candidates:
        return None

    target = min(
        candidates,
        key=lambda node: getattr(node, "end_lineno", node.lineno) - node.lineno,
    )
    positional = [*target.args.posonlyargs, *target.args.args]
    is_method = bool(positional and positional[0].arg in {"self", "cls"})
    if is_method:
        positional = positional[1:]

    accepted = {arg.arg for arg in positional}
    accepted.update(arg.arg for arg in target.args.kwonlyargs)
    if is_method:
        # The coverage wrapper consumes this special argument before calling the method.
        accepted.add("config")

    return accepted, target.args.kwarg is not None


def _parse_implementation_location(location: str) -> tuple[str, int]:
    """Parse a benchmark implementation location into a code-relative path and line."""
    relative = location.removeprefix("code/")
    if ":line " in relative:
        path_part, range_part = relative.split(":line ", 1)
    elif ":" in relative:
        path_part, range_part = relative.rsplit(":", 1)
    else:
        return "", 0

    try:
        line_start = int(range_part.split("-", 1)[0])
    except ValueError:
        return "", 0
    return path_part, line_start


def _validate_test_cases(
    tree: ast.AST,
    signature: tuple[set[str], bool],
) -> list[str]:
    """Check statically-readable test case inputs against the target signature."""
    accepted_names, accepts_kwargs = signature
    errors: list[str] = []

    for category, case_index, case_node in _iter_test_case_nodes(tree):
        inputs = _static_case_input_names(case_node)
        if inputs is not None and not accepts_kwargs:
            unexpected = sorted(name for name in inputs if name not in accepted_names)
            if unexpected:
                errors.append(
                    f"{category}[{case_index}] passes unexpected input(s): "
                    f"{', '.join(unexpected)}. Target accepts: "
                    f"{', '.join(sorted(accepted_names)) or '(no inputs)'}."
                )

        if category == "error":
            expected_error = _static_expected_error(case_node)
            if expected_error is None:
                errors.append(
                    f"{category}[{case_index}] must define expected_error as an exception class."
                )
            elif not _is_exception_expression(expected_error):
                errors.append(
                    f"{category}[{case_index}] expected_error must be an exception class "
                    "or non-empty tuple of exception classes, not a string or value."
                )

    return errors


def _iter_test_case_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "test_cases" for target in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if key.value not in {"normal", "edge", "error"}:
                continue
            if not isinstance(value, (ast.List, ast.Tuple)):
                continue
            for index, case_node in enumerate(value.elts):
                yield key.value, index, case_node


def _static_case_input_names(case_node: ast.AST) -> set[str] | None:
    if isinstance(case_node, ast.Call) and isinstance(case_node.func, ast.Name) and case_node.func.id == "case":
        return {
            keyword.arg
            for keyword in case_node.keywords
            if keyword.arg is not None and keyword.arg not in {"note", "expected_error"}
        }

    if isinstance(case_node, ast.Dict):
        for key, value in zip(case_node.keys, case_node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "inputs"
                and isinstance(value, ast.Dict)
            ):
                names = set()
                for input_key in value.keys:
                    if not isinstance(input_key, ast.Constant) or not isinstance(input_key.value, str):
                        return None
                    names.add(input_key.value)
                return names
    return None


def _static_expected_error(case_node: ast.AST) -> ast.AST | None:
    if isinstance(case_node, ast.Call) and isinstance(case_node.func, ast.Name) and case_node.func.id == "case":
        return next(
            (keyword.value for keyword in case_node.keywords if keyword.arg == "expected_error"),
            None,
        )

    if isinstance(case_node, ast.Dict):
        for key, value in zip(case_node.keys, case_node.values):
            if isinstance(key, ast.Constant) and key.value == "expected_error":
                return value
    return None


def _is_exception_expression(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.Tuple) and node.elts:
        return all(_is_exception_expression(item) for item in node.elts)
    return False


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
    Resolve the longest project-module prefix, then verify that its first
    attribute is bound in that module. Later attributes may belong to an
    imported module or object, matching unittest.mock.patch resolution.
    """
    parts = target.split(".")
    if len(parts) < 2:
        return False

    for prefix_length in range(len(parts) - 1, 0, -1):
        module = parts[:prefix_length]
        attributes = parts[prefix_length:]
        module_suffix = os.path.join(*module) + ".py"
        package_suffix = os.path.join(*module, "__init__.py")
        module_files = _matching_code_files(code_root, module_suffix)
        package_files = _matching_code_files(code_root, package_suffix)
        if not module_files and not package_files:
            continue

        if not attributes:
            return True
        first_attribute = attributes[0]
        for full in [*module_files, *package_files]:
            try:
                with open(full, encoding="utf-8") as source:
                    module_tree = ast.parse(source.read())
                if first_attribute in _module_bound_names(module_tree):
                    return True
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
        return False

    return False


def _matching_code_files(code_root: str, suffix: str) -> list[str]:
    matches = []
    normalized_suffix = os.path.normpath(suffix)
    for root, dirs, files in os.walk(code_root):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in ("test_code", "__pycache__", ".pytest_cache", "tests")
        ]
        for filename in files:
            path = os.path.join(root, filename)
            if path.endswith(os.sep + normalized_suffix):
                matches.append(path)
    return matches


def _module_bound_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return names


def _top_level_module_exists(code_root: str, module: str) -> bool:
    """Return whether a top-level module/package is provided by the benchmark."""
    for root, dirs, files in os.walk(code_root):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in ("test_code", "__pycache__", ".pytest_cache", "tests")
        ]
        if f"{module}.py" in files:
            return True
        package_dir = os.path.join(root, module)
        if os.path.isfile(os.path.join(package_dir, "__init__.py")):
            return True
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


def _extract_import_modules(tree: ast.AST) -> list[str]:
    """Extract full module names imported anywhere in the test input AST."""
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return sorted(set(imports))


def _is_import_available(module_top: str, known: frozenset[str]) -> bool:
    """Check if a top-level module is available."""
    if _STDLIB_MODULES and module_top in _STDLIB_MODULES:
        return True
    if module_top in known or any(
        package.startswith(module_top + ".")
        for package in known
    ):
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
