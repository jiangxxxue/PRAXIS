import ast
import shutil
from pathlib import Path

from .config import PROJECT_ROOT

from runner import _stub_one_function, _parse_impl_location


def build_practice_ws(spec: dict, dst: Path) -> dict:
    """Build practice workspace once. Reused across iterations via _restub_by_name.

    Symmetric with cmd_infer's workspace: code/ + knowledge_corpus/, test_code excluded.
    Stubs ONLY the practice function. Other GT functions remain intact (matches eval).
    Layout: dst/workspace/{code, knowledge_corpus}
    """
    framework = spec["framework"]
    example = spec["example"]
    code_src = PROJECT_ROOT / framework / "test_examples" / example / "code"
    kc_src = PROJECT_ROOT / framework / "knowledge_corpus"

    if dst.exists():
        shutil.rmtree(dst)
    ws = dst / "workspace"
    ws.mkdir(parents=True)

    def _ignore(_d, contents):
        return {c for c in contents if c in ("test_code", "__pycache__", ".pytest_cache")}

    shutil.copytree(code_src, ws / "code", symlinks=True, ignore=_ignore)
    shutil.copytree(kc_src, ws / "knowledge_corpus", symlinks=True)

    rel, start, end = _parse_impl_location(spec["implementation_location"])
    code_file = ws / "code" / rel
    lines = code_file.read_text().splitlines(keepends=True)
    code_file.write_text("".join(_stub_one_function(lines, start, end)))

    return {
        "workspace": str(ws),
        "code": str(ws / "code"),
        "knowledge_corpus": str(ws / "knowledge_corpus"),
    }


def _restub_by_name(file_path: str, function_name: str) -> None:
    """Re-stub a function by name using AST to find current boundaries.

    Safe regardless of how many lines the agent's previous implementation had —
    AST locates the function by name, not by pre-known line numbers.
    Supports dotted names like "ClassName.method_name" — verifies the parent
    ClassDef to avoid ambiguity with same-named top-level functions.
    """
    source = Path(file_path).read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    parts = function_name.split(".")

    if len(parts) == 2:
        # Dotted name: find method inside a specific class
        class_name, method_name = parts
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name == method_name):
                        new_lines = _stub_one_function(lines, item.lineno, item.end_lineno)
                        Path(file_path).write_text("".join(new_lines))
                        return
    else:
        # Simple name: find anywhere in the tree (top-level or nested method).
        # First match wins; for current candidates every name is unique.
        target_name = parts[0]
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == target_name):
                new_lines = _stub_one_function(lines, node.lineno, node.end_lineno)
                Path(file_path).write_text("".join(new_lines))
                return

    raise RuntimeError(f"Function {function_name!r} not found in {file_path}")
