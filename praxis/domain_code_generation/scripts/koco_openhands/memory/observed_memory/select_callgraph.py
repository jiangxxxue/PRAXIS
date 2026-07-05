"""Call-graph based function selection strategies.

Strategies:
- indegree:  Select functions most frequently called by other project code.
             Rationale: high in-degree → core utility / domain entry point.
- outdegree: Select functions that call the most other project-internal functions.
             Rationale: high out-degree → convergence point requiring broad understanding.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path



def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _simple_name(full_name: str) -> str:
    """Extract the leaf function/method name from a potentially dotted name.

    'RayPPOTrainer.update_difficulty_and_skip_gid_set' → 'update_difficulty_and_skip_gid_set'
    'compute_score' → 'compute_score'
    """
    return full_name.rsplit(".", 1)[-1]


def extract_call_names(tree: ast.AST) -> list[str]:
    """Extract all call target names from an AST subtree.

    Handles both direct calls (foo()) and attribute calls (obj.method()).
    """
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return calls


# Backward-compat alias
_extract_call_names = extract_call_names


EXCLUDE_DIRS = frozenset({"test_code", "__pycache__", "tests", "__pypackages__"})


def is_excluded(rel_path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in rel_path.parts)


_is_excluded = is_excluded


def _normalize(name: str) -> str:
    """Normalize a name for case/hyphen/underscore-insensitive matching."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def determine_scope_dirs(code_root: str, example: str, framework: str = "") -> set[str] | None:
    """Find directories under code/ that correspond to the benchmark example."""

    code_path = Path(code_root)
    target = _normalize(example)
    prefixes: set[str] = set()

    for directory in code_path.rglob("*"):
        if not directory.is_dir():
            continue
        rel = directory.relative_to(code_path)
        if is_excluded(rel):
            continue
        if _normalize(directory.name) == target:
            prefixes.add(str(rel).replace("\\", "/"))
    if prefixes:
        return prefixes

    if framework:
        framework_name = _normalize(framework)
        for directory in code_path.iterdir():
            if not directory.is_dir() or is_excluded(Path(directory.name)):
                continue
            if _normalize(directory.name) == framework_name:
                continue
            prefixes.add(directory.name)
    return prefixes if prefixes else None


def collect_project_functions(
    code_root: Path, scope_dirs: set[str] | None = None
) -> set[str]:
    """Parse .py files in scope_dirs (or all) and return the set of defined function/method names."""
    names = set()
    for py_file in sorted(code_root.rglob("*.py")):
        rel = py_file.relative_to(code_root)
        if is_excluded(rel):
            continue
        if scope_dirs is not None:
            rel_str = str(rel).replace("\\", "/")
            if not any(rel_str.startswith(d + "/") or rel_str == d for d in scope_dirs):
                continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


_collect_project_functions = collect_project_functions


def collect_project_classes(code_root: Path) -> dict[str, dict]:
    """Parse all .py files and return class metadata.

    Returns dict mapping class_name -> {"methods": set[str], "file_path": str, "module": str}.
    """
    classes: dict[str, dict] = {}
    for py_file in sorted(code_root.rglob("*.py")):
        if is_excluded(py_file.relative_to(code_root)):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel_path = str(py_file.relative_to(code_root)).replace("\\", "/")
        module = rel_path.replace("/", ".").removesuffix(".py")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = set()
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
                classes[node.name] = {
                    "methods": methods,
                    "file_path": rel_path,
                    "module": module,
                }
    return classes


def resolve_call_edges(
    tree: ast.AST,
    project_funcs: set[str],
    project_classes: dict[str, dict],
    enclosing_class: str | None = None,
) -> list[dict]:
    """Extract call edges from an AST, resolving targets to node IDs.

    Resolution rules (best-effort):
    1. foo() → function 'foo' if in project_funcs
    2. self.method() / cls.method() → method of enclosing class
    3. OtherClass.method() → method of known class
    4. Third-party calls → skipped

    Returns list of {"target": str, "kind": "call", "lineno": int} dicts.
    Caller fills in the "source" field after all nodes are collected.
    """
    edges = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        lineno = node.lineno
        func = node.func

        if isinstance(func, ast.Attribute):
            attr = func.attr
            if isinstance(func.value, ast.Name):
                obj_name = func.value.id
                if obj_name in ("self", "cls") and enclosing_class:
                    edges.append({
                        "target": f"{enclosing_class}.{attr}",
                        "kind": "call", "lineno": lineno,
                    })
                elif obj_name in project_classes and attr in project_classes[obj_name]["methods"]:
                    edges.append({
                        "target": f"{obj_name}.{attr}",
                        "kind": "call", "lineno": lineno,
                    })
            # foo.bar() where foo might be a module
            elif isinstance(func.value, ast.Attribute):
                # ClassName.method() via module.ClassName
                if func.value.attr in project_classes and attr in project_classes[func.value.attr]["methods"]:
                    edges.append({
                        "target": f"{func.value.attr}.{attr}",
                        "kind": "call", "lineno": lineno,
                    })
        elif isinstance(func, ast.Name):
            if func.id in project_funcs:
                edges.append({
                    "target": func.id,
                    "kind": "call", "lineno": lineno,
                })

    return edges


def count_indegree(
    code_root: Path, gt_records: list[dict], scope_dirs: set[str] | None = None
) -> dict[str, int]:
    """Count how many times each GT function is called by other project code.

    Scans .py files in scope_dirs (or all if None) and counts calls whose
    name matches a GT function's leaf name.
    """
    # Map leaf name → full GT name
    leaf_to_full: dict[str, str] = {}
    for r in gt_records:
        leaf = _simple_name(r["function_name"])
        leaf_to_full[leaf] = r["function_name"]

    indegree: dict[str, int] = defaultdict(int)

    for py_file in sorted(code_root.rglob("*.py")):
        rel = py_file.relative_to(code_root)
        if _is_excluded(rel):
            continue
        if scope_dirs is not None:
            rel_str = str(rel).replace("\\", "/")
            if not any(rel_str.startswith(d + "/") or rel_str == d for d in scope_dirs):
                continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for call_name in _extract_call_names(tree):
            if call_name in leaf_to_full:
                indegree[leaf_to_full[call_name]] += 1

    return dict(indegree)


def count_outdegree(
    code_root: Path, gt_records: list[dict], scope_dirs: set[str] | None = None
) -> dict[str, int]:
    """Count how many distinct project-internal functions each GT function calls.

    First builds a complete set of all function/method names defined in the
    instance scope, then for each GT function counts distinct calls.
    """
    project_funcs = _collect_project_functions(code_root, scope_dirs)

    outdegree: dict[str, int] = {}

    for r in gt_records:
        impl = r.get("func_implementation", "")
        if not impl:
            outdegree[r["function_name"]] = 0
            continue

        try:
            tree = ast.parse(impl)
        except SyntaxError:
            outdegree[r["function_name"]] = 0
            continue

        call_names = set(_extract_call_names(tree))

        # Remove self-calls
        call_names.discard(_simple_name(r["function_name"]))

        # Count only calls to project-internal functions
        project_calls = call_names & project_funcs
        outdegree[r["function_name"]] = len(project_calls)

    return outdegree


def select_by_strategy(
    strategy: str,
    data_file: str,
    code_root: str,
    top_k: int | None = None,
    framework: str = "",
    example: str = "",
) -> list[dict]:
    """Select functions using the specified call-graph strategy.

    Args:
        strategy: "indegree" or "outdegree"
        data_file: Path to JSONL file with GT function records
        code_root: Path to the code/ directory
        top_k: Max number of functions to return (None = all)
        framework: Framework name (for scope filtering)
        example: Test example name (for scope filtering)

    Returns:
        List of candidate dicts compatible with candidates.json
    """
    records = load_jsonl(data_file)
    gt_functions = {r["function_name"]: r for r in records}
    code_path = Path(code_root)

    scope_dirs = determine_scope_dirs(code_root, example, framework)
    if scope_dirs:
        print(f"    Scope dirs: {sorted(scope_dirs)[:5]}" + ("..." if len(scope_dirs) > 5 else ""))

    if strategy == "indegree":
        scores = count_indegree(code_path, records, scope_dirs)
        label = "in-degree"
    elif strategy == "outdegree":
        scores = count_outdegree(code_path, records, scope_dirs)
        label = "out-degree"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Sort by score descending, then by name for stability
    sorted_funcs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    if top_k:
        sorted_funcs = sorted_funcs[:top_k]

    candidates = []
    for func_name, count in sorted_funcs:
        record = gt_functions[func_name]
        candidates.append({
            "function_name": func_name,
            "implementation_location": record["implementation_location"],
            "rationale": f"{label}={count}",
        })

    # Append functions with 0 score at the end
    seen = {c["function_name"] for c in candidates}
    for r in records:
        if r["function_name"] not in seen:
            candidates.append({
                "function_name": r["function_name"],
                "implementation_location": r["implementation_location"],
                "rationale": f"{label}=0",
            })

    return candidates
