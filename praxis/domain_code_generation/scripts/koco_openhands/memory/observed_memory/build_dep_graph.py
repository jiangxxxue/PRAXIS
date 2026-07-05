"""Program dependency graph builder — pure AST analysis, no LLM.

Builds a graph of classes, functions, and methods with call/containment/
inheritance edges.  Runs in ~50ms for typical project sizes.

Usage:
    from memory.observed_memory.build_dep_graph import build_dep_graph, save_dep_graph

    graph = build_dep_graph(code_root, framework, example)
    save_dep_graph(graph, framework, example)
"""

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

from memory.config import dep_graph_path


EXCLUDE_DIRS = frozenset({"test_code", "__pycache__", "tests", "__pypackages__"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_excluded(rel_path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in rel_path.parts)


def _module_name(rel_path: str) -> str:
    """Derive Python module name from a relative file path."""
    return rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py").removesuffix(".__init__")


def _get_docstring(node: ast.AST) -> str | None:
    """Extract the first line of a docstring, or None."""
    doc = ast.get_docstring(node)
    if doc:
        return doc.split("\n", 1)[0].strip()
    return None


def _base_name(base: ast.expr) -> str:
    """Extract a readable name from a base-class expression."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


def _decorator_names(node: ast.AST) -> list[str]:
    """Extract decorator names from a function or class definition."""
    decorators = getattr(node, "decorator_list", None) or []
    names = []
    for d in decorators:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(d.attr)
        elif isinstance(d, ast.Call):
            if isinstance(d.func, ast.Name):
                names.append(d.func.id)
            elif isinstance(d.func, ast.Attribute):
                names.append(d.func.attr)
    return names


def _parse_file_safe(py_file: Path) -> ast.AST | None:
    try:
        return ast.parse(py_file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def collect_project_functions(code_root: Path) -> set[str]:
    """Parse Python files and return defined function/method leaf names."""

    names: set[str] = set()
    for py_file in sorted(code_root.rglob("*.py")):
        rel = py_file.relative_to(code_root)
        if _is_excluded(rel):
            continue
        tree = _parse_file_safe(py_file)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


def collect_project_classes(code_root: Path) -> dict[str, dict]:
    """Parse Python files and return class metadata for method resolution."""

    classes: dict[str, dict] = {}
    for py_file in sorted(code_root.rglob("*.py")):
        rel = py_file.relative_to(code_root)
        if _is_excluded(rel):
            continue
        tree = _parse_file_safe(py_file)
        if tree is None:
            continue
        rel_path = str(rel).replace("\\", "/")
        module = _module_name(rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
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
    """Extract best-effort project-internal call edges from an AST subtree."""

    edges: list[dict] = []
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
                        "kind": "call",
                        "lineno": lineno,
                    })
                elif obj_name in project_classes and attr in project_classes[obj_name]["methods"]:
                    edges.append({
                        "target": f"{obj_name}.{attr}",
                        "kind": "call",
                        "lineno": lineno,
                    })
            elif isinstance(func.value, ast.Attribute):
                class_name = func.value.attr
                if class_name in project_classes and attr in project_classes[class_name]["methods"]:
                    edges.append({
                        "target": f"{class_name}.{attr}",
                        "kind": "call",
                        "lineno": lineno,
                    })
        elif isinstance(func, ast.Name) and func.id in project_funcs:
            edges.append({
                "target": func.id,
                "kind": "call",
                "lineno": lineno,
            })

    return edges


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_dep_graph(code_root: str, framework: str, example: str) -> dict:
    """Build a complete dependency graph from a code directory.

    Args:
        code_root: Path to the code/ directory (project source).
        framework: Framework name (e.g. "verl").
        example: Test example name (e.g. "prime").

    Returns:
        Dict with "$schema", "meta", "nodes", "edges" keys.
    """
    code_path = Path(code_root)

    # Pre-collect project-level name registries for call resolution
    project_funcs = collect_project_functions(code_path)
    project_classes = collect_project_classes(code_path)

    nodes: list[dict] = []
    edges: list[dict] = []

    # ── Pass 1: collect all nodes + containment edges ──
    for py_file in sorted(code_path.rglob("*.py")):
        rel_path = py_file.relative_to(code_path)
        if _is_excluded(rel_path):
            continue

        tree = _parse_file_safe(py_file)
        if tree is None:
            continue

        rel_str = str(rel_path).replace("\\", "/")
        module = _module_name(rel_str)

        # Walk top-level nodes only
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                _add_class_node(stmt, nodes, edges, rel_str, module, project_funcs, project_classes)

            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _add_function_node(stmt, nodes, edges, rel_str, module, project_funcs, project_classes)

    # ── Pass 2: resolve inheritance edge targets ──
    _resolve_inheritance_targets(edges, nodes)

    # ── Pass 3: deduplicate edges ──
    edges = _dedup_edges(edges)

    # ── Stats ──
    skipped_calls = sum(
        1 for e in edges if e["kind"] == "call" and not e.get("target")
    )

    return {
        "$schema": "DEP_GRAPH_V1",
        "meta": {
            "framework": framework,
            "example": example,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(code_path),
            "graph_version": "1.0",
        },
        "nodes": nodes,
        "edges": edges,
        "_stats": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "call_edges": sum(1 for e in edges if e["kind"] == "call"),
            "containment_edges": sum(1 for e in edges if e["kind"] == "contains"),
            "inheritance_edges": sum(1 for e in edges if e["kind"] == "inherits"),
            "skipped_calls": skipped_calls,
        },
    }


def _add_class_node(
    cls_def: ast.ClassDef,
    nodes: list[dict],
    edges: list[dict],
    file_path: str,
    module: str,
    project_funcs: set[str],
    project_classes: dict[str, dict],
):
    """Add a class node and process its methods."""
    class_id = cls_def.name
    method_ids = []

    node = {
        "id": class_id,
        "kind": "class",
        "name": cls_def.name,
        "qualified_name": f"{module}.{cls_def.name}",
        "file_path": file_path,
        "lineno": cls_def.lineno,
        "end_lineno": cls_def.end_lineno,
        "docstring": _get_docstring(cls_def),
        "decorators": _decorator_names(cls_def),
        "bases": [_base_name(b) for b in cls_def.bases],
        "methods": [],
    }
    nodes.append(node)

    # Process methods
    for stmt in cls_def.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        method_id = f"{class_id}.{stmt.name}"
        method_ids.append(method_id)

        method_node = {
            "id": method_id,
            "kind": "method",
            "name": stmt.name,
            "qualified_name": f"{module}.{class_id}.{stmt.name}",
            "parent_class": class_id,
            "file_path": file_path,
            "lineno": stmt.lineno,
            "end_lineno": stmt.end_lineno,
            "docstring": _get_docstring(stmt),
            "decorators": _decorator_names(stmt),
        }
        nodes.append(method_node)

        # Containment edge
        edges.append({
            "source": class_id,
            "target": method_id,
            "kind": "contains",
            "lineno": None,
        })

        # Call edges from this method
        call_edges = resolve_call_edges(
            stmt, project_funcs, project_classes, enclosing_class=class_id,
        )
        for ce in call_edges:
            edges.append({
                "source": method_id,
                "target": ce["target"],
                "kind": "call",
                "lineno": ce["lineno"],
            })

    node["methods"] = method_ids

    # Inheritance edges (target resolved in pass 2)
    for base in cls_def.bases:
        base_name = _base_name(base)
        if base_name and base_name != "object":
            edges.append({
                "source": class_id,
                "target": base_name,
                "kind": "inherits",
                "lineno": cls_def.lineno,
            })


def _add_function_node(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
    nodes: list[dict],
    edges: list[dict],
    file_path: str,
    module: str,
    project_funcs: set[str],
    project_classes: dict[str, dict],
):
    """Add a top-level function node."""
    func_id = func_def.name

    node = {
        "id": func_id,
        "kind": "function",
        "name": func_def.name,
        "qualified_name": f"{module}.{func_def.name}",
        "file_path": file_path,
        "lineno": func_def.lineno,
        "end_lineno": func_def.end_lineno,
        "docstring": _get_docstring(func_def),
        "decorators": _decorator_names(func_def),
    }
    nodes.append(node)

    # Call edges from this function
    call_edges = resolve_call_edges(
        func_def, project_funcs, project_classes, enclosing_class=None,
    )
    for ce in call_edges:
        edges.append({
            "source": func_id,
            "target": ce["target"],
            "kind": "call",
            "lineno": ce["lineno"],
        })


def _resolve_inheritance_targets(edges: list[dict], nodes: list[dict]):
    """Resolve inheritance edge targets to known class node IDs."""
    known_ids = {n["id"] for n in nodes}
    for edge in edges:
        if edge["kind"] != "inherits":
            continue
        target = edge.get("target")
        if target and target not in known_ids:
            # Try parent-class resolution in case of external base
            pass  # keep as-is for external bases


def _dedup_edges(edges: list[dict]) -> list[dict]:
    """Remove duplicate edges (same source, target, kind)."""
    seen = set()
    unique = []
    for e in edges:
        key = (e["source"], e.get("target"), e["kind"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_dep_graph(graph: dict, framework: str, example: str) -> Path:
    """Write dependency graph to derived/graph_knowledge/{framework}/{example}/dep_graph.json."""

    out = dep_graph_path(framework, example)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Strip internal _stats but keep a copy under "stats"
    stats = graph.get("_stats")
    to_write = {k: v for k, v in graph.items() if not k.startswith("_")}
    if stats:
        to_write["stats"] = stats
    out.write_text(json.dumps(to_write, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_dep_graph(framework: str, example: str) -> dict | None:
    """Read dependency graph, or None if not found."""
    path = dep_graph_path(framework, example)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
