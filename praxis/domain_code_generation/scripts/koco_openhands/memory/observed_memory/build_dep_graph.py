"""Program dependency graph builder — pure AST analysis, no LLM.

Builds a graph of classes, functions, and methods with call/data/containment/
inheritance edges.

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


class _SharedStateVisitor(ast.NodeVisitor):
    def __init__(self, enclosing_class: str | None):
        self.enclosing_class = enclosing_class
        self.reads: set[str] = set()
        self.writes: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        symbol = self._attribute_symbol(node)
        if symbol:
            if isinstance(node.ctx, ast.Load):
                self.reads.add(symbol)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                self.writes.add(symbol)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._record_write_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._record_write_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        symbol = self._target_symbol(node.target)
        if symbol:
            self.reads.add(symbol)
            self.writes.add(symbol)
        self.visit(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._record_write_target(target)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def _attribute_symbol(self, node: ast.Attribute) -> str:
        if (
            self.enclosing_class
            and isinstance(node.value, ast.Name)
            and node.value.id in {"self", "cls"}
        ):
            return f"{self.enclosing_class}.{node.attr}"
        return ""

    def _record_write_target(self, target: ast.expr) -> None:
        symbol = self._target_symbol(target)
        if symbol:
            self.writes.add(symbol)
            if isinstance(target, ast.Subscript):
                self.visit(target.slice)

    def _target_symbol(self, target: ast.expr) -> str:
        if isinstance(target, ast.Attribute):
            return self._attribute_symbol(target)
        if isinstance(target, ast.Subscript):
            value = target.value
            if isinstance(value, ast.Attribute):
                return self._attribute_symbol(value)
        return ""


def collect_shared_state_accesses(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    enclosing_class: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return stable shared-state symbols read and written by one callable."""

    visitor = _SharedStateVisitor(enclosing_class)
    for statement in function.body:
        visitor.visit(statement)
    return sorted(visitor.reads), sorted(visitor.writes)


def add_data_dependency_edges(
    nodes: list[dict],
    edges: list[dict],
) -> None:
    """Add consumer-to-producer edges for shared state without direct calls."""

    call_pairs = {
        frozenset((str(edge.get("source")), str(edge.get("target"))))
        for edge in edges
        if edge.get("kind") == "call" and edge.get("source") and edge.get("target")
    }
    writers: dict[str, list[dict]] = {}
    readers: dict[str, list[dict]] = {}
    for node in nodes:
        for symbol in node.get("data_writes") or []:
            writers.setdefault(symbol, []).append(node)
        for symbol in node.get("data_reads") or []:
            readers.setdefault(symbol, []).append(node)

    for symbol in sorted(set(writers) & set(readers)):
        for consumer in readers[symbol]:
            for producer in writers[symbol]:
                consumer_id = str(consumer.get("id") or "")
                producer_id = str(producer.get("id") or "")
                if not consumer_id or not producer_id or consumer_id == producer_id:
                    continue
                if frozenset((consumer_id, producer_id)) in call_pairs:
                    continue
                edges.append({
                    "source": consumer_id,
                    "target": producer_id,
                    "kind": "data",
                    "data_symbols": [symbol],
                    "lineno": consumer.get("lineno"),
                })


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

    # ── Pass 3: add shared-state data dependencies ──
    add_data_dependency_edges(nodes, edges)

    # ── Pass 4: deduplicate edges ──
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
            "graph_version": "1.1",
        },
        "nodes": nodes,
        "edges": edges,
        "_stats": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "call_edges": sum(1 for e in edges if e["kind"] == "call"),
            "data_edges": sum(1 for e in edges if e["kind"] == "data"),
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
        data_reads, data_writes = collect_shared_state_accesses(
            stmt,
            enclosing_class=class_id,
        )

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
            "data_reads": data_reads,
            "data_writes": data_writes,
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
        if key in seen:
            if e["kind"] == "data":
                existing = next(
                    item
                    for item in unique
                    if (item["source"], item.get("target"), item["kind"]) == key
                )
                symbols = set(existing.get("data_symbols") or [])
                symbols.update(e.get("data_symbols") or [])
                existing["data_symbols"] = sorted(symbols)
            continue
        seen.add(key)
        unique.append(e)
    return unique


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_dep_graph(graph: dict, framework: str, example: str) -> Path:
    """Write the dependency graph to the run-scoped graph directory."""

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
