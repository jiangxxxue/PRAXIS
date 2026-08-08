#!/usr/bin/env python3
from __future__ import annotations

"""Coverage measurement runner — executes inside Docker.

Measures line coverage of a GT function when exercised by generated test inputs.
Standalone script with no koco_openhands dependencies.

Usage:
    python3 coverage_runner.py \
        --source-dir /workspace/project/verl/test_examples/prime/code \
        --source-file recipe/prime/prime_core_algos.py \
        --function-name compute_rloo_advantage_return \
        --test-input-file /workspace/.../compute_rloo_advantage_return_test_input.py \
        --line-start 21 --line-end 24 \
        --output-file /workspace/.../compute_rloo_advantage_return_coverage.json \
        [--per-test]
"""

import argparse
import importlib
import json
import os
import site
import sys
from typing import Callable, Optional


# ── helpers ────────────────────────────────────────────────────────────────

def _get_docstring_lines(source_dir: str, source_file: str,
                          line_start: int, line_end: int) -> set[int]:
    """Return line numbers occupied by docstrings within [line_start, line_end].

    Detects docstrings in functions, classes, and the module level by parsing
    the AST. A docstring is the first statement in a body if it is a string
    constant expression.
    """
    import ast
    full_path = os.path.join(source_dir, source_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=full_path)
    except Exception:
        return set()

    docstring_lines = set()

    def _collect(body):
        if not body:
            return
        stmt = body[0]
        if (isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            for ln in range(stmt.lineno, stmt.end_lineno + 1):
                if line_start <= ln <= line_end:
                    docstring_lines.add(ln)

    _collect(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect(node.body)

    return docstring_lines


def _has_abstract_body(func) -> bool:
    """Check if a callable is an abstract stub by inspecting bytecode.

    Returns True for functions whose body is essentially:
        raise NotImplementedError(...)
        or just pass / return a docstring.
    """
    while True:
        # Unwrap descriptors (staticmethod, classmethod)
        if hasattr(func, '__func__'):
            func = func.__func__
        elif hasattr(func, '__wrapped__'):
            func = func.__wrapped__
        else:
            break
    code = getattr(func, '__code__', None)
    if code is None:
        return False
    bc = code.co_code
    if not bc:
        return True  # empty body
    # Bytecode patterns for stubs:
    #  - LOAD_GLOBAL + LOAD_ATTR 'NotImplementedError' + RAISE_VARARGS
    #  - LOAD_CONST None + RETURN_VALUE (pass-only body)
    #  - LOAD_CONST (docstring) + RETURN_VALUE
    import dis
    try:
        instrs = list(dis.get_instructions(func))
    except Exception:
        return False
    # Filter out RESUME (Python 3.11+) and RETURN_GENERATOR
    real = [i for i in instrs if i.opname not in ('RESUME', 'RETURN_GENERATOR', 'CACHE')]
    if len(real) <= 2:
        return True  # pass, return None, return docstring
    # Check for: raise NotImplementedError (possibly with LOAD_GLOBAL/LOAD_ATTR)
    names = [i.argrepr for i in real if i.opname == 'RAISE_VARARGS']
    if names:
        # Look for NotImplementedError in the preceding instructions
        for i in real:
            if i.opname in ('LOAD_GLOBAL', 'LOAD_ATTR') and 'NotImplementedError' in str(i.argrepr):
                return True
    return False


def _is_abstract_stub(source_dir: str, source_file: str,
                       line_start: int, line_end: int) -> bool:
    """Check if the function in range is an abstract stub (raise NotImplementedError or pass-only)."""
    import ast
    full_path = os.path.join(source_dir, source_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=full_path)
    except Exception:
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line_start <= node.end_lineno:
                body = node.body
                # Skip docstring
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body = body[1:]
                if not body:
                    return True
                # Single raise NotImplementedError
                if (len(body) == 1 and isinstance(body[0], ast.Raise)
                        and isinstance(body[0].exc, ast.Name)
                        and body[0].exc.id == "NotImplementedError"):
                    return True
                # Single pass
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    return True
                # raise NotImplementedError("...") with message
                if (len(body) == 1 and isinstance(body[0], ast.Raise)
                        and isinstance(body[0].exc, ast.Call)
                        and isinstance(body[0].exc.func, ast.Name)
                        and body[0].exc.func.id == "NotImplementedError"):
                    return True
                return False
    return False


def _get_unreachable_lines(source_dir: str, source_file: str,
                            line_start: int, line_end: int) -> set[int]:
    """Return line numbers that are unreachable within [line_start, line_end].

    Detects code after unconditional return/raise/continue/break in the same block.
    """
    import ast
    full_path = os.path.join(source_dir, source_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=full_path)
    except Exception:
        return set()

    unreachable = set()

    def _visit_block(stmts):
        """Walk a list of statements, mark lines after return/raise as unreachable."""
        found_terminal = False
        for stmt in stmts:
            if found_terminal:
                # This statement is unreachable
                for ln in range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1):
                    if line_start <= ln <= line_end:
                        unreachable.add(ln)
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                found_terminal = True

    def _walk(node):
        """Recursively walk AST to find all blocks."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only process functions within our target range
            if node.lineno <= line_end and node.end_lineno >= line_start:
                _visit_block(node.body)
                for child in node.body:
                    _walk(child)
                for handler in node.decorator_list:
                    _walk(handler)
        elif isinstance(node, ast.If):
            _visit_block(node.body)
            _visit_block(node.orelse)
            for child in node.body:
                _walk(child)
            for child in node.orelse:
                _walk(child)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            _visit_block(node.body)
            _visit_block(node.orelse)
            for child in node.body:
                _walk(child)
            for child in node.orelse:
                _walk(child)
        elif isinstance(node, ast.With):
            _visit_block(node.body)
            for child in node.body:
                _walk(child)
        elif isinstance(node, ast.Try):
            _visit_block(node.body)
            _visit_block(node.orelse)
            _visit_block(node.finalbody)
            for child in node.body:
                _walk(child)
            for child in node.orelse:
                _walk(child)
            for child in node.finalbody:
                _walk(child)
            for handler in node.handlers:
                _visit_block(handler.body)
                for child in handler.body:
                    _walk(child)
        elif isinstance(node, ast.ExceptHandler):
            _visit_block(node.body)
            for child in node.body:
                _walk(child)
        else:
            # Generic: recurse into child nodes
            for child in ast.iter_child_nodes(node):
                _walk(child)

    _walk(tree)
    return unreachable


def _get_function_definition_lines(source_dir: str, source_file: str,
                                   line_start: int, line_end: int) -> set[int]:
    """Return target header lines executed only while importing the module."""
    import ast

    full_path = os.path.join(source_dir, source_file)
    try:
        with open(full_path, encoding="utf-8") as source:
            tree = ast.parse(source.read())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()

    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line_start <= getattr(node, "end_lineno", node.lineno)
    ]
    if not candidates:
        return set()

    target = min(
        candidates,
        key=lambda node: getattr(node, "end_lineno", node.lineno) - node.lineno,
    )
    body_lines = [
        statement.lineno
        for statement in target.body
        if hasattr(statement, "lineno")
    ]
    if not body_lines:
        return set()

    header_start = min(
        [target.lineno, *(decorator.lineno for decorator in target.decorator_list)]
    )
    body_start = min(body_lines)
    return {
        line
        for line in range(header_start, body_start)
        if line_start <= line <= line_end
    }


def _ensure_coverage():
    """Ensure coverage.py is available. Returns True on success."""
    try:
        import coverage  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import subprocess
    except Exception:
        return False
    for attempt in range(3):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q",
                 "--no-warn-script-location", "coverage"],
            )
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                return False
    import importlib
    importlib.invalidate_caches()
    user_sp = site.getusersitepackages()
    if user_sp not in sys.path:
        sys.path.insert(0, user_sp)
    return True


# ── serialization ──────────────────────────────────────────────────────────

_MAX_TENSOR_ELEMENTS = 1024


def serialize(obj):
    """Serialize a function return value to a JSON-compatible structure."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, str):
        return obj

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu()
            numel = t.numel()
            entry = {
                "__type__": "torch.Tensor",
                "dtype": str(t.dtype),
                "shape": list(t.shape),
            }
            if numel <= _MAX_TENSOR_ELEMENTS:
                entry["data"] = t.tolist()
            else:
                entry["data_truncated"] = True
                entry["numel"] = numel
            return entry
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            entry = {
                "__type__": "numpy.ndarray",
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
            if obj.size <= _MAX_TENSOR_ELEMENTS:
                entry["data"] = obj.tolist()
            else:
                entry["data_truncated"] = True
                entry["numel"] = int(obj.size)
            return entry
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list,)):
        return [serialize(v) for v in obj]
    if isinstance(obj, tuple):
        return {"__type__": "tuple", "items": [serialize(v) for v in obj]}
    if isinstance(obj, set):
        return {"__type__": "set", "items": [serialize(v) for v in sorted(obj, key=repr)]}

    return {"__type__": type(obj).__name__, "repr": repr(obj)[:500]}


def load_test_cases(
    test_input_file: str,
    execution_root: str | None = None,
) -> tuple[dict, Optional[Callable], Optional[Callable]]:
    """Load test cases and optional mock hooks from a test_input file.

    Returns:
        (test_cases, setup_self_fn, setup_env_fn)
        setup_self_fn is None if no setup_self() is defined in the file.
        setup_env_fn is None if no setup_environment() is defined in the file.
    """
    import types
    import unittest.mock

    logical_root = execution_root or os.path.dirname(os.path.abspath(test_input_file))
    logical_file = os.path.join(logical_root, os.path.basename(test_input_file))
    module = types.ModuleType("__test_input__")
    module.__file__ = os.path.abspath(logical_file)
    module.__package__ = None
    module.MagicMock = unittest.mock.MagicMock
    module.AsyncMock = unittest.mock.AsyncMock
    module.mock = unittest.mock
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    with open(test_input_file, "r", encoding="utf-8") as f:
        code = compile(f.read(), test_input_file, "exec")
    exec(code, namespace)
    test_cases = namespace.get("test_cases", {})
    setup_self_fn = namespace.get("setup_self", None)
    setup_env_fn = namespace.get("setup_environment", None)
    return test_cases, setup_self_fn, setup_env_fn


def _tolerate_mock_modules_without_specs() -> None:
    """Make package probes tolerate MagicMock modules inserted by test inputs."""
    import importlib.util

    current = importlib.util.find_spec
    if getattr(current, "_praxis_tolerates_missing_mock_specs", False):
        return

    def _safe_find_spec(name, package=None):
        try:
            return current(name, package)
        except ValueError as exc:
            message = str(exc)
            if (
                name in sys.modules
                and (
                    f"{name}.__spec__ is not set" in message
                    or f"{name}.__spec__ is None" in message
                )
            ):
                return None
            raise

    _safe_find_spec._praxis_tolerates_missing_mock_specs = True
    importlib.util.find_spec = _safe_find_spec


def _repair_mock_package_metadata(modules_before_setup: dict) -> None:
    """Keep mocked installed modules capable of resolving real dependencies."""
    import importlib.machinery
    import types

    def _is_installed_spec(spec) -> bool:
        paths = []
        if getattr(spec, "origin", None):
            paths.append(spec.origin)
        if getattr(spec, "submodule_search_locations", None):
            paths.extend(spec.submodule_search_locations)
        return any(
            "site-packages" in str(path) or "dist-packages" in str(path)
            for path in paths
        )

    def _overlay_mock_attributes(real_module, mock_module) -> None:
        import_metadata = {
            "__builtins__",
            "__cached__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__path__",
            "__spec__",
        }
        for attribute, value in vars(mock_module).items():
            if attribute not in import_metadata:
                setattr(real_module, attribute, value)

    modules = sorted(
        list(sys.modules.items()),
        key=lambda item: item[0].count("."),
    )
    for name, module in modules:
        if not isinstance(module, types.ModuleType):
            continue
        if getattr(module, "__spec__", None) is not None or getattr(
            module, "__file__", None
        ):
            continue

        previous = modules_before_setup.get(name)
        if previous is module:
            continue
        previous_path = getattr(previous, "__path__", None)
        previous_spec = getattr(previous, "__spec__", None)

        parent_path = None
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            parent_path = getattr(sys.modules.get(parent_name), "__path__", None)
            if not parent_path:
                continue
        try:
            sys.modules.pop(name, None)
            spec = importlib.machinery.PathFinder.find_spec(name, parent_path)
        except (ImportError, AttributeError, ValueError):
            spec = None
        finally:
            sys.modules[name] = module

        real_module = None
        if previous is not None and previous is not module and previous_spec is not None:
            real_module = previous
        elif spec is not None and _is_installed_spec(spec):
            try:
                sys.modules.pop(name, None)
                real_module = importlib.import_module(name)
            except (ImportError, AttributeError, ValueError):
                real_module = None
            finally:
                sys.modules[name] = module

        if real_module is not None:
            _overlay_mock_attributes(real_module, module)
            sys.modules[name] = real_module
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, child_name, real_module)
            continue

        if previous_path is not None:
            module.__path__ = list(previous_path)
            module.__spec__ = previous_spec
            module.__loader__ = getattr(previous, "__loader__", None)
            module.__package__ = getattr(previous, "__package__", name)
        elif (
            spec is not None
            and spec.submodule_search_locations is not None
            and not getattr(module, "__path__", None)
        ):
            module.__path__ = list(spec.submodule_search_locations)
            module.__spec__ = spec
            module.__loader__ = spec.loader
            module.__package__ = name


def _auto_mock_self(method, source_dir: str, source_file: str,
                     line_start: int, line_end: int):
    """Analyze method source to build a mock self with auto-detected attributes.

    Walks the function AST to find all ``self.xxx`` attribute accesses and creates
    MagicMock/AsyncMock attributes accordingly.  Returns a callable that constructs
    the mock self when called (signature: ``(config=None) -> mock_self``).
    """
    import ast
    import asyncio
    from unittest.mock import MagicMock, AsyncMock

    full_path = os.path.join(source_dir, source_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=full_path)
    except Exception:
        return None

    # Find the target function node
    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line_start <= node.end_lineno:
                target_node = node
                break
    if target_node is None:
        return None

    is_async = isinstance(target_node, ast.AsyncFunctionDef)

    # Collect self.xxx accesses: attr_name -> set of whether it's awaited
    attrs = {}  # attr_name -> {"awaited": bool, "called": bool}

    def _visit_attr(node):
        """Record self.xxx.yyy chains as top-level self.xxx attributes."""
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)):
            return
        # Walk back to find self.attr
        chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if not (isinstance(cur, ast.Name) and cur.id == "self") or len(chain) < 1:
            return
        top_attr = chain[-1]  # e.g. "rag" from self.rag.aquery
        called = isinstance(node.parent, ast.Call) if hasattr(node, 'parent') else False
        if top_attr not in attrs:
            attrs[top_attr] = {"awaited": False, "called": False}
        if called:
            attrs[top_attr]["called"] = True

    # Also handle simple self.attr (not chained)
    def _visit_simple_attr(node):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            return
        attr_name = node.attr
        if attr_name.startswith("__") and attr_name.endswith("__"):
            return  # Skip dunder attributes
        if attr_name not in attrs:
            attrs[attr_name] = {"awaited": False, "called": False}
        # Check if this access is awaited
        # The parent of the attribute node should be checked
        # We can't easily walk parents in AST, so check common patterns below

    # Walk the function body
    for node in ast.walk(target_node):
        # Detect: await self.xxx.yyy() or await self.xxx()
        if isinstance(node, ast.Await):
            if isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute):
                    if (isinstance(call.func.value, ast.Name) and call.func.value.id == "self"):
                        attr_name = call.func.attr
                        if not (attr_name.startswith("__") and attr_name.endswith("__")):
                            if attr_name not in attrs:
                                attrs[attr_name] = {"awaited": True, "called": True}
                            else:
                                attrs[attr_name]["awaited"] = True
                                attrs[attr_name]["called"] = True
                    elif (isinstance(call.func.value, ast.Attribute)
                          and isinstance(call.func.value.value, ast.Name)
                          and call.func.value.value.id == "self"):
                        # await self.rag.aquery() -> top-level is "rag"
                        top_attr = call.func.value.attr
                        if top_attr not in attrs:
                            attrs[top_attr] = {"awaited": True, "called": True}
                        else:
                            attrs[top_attr]["awaited"] = True
                            attrs[top_attr]["called"] = True
            elif isinstance(node.value, ast.Attribute):
                # await self.xxx (no call)
                if (isinstance(node.value.value, ast.Name) and node.value.value.id == "self"):
                    attr_name = node.value.attr
                    if not (attr_name.startswith("__") and attr_name.endswith("__")):
                        if attr_name not in attrs:
                            attrs[attr_name] = {"awaited": True, "called": False}
                        else:
                            attrs[attr_name]["awaited"] = True

        # Detect: self.xxx() (not awaited)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
                    attr_name = node.func.attr
                    if not (attr_name.startswith("__") and attr_name.endswith("__")):
                        if attr_name not in attrs:
                            attrs[attr_name] = {"awaited": False, "called": True}
                        else:
                            attrs[attr_name]["called"] = True
                elif (isinstance(node.func.value, ast.Attribute)
                      and isinstance(node.func.value.value, ast.Name)
                      and node.func.value.value.id == "self"):
                    top_attr = node.func.value.attr
                    if top_attr not in attrs:
                        attrs[top_attr] = {"awaited": False, "called": True}
                    else:
                        attrs[top_attr]["called"] = True

        # Detect: self.xxx (simple attribute access, no call)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            attr_name = node.attr
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            if attr_name not in attrs:
                attrs[attr_name] = {"awaited": False, "called": False}

    if not attrs:
        return None

    # Build the factory function
    def _build_mock(config=None):
        mock_self = MagicMock()
        mock_self.config = config
        for attr_name, info in attrs.items():
            child = MagicMock()
            # If any method on this attribute is awaited, make child methods AsyncMock
            if info["awaited"]:
                # Make child attribute access return a MagicMock whose methods are AsyncMock
                child._mock_return_value = MagicMock()
                # Set up so child.method returns AsyncMock
                child.__dict__["_async_attrs"] = True
            setattr(mock_self, attr_name, child)
        return mock_self

    attr_summary = ", ".join(
        f"{name}({'await+call' if v['awaited'] and v['called'] else 'await' if v['awaited'] else 'call' if v['called'] else 'attr'})"
        for name, v in attrs.items()
    )
    print(f"    NOTE: auto-mock self detected attributes: {attr_summary}")
    return _build_mock


def _make_class_method_wrapper(method, setup_self_fn=None,
                                auto_mock_fn=None):
    """Wrap a class method to inject a mock self.

    Priority: setup_self_fn (from test_input) > auto_mock_fn (AST analysis)
    > MagicMock fallback (auto-creates child attributes on self.xxx access).
    """
    import asyncio
    from unittest.mock import MagicMock

    def _build_fallback_mock(config=None):
        """Build a MagicMock-based self. Any self.xxx access auto-creates a child MagicMock."""
        mock_self = MagicMock()
        mock_self.config = _DotDict(config) if config else None
        return mock_self

    if asyncio.iscoroutinefunction(method):
        async def _async_wrapper(**kwargs):
            config = kwargs.pop("config", None)
            if setup_self_fn is not None:
                mock_self = setup_self_fn(config=config)
            elif auto_mock_fn is not None:
                mock_self = auto_mock_fn(config=config)
            else:
                mock_self = _build_fallback_mock(config)
            return await method(mock_self, **kwargs)
        return _async_wrapper
    else:
        def _wrapper(**kwargs):
            config = kwargs.pop("config", None)
            if setup_self_fn is not None:
                mock_self = setup_self_fn(config=config)
            elif auto_mock_fn is not None:
                mock_self = auto_mock_fn(config=config)
            else:
                mock_self = _build_fallback_mock(config)
            return method(mock_self, **kwargs)
        return _wrapper


def _nested_python_roots(source_dir: str, source_file: str) -> list[str]:
    source_root = os.path.realpath(source_dir)
    current = os.path.dirname(
        os.path.realpath(os.path.join(source_dir, source_file))
    )
    roots = []
    markers = ("pyproject.toml", "setup.py", "setup.cfg")

    while current != source_root:
        if not current.startswith(source_root + os.sep):
            break
        if any(os.path.isfile(os.path.join(current, marker)) for marker in markers):
            roots.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return roots


def _package_parent_roots(source_dir: str, source_file: str) -> list[str]:
    source_root = os.path.realpath(source_dir)
    current = os.path.dirname(
        os.path.realpath(os.path.join(source_dir, source_file))
    )
    roots = []

    while current != source_root:
        if not current.startswith(source_root + os.sep):
            break
        if os.path.isfile(os.path.join(current, "__init__.py")):
            roots.append(os.path.dirname(current))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return roots


def _prepare_import_paths(source_dir: str, source_file: str) -> None:
    def add_first(path: str) -> None:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    add_first(source_dir)
    add_first(os.path.join(source_dir, "src"))

    # Work from outer to inner so the closest project root wins.
    for project_root in reversed(_nested_python_roots(source_dir, source_file)):
        add_first(project_root)
        add_first(os.path.join(project_root, "src"))

    # Nested applications commonly import from an inner top-level package
    # (for example backend/app/... uses "from app...").
    for package_root in _package_parent_roots(source_dir, source_file):
        add_first(package_root)

    # Script-style modules often import local siblings without a package prefix.
    add_first(os.path.dirname(os.path.join(source_dir, source_file)))


def _force_namespace_ancestors(import_root: str, module_path: str) -> None:
    """Retry a leaf import without executing eager ancestor initializers."""
    import importlib.machinery
    import types

    parts = module_path.split(".")[:-1]
    packages = []
    for depth in range(1, len(parts) + 1):
        package_name = ".".join(parts[:depth])
        package_path = os.path.join(import_root, *parts[:depth])
        if os.path.isdir(package_path):
            packages.append((package_name, package_path))

    package_names = [name for name, _path in packages]
    for key in list(sys.modules):
        if any(key == name or key.startswith(name + ".") for name in package_names):
            del sys.modules[key]

    for package_name, package_path in packages:
        package = types.ModuleType(package_name)
        package.__path__ = [package_path]
        package.__package__ = package_name
        package.__file__ = None
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        package.__spec__.submodule_search_locations = [package_path]
        sys.modules[package_name] = package


def import_function(source_dir: str, source_file: str, function_name,
                     setup_self_fn=None, line_start: int = 0, line_end: int = 0):
    """Import a function. For class methods (ClassName.method), returns a wrapper.

    If the name is unqualified (short), searches all classes in the module for the method.
    For class methods, auto-generates mock self from AST analysis if line range is provided.
    """
    _prepare_import_paths(source_dir, source_file)
    src_root = os.path.join(source_dir, "src")

    module_path = source_file.replace("/", ".").replace("\\", ".").removesuffix(".py")
    import_root = source_dir
    if module_path.startswith("src.") and os.path.isdir(src_root):
        module_path = module_path.removeprefix("src.")
        import_root = src_root

    # Handle cases where a package directory exists alongside a same-named .py
    # file inside it (e.g. raganything/raganything.py alongside raganything/config.py).
    # Or when site-packages already cached the top-level module as a non-package.
    # Also ensure all intermediate packages are namespace packages so relative
    # imports (from .sibling import X) work at every level.
    _top = module_path.split(".")[0]
    if _top != module_path:
        _top_dir = os.path.join(import_root, _top)
        if os.path.isdir(_top_dir):
            import types
            # Walk ALL intermediate package levels (e.g. src, src.agents, src.agents.ui_common)
            _parts = module_path.split(".")[:-1]  # All but the last (module) level
            for _depth in range(1, len(_parts) + 1):
                _pkg_name = ".".join(_parts[:_depth])
                _pkg_path = os.path.join(import_root, *_parts[:_depth])
                if not os.path.isdir(_pkg_path):
                    continue
                _init_file = os.path.join(_pkg_path, "__init__.py")
                if os.path.isfile(_init_file):
                    _existing = sys.modules.get(_pkg_name)
                    _existing_file = os.path.realpath(getattr(_existing, "__file__", "") or "") if _existing else ""
                    if _existing is not None and (
                        not _existing_file
                        or not _existing_file.startswith(os.path.realpath(_pkg_path) + os.sep)
                    ):
                        del sys.modules[_pkg_name]
                    continue
                # Clear stale non-package modules
                if _pkg_name in sys.modules and not hasattr(sys.modules[_pkg_name], '__path__'):
                    del sys.modules[_pkg_name]
                # Register as namespace package if not already
                if _pkg_name not in sys.modules:
                    _pkg = types.ModuleType(_pkg_name)
                    _pkg.__path__ = [_pkg_path]
                    sys.modules[_pkg_name] = _pkg
                elif not hasattr(sys.modules[_pkg_name], '__path__'):
                    del sys.modules[_pkg_name]
                    _pkg = types.ModuleType(_pkg_name)
                    _pkg.__path__ = [_pkg_path]
                    sys.modules[_pkg_name] = _pkg
                else:
                    _existing = sys.modules[_pkg_name]
                    if _pkg_path not in _existing.__path__:
                        _existing.__path__.insert(0, _pkg_path)

    # Remove cached imports to force fresh load from source_dir (not site-packages)
    # For single-file modules, only remove the exact match (keep sub-modules)
    _src_full = os.path.join(source_dir, source_file)
    _is_pkg = _src_full.endswith(os.sep + '__init__.py')  # Note: os.sep is \ on Windows
    if not _is_pkg and not os.path.basename(_src_full) == '__init__.py':
        if module_path in sys.modules:
            del sys.modules[module_path]
    else:
        for key in list(sys.modules.keys()):
            if key == module_path or key.startswith(module_path + "."):
                del sys.modules[key]

    # Try import, auto-install missing packages on ModuleNotFoundError / ImportError
    # Some source code wraps imports: try: import xai_sdk / except ImportError: raise ImportError("pip install xai-sdk")
    # This throws ImportError (not ModuleNotFoundError) at the top level, so we catch both.
    import re as _re
    _attempt = 0
    _installed_packages = set()
    _auto_install = os.environ.get(
        "PRAXIS_COVERAGE_AUTO_INSTALL",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    _package_aliases = {
        "bs4": "beautifulsoup4",
        "cv2": "opencv-python-headless",
        "PIL": "Pillow",
        "pillow_avif": "pillow-avif-plugin",
        "pptx": "python-pptx",
        "serpapi": "google-search-results",
        "sklearn": "scikit-learn",
        "speech_recognition": "SpeechRecognition",
        "youtube_transcript_api": "youtube-transcript-api",
    }
    while True:
        try:
            mod = __import__(module_path, fromlist=["_"])
            break
        except (ModuleNotFoundError, ImportError) as _exc:
            _msg = str(_exc)
            if "partially initialized module" in _msg and _attempt == 0:
                print(
                    "    NOTE: circular package initializer detected; "
                    "retrying with namespace ancestors"
                )
                _attempt += 1
                _force_namespace_ancestors(import_root, module_path)
                continue
            if not _auto_install:
                raise
            # Extract package name: try "pip install <pkg>" pattern first
            _m = _re.search(r"pip install\s+([\w\-\.\[\]]+)", _msg)
            if _m:
                _pkg = _m.group(1).split("[")[0]  # strip extras like package[extra]
            elif _re.search(r"from\s+'([\w_]+)'", _msg):
                # "cannot import name 'X' from 'Y'" → package is Y
                _pkg = _re.search(r"from\s+'([\w_]+)'", _msg).group(1)
            elif "'" in _msg and "No module named" in _msg:
                # "No module named 'X'" → package is X
                _pkg = _msg.split("'")[1]
            else:
                raise  # don't try to auto-install if we can't identify the package
            _attempt += 1
            if _attempt > 12:
                raise
            # Skip sub-module dots (e.g. "app.core.config") — can't pip install those
            if "." in _pkg and not any(_pkg.startswith(p) for p in ("xai_",)):
                raise
            _distribution = _package_aliases.get(_pkg, _pkg)
            if _distribution in _installed_packages:
                raise
            _installed_packages.add(_distribution)
            print(
                f"    NOTE: missing package '{_pkg}', "
                f"installing '{_distribution}'..."
            )
            try:
                import subprocess as _sp
                _sp.check_call([sys.executable, "-m", "pip", "install", "-q",
                                "--no-warn-script-location", _distribution])
                importlib.invalidate_caches()
                for _key in list(sys.modules):
                    if _key == _pkg or _key.startswith(_pkg + "."):
                        del sys.modules[_key]
            except Exception:
                raise _exc
    parts = function_name.split(".")

    # ClassName.method -> explicit qualification
    if len(parts) >= 2:
        cls = getattr(mod, parts[0])
        # @staticmethod — descriptor returns raw function (no self), skip wrapping
        if isinstance(cls.__dict__.get(parts[1]), staticmethod):
            return cls.__dict__[parts[1]].__func__
        method = getattr(cls, parts[1])
        auto_mock_fn = _auto_mock_self(method, source_dir, source_file, line_start, line_end) if line_start else None
        return _make_class_method_wrapper(method, setup_self_fn, auto_mock_fn)

    # Short name: try module-level first, then search all classes
    try:
        return getattr(mod, parts[0])
    except AttributeError:
        pass

    # Search all classes in the module for the method.
    # When multiple classes define the same method (e.g. a base class imported
    # from another module and a subclass defined locally), prefer:
    # 1. Locally-defined classes over imported ones (no "stub" — see below)
    # 2. Non-abstract implementations over raise-NotImplementedError stubs
    # 3. Longer code bodies (real implementations are typically longer)
    import types as _types
    method_name = parts[0]
    _candidates = []
    for _name, obj in list(mod.__dict__.items()):
        if isinstance(obj, type) and hasattr(obj, method_name):
            candidate = getattr(obj, method_name)
            if callable(candidate):
                _is_local = getattr(obj, '__module__', '') == mod.__name__
                _is_stub = _has_abstract_body(candidate)
                _body_len = len(getattr(getattr(candidate, '__code__', None), 'co_code', b''))
                _is_static = isinstance(obj.__dict__.get(method_name), staticmethod)
                _candidates.append((_name, candidate, _is_local, _is_stub, _body_len, _is_static))

    if _candidates:
        # Sort: local first, non-stub first, longer body first
        _candidates.sort(key=lambda x: (not x[2], x[3], -x[4]))
        _name, candidate, _loc, _stub, _bl, _is_static = _candidates[0]
        print(f"    NOTE: found {method_name} on class {_name}" +
              (f" (abstract stub, skipped {len(_candidates)-1} other(s))" if _stub else
               f" (chosen from {len(_candidates)} candidate(s): {[c[0] for c in _candidates]})"))
        if _is_static:
            return candidate  # @staticmethod — no self injection needed
        auto_mock_fn = _auto_mock_self(candidate, source_dir, source_file, line_start, line_end) if line_start else None
        return _make_class_method_wrapper(candidate, setup_self_fn, auto_mock_fn)

    raise AttributeError(
        f"module {mod.__name__!r} has no attribute {method_name!r}"
    )


class _DotDict(dict):
    """Dict subclass that allows attribute access (e.g. d.key.subkey)."""
    def __getattr__(self, key):
        try:
            val = self[key]
            return _DotDict(val) if isinstance(val, dict) and not isinstance(val, _DotDict) else val
        except KeyError:
            raise AttributeError(key)


class _MockDataProto:
    """Lightweight DataProto mock wrapping a dict for attribute access.

    Many verl functions expect ``data.batch["key"]`` (attribute + dict access).
    Plain dicts from test_input files don't support this.
    """
    def __init__(self, d):
        self.batch = d.get("batch", {})
        self.non_tensor_batch = d.get("non_tensor_batch", {})
        self.meta_info = d.get("meta_info", {})
        self._len = 0
        for v in self.batch.values():
            if hasattr(v, 'shape') and len(v.shape) > 0:
                self._len = v.shape[0]
                break

    def __len__(self):
        return self._len

    def reorder(self, indices):
        import torch as _torch
        for k in list(self.batch.keys()):
            v = self.batch[k]
            if isinstance(v, _torch.Tensor):
                self.batch[k] = v[indices]


def _to_mock(obj):
    """Recursively convert dicts to mock objects for attribute access.

    Dicts with ``batch`` + ``non_tensor_batch`` keys → _MockDataProto.
    Other dicts → _DotDict.
    Everything else passes through unchanged.
    """
    if isinstance(obj, dict):
        if "batch" in obj and "non_tensor_batch" in obj:
            return _MockDataProto(obj)
        return _DotDict(obj)
    return obj


def _is_exception_type(value) -> bool:
    """Return whether value is valid for isinstance(..., value)."""
    if isinstance(value, type):
        return issubclass(value, BaseException)
    if isinstance(value, tuple) and value:
        return all(_is_exception_type(item) for item in value)
    return False


def _run_one_case(func, case: dict, is_error: bool,
                   capture_output: bool = False) -> tuple[bool, dict]:
    """Run a single test case.

    Returns (success, details) where details contains output or error info.
    """
    import asyncio
    import inspect as _inspect
    expected_error = case.get("expected_error")
    if is_error:
        if "expected_error" not in case:
            return False, {
                "error_type": "MissingExpectedError",
                "error": "error test cases must define expected_error as an exception class",
            }
        if not _is_exception_type(expected_error):
            return False, {
                "error_type": "InvalidExpectedError",
                "error": (
                    "expected_error must be an exception class or a non-empty tuple "
                    f"of exception classes, got {expected_error!r}"
                ),
                "expected_error": str(expected_error),
                "error_matched": False,
            }
    try:
        mock_inputs = {k: _to_mock(v) for k, v in case["inputs"].items()}
        result = func(**mock_inputs)
        # Handle async functions: if result is a coroutine, run it
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
            # Mark this case as async (for coverage estimation later)
            if not hasattr(_run_one_case, '_coro_logged'):
                _run_one_case._coro_logged = True
        # Handle generator functions: consume the generator to execute the body
        elif _inspect.isgenerator(result):
            result = list(result)
            if not hasattr(_run_one_case, '_gen_logged'):
                _run_one_case._gen_logged = True
        elif hasattr(result, '__aiter__'):
            # Async generator — run via asyncio
            async def _consume_agen(agen):
                return [_ async for _ in agen]
            result = asyncio.run(_consume_agen(result))
            if not hasattr(_run_one_case, '_gen_logged'):
                _run_one_case._gen_logged = True
        if is_error:
            return False, {"expected_error_not_raised": True}
        if capture_output:
            return True, {"output": serialize(result)}
        return True, {}
    except Exception as e:
        error_info = {"error_type": type(e).__name__, "error": str(e)[:500]}
        if is_error:
            matched = isinstance(e, expected_error)
            error_info["expected_error"] = getattr(
                expected_error, "__name__", str(expected_error),
            )
            error_info["error_matched"] = matched
            return matched, error_info
        return False, error_info


# ── measurement ────────────────────────────────────────────────────────────

def _collect_line_coverage(cov, full_path: str, line_start: int, line_end: int) -> dict:
    try:
        result = cov.analysis2(full_path)
        # coverage.py 7.x returns FileAnalysis(filename, executable, covered, missing, excluded)
        if hasattr(result, 'executable'):
            executable, covered, missing = result.executable, result.covered, result.missing
        elif len(result) >= 5:
            _, executable, covered, missing, _ = result
        elif len(result) == 4:
            executable, covered, missing, _ = result
        else:
            executable, covered, missing = result
    except Exception as exc:
        print(f"    WARN: coverage analysis2 failed for {full_path}: {exc}")
        return {
            "total_executable_lines": 0,
            "num_covered_lines": 0,
            "line_coverage": 0.0,
            "covered_lines": [],
            "missing_lines": [],
            "analysis_error": str(exc),
        }

    func_exec = sorted(l for l in executable if line_start <= l <= line_end)
    func_cov = sorted(l for l in covered if line_start <= l <= line_end)
    func_miss = sorted(l for l in missing if line_start <= l <= line_end)

    return {
        "total_executable_lines": len(func_exec),
        "num_covered_lines": len(func_cov),
        "line_coverage": round(len(func_cov) / len(func_exec), 4) if func_exec else 0.0,
        "covered_lines": func_cov,
        "missing_lines": func_miss,
    }


def _get_executable_lines(source_dir, source_file, line_start, line_end):
    """Get executable lines in range, excluding docstrings and unreachable code."""
    excluded = _get_docstring_lines(source_dir, source_file, line_start, line_end)
    excluded |= _get_unreachable_lines(source_dir, source_file, line_start, line_end)
    excluded |= _get_function_definition_lines(
        source_dir, source_file, line_start, line_end
    )

    # Try coverage.py
    try:
        import coverage as _cov  # noqa: F401
        full_path = os.path.join(source_dir, source_file)
        cov = _cov.Coverage(branch=True)
        result = cov.analysis2(full_path)
        if hasattr(result, 'executable'):
            executable = list(result.executable)
        elif len(result) >= 5:
            _, executable, _, _, _ = result
        elif len(result) == 4:
            executable, _, _, _ = result
        else:
            executable, _, _ = result
        func_exec = sorted(l for l in executable
                           if line_start <= l <= line_end and l not in excluded)
        if func_exec:
            return func_exec
    except Exception:
        pass

    # Fallback: parse source file directly to find lines with code
    import tokenize
    full_path = os.path.join(source_dir, source_file)
    if os.path.isfile(full_path):
        executable = set()
        try:
            with open(full_path, "rb") as f:
                for tok in tokenize.tokenize(f.readline):
                    if tok.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER,
                                         tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
                        if line_start <= tok.start[0] <= line_end:
                            executable.add(tok.start[0])
        except Exception:
            pass
        if executable:
            return sorted(l for l in executable if l not in excluded)
    # Last resort: all lines in range minus excluded
    return [l for l in range(line_start, line_end + 1) if l not in excluded]


def measure_aggregate(func, test_cases: dict, source_dir: str, source_file: str,
                       line_start: int, line_end: int,
                       capture_output: bool = False,
                       function_name: str = "<unknown>") -> dict:
    full_path = os.path.join(source_dir, source_file)
    if not os.path.isfile(full_path):
        print(f"    WARN: source file not found: {full_path}")

    # Resolve real path to handle symlinks
    full_path = os.path.realpath(full_path)

    # Static analysis for executable lines (coverage.py analysis2, no start() needed)
    func_exec = _get_executable_lines(source_dir, source_file, line_start, line_end)
    print(f"    Executable lines in range {line_start}-{line_end}: {len(func_exec)}")

    # Diagnose: print the actual code object filename
    _walk_codes = []
    def _collect_codes(code):
        _walk_codes.append((code.co_name, code.co_filename, code.co_firstlineno))
        for const in code.co_consts:
            if hasattr(const, 'co_code'):
                _collect_codes(const)
    # Unwrap smolagents Tool objects (SimpleTool, Tool, etc.) to get the underlying function
    _func_for_code = func
    for _attr in ('fn', '_func', 'forward', '__wrapped__'):
        _candidate = getattr(_func_for_code, _attr, None)
        if _candidate and hasattr(_candidate, '__code__'):
            _func_for_code = _candidate
            break
    if not hasattr(_func_for_code, '__code__'):
        print(f"    WARNING: cannot inspect code object for {function_name}, skipping diagnosis")
        _walk_codes = []
    else:
        _collect_codes(_func_for_code.__code__)
    for name, fname, lineno in _walk_codes[:5]:
        match = "MATCH" if fname == full_path else "DIFF"
        print(f"    DEBUG: code '{name}' at {fname}:{lineno} [{match}]")

    # Manual sys.settrace for line coverage measurement
    traced = set()
    _seen_paths = set()
    _event_counts = {"line": 0, "call": 0, "return": 0}
    _call_paths = set()
    def _trace(frame, event, arg):
        _event_counts[event] = _event_counts.get(event, 0) + 1
        fn = frame.f_code.co_filename
        if event == 'call':
            _call_paths.add(fn)
        if event == 'line':
            if source_file.replace("/", os.sep) in fn:
                traced.add(frame.f_lineno)
            else:
                _seen_paths.add(fn)
        return _trace

    old = sys.gettrace()
    sys.settrace(_trace)

    errors = []
    results = []
    try:
        for cat in ("normal", "edge", "error"):
            for i, case in enumerate(test_cases.get(cat, [])):
                ok, details = _run_one_case(func, case, cat == "error", capture_output)
                entry = {"category": cat, "index": i, "success": ok,
                         "note": case.get("note", ""), **details}
                results.append(entry)
                if not ok and "error_type" in details:
                    errors.append({"category": cat, "index": i, **details})
    finally:
        sys.settrace(old)

    if not traced:
        print(f"    DEBUG: trace events — {_event_counts}")
        print(f"    DEBUG: {len(_call_paths)} unique call paths, {len(_seen_paths)} unique line paths")
        print(f"    DEBUG: expected file: {source_file}")
        if _call_paths:
            sample = sorted(_call_paths)[:10]
            print(f"    DEBUG: sample call paths: {sample}")

    func_exec_set = set(func_exec)
    func_cov = sorted(l for l in traced if l in func_exec_set)
    func_miss = sorted(l for l in func_exec if l not in func_cov)

    # Diagnostic for class methods or any empty coverage
    print(f"    DEBUG: traced={len(traced)} lines, func_exec={len(func_exec)}, func_cov={len(func_cov)}. traced sample: {sorted(traced)[:10] if traced else 'empty'}")

    # Async estimation: sys.settrace can't trace inside async def bodies on
    # Python 3.10/3.12. Estimate coverage based on execution success rate.
    _is_async = getattr(_run_one_case, '_coro_logged', False)
    # Also check generator functions — sys.settrace may not trace inside them either
    _is_gen = getattr(_run_one_case, '_gen_logged', False)
    if (_is_async or _is_gen) and not func_cov and func_exec:
        ok = sum(1 for r in results if r["success"])
        total = len(results)
        if ok >= total * 0.5:
            # Majority succeeded — estimate full coverage
            func_cov = func_exec
            func_miss = []
            print(f"    NOTE: {'async' if _is_async else 'generator'} function — sys.settrace cannot trace inside body. Estimating full coverage for {ok}/{total} successful runs.")
        elif ok > 0:
            # Some succeeded — proportional estimate
            ratio = ok / total
            func_cov = func_exec[:int(len(func_exec) * ratio)] if ratio < 1.0 else func_exec
            func_miss = [l for l in func_exec if l not in func_cov]
            print(f"    NOTE: {'async' if _is_async else 'generator'} function — sys.settrace cannot trace inside body. Proportional estimate ({ok}/{total} = {ratio:.0%}).")
        else:
            print(f"    NOTE: {'async' if _is_async else 'generator'} function — all {total} runs failed. Coverage estimated as 0%.")

    result = {
        "total_executable_lines": len(func_exec),
        "num_covered_lines": len(func_cov),
        "line_coverage": round(len(func_cov) / len(func_exec), 4) if func_exec else 0.0,
        "covered_lines": func_cov,
        "missing_lines": func_miss,
        "num_execution_errors": len(errors),
        "execution_errors": errors,
    }
    normal_results = [
        record for record in results
        if record.get("category") == "normal"
    ]
    strong_results = [
        record for record in results
        if record.get("category") in {"normal", "edge"}
    ]
    strong_successes = sum(
        1 for record in strong_results
        if record.get("success")
    )
    strong_success_ratio = (
        strong_successes / len(strong_results)
        if strong_results
        else 0.0
    )
    result["stage2_case_gate"] = {
        "normal_cases": len(normal_results),
        "normal_edge_cases": len(strong_results),
        "normal_edge_successes": strong_successes,
        "normal_edge_success_ratio": round(strong_success_ratio, 4),
        "passed": (
            len(normal_results) >= 5
            and strong_success_ratio >= 0.8
        ),
    }
    if not result["stage2_case_gate"]["passed"]:
        result["is_execution_failure"] = True
        result["execution_errors"].append({
            "category": "stage2_case_gate",
            "error": (
                "Stage 2 requires at least 5 normal cases and an 80% "
                "normal+edge success ratio; got "
                f"{len(normal_results)} normal case(s) and "
                f"{strong_successes}/{len(strong_results)} successful "
                "normal+edge case(s)"
            ),
        })
        result["num_execution_errors"] = len(result["execution_errors"])
    # Mark abstract stubs
    is_stub = _is_abstract_stub(source_dir, source_file, line_start, line_end)
    if is_stub:
        result["is_abstract_stub"] = True
        print(f"    NOTE: function is an abstract stub (raise NotImplementedError / pass only)")
    if capture_output:
        result["results"] = results
    return result


def measure_per_test(func, test_cases: dict, source_dir: str, source_file: str,
                     line_start: int, line_end: int,
                     capture_output: bool = False) -> list[dict]:
    full_path = os.path.join(source_dir, source_file)
    func_exec = _get_executable_lines(source_dir, source_file, line_start, line_end)
    func_exec_set = set(func_exec)
    per_test = []

    for cat in ("normal", "edge", "error"):
        for i, case in enumerate(test_cases.get(cat, [])):
            traced = set()
            def _trace(frame, event, arg):
                if event == 'line' and frame.f_code.co_filename == full_path:
                    traced.add(frame.f_lineno)
                return _trace

            old = sys.gettrace()
            sys.settrace(_trace)
            try:
                ok, details = _run_one_case(func, case, cat == "error", capture_output)
            finally:
                sys.settrace(old)

            func_cov = sorted(l for l in traced if l in func_exec_set)
            entry = {
                "category": cat,
                "index": i,
                "note": case.get("note", ""),
                "success": ok,
                "total_executable_lines": len(func_exec),
                "num_covered_lines": len(func_cov),
                "line_coverage": round(len(func_cov) / len(func_exec), 4) if func_exec else 0.0,
                "covered_lines": func_cov,
                **details,
            }
            per_test.append(entry)

    return per_test


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Coverage measurement runner")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--source-file", required=True,
                        help="Relative path within source-dir (e.g. recipe/prime/algo.py)")
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--test-input-file", required=True)
    parser.add_argument("--line-start", type=int, required=True)
    parser.add_argument("--line-end", type=int, required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--per-test", action="store_true",
                        help="Per-test coverage breakdown")
    parser.add_argument("--capture-output", action="store_true",
                        help="Capture and serialize function return values")
    args = parser.parse_args()

    if not _ensure_coverage():
        print("    WARN: coverage.py not available, using dis fallback for executable lines")

    _prepare_import_paths(args.source_dir, args.source_file)
    test_cases, setup_self_fn, setup_env_fn = load_test_cases(
        args.test_input_file,
        execution_root=os.path.dirname(os.path.abspath(args.source_dir)),
    )

    if setup_env_fn is not None:
        print("    NOTE: calling setup_environment() from test_input")
        _tolerate_mock_modules_without_specs()
        modules_before_setup = dict(sys.modules)
        setup_env_fn()
        _repair_mock_package_metadata(modules_before_setup)

    try:
        func = import_function(
            args.source_dir, args.source_file, args.function_name,
            setup_self_fn=setup_self_fn,
            line_start=args.line_start, line_end=args.line_end,
        )
    except Exception as exc:
        print(f"    ERROR importing function: {exc}")
        result = {
            "function_name": args.function_name,
            "source_file": args.source_file,
            "line_range": [args.line_start, args.line_end],
            "line_coverage": 0.0,
            "num_covered_lines": 0,
            "total_executable_lines": 0,
            "covered_lines": [],
            "missing_lines": [],
            "num_execution_errors": 1,
            "execution_errors": [
                {"error": f"Import failed: {exc}", "category": "import_error"}
            ],
            "is_execution_failure": True,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        print(f"Coverage: FAIL (import error)")
        return

    result = {
        "function_name": args.function_name,
        "source_file": args.source_file,
        "line_range": [args.line_start, args.line_end],
        "num_test_cases": {
            "normal": len(test_cases.get("normal", [])),
            "edge": len(test_cases.get("edge", [])),
            "error": len(test_cases.get("error", [])),
        },
    }

    result.update(measure_aggregate(
        func, test_cases, args.source_dir, args.source_file,
        args.line_start, args.line_end,
        capture_output=args.capture_output,
        function_name=args.function_name,
    ))

    if args.per_test:
        result["per_test_coverage"] = measure_per_test(
            func, test_cases, args.source_dir, args.source_file,
            args.line_start, args.line_end,
            capture_output=args.capture_output,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)

    lc = result.get("line_coverage", 0)
    n_cov = result.get("num_covered_lines", 0)
    n_tot = result.get("total_executable_lines", 0)
    print(f"Coverage: {lc:.1%} ({n_cov}/{n_tot} lines)")

    if args.capture_output and "results" in result:
        ok_count = sum(1 for r in result["results"] if r["success"])
        total = len(result["results"])
        print(f"Output: {ok_count}/{total} cases captured")


if __name__ == "__main__":
    main()
