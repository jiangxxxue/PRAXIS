#!/usr/bin/env python3
"""Self-contained Docker-side script: import target function, run test cases,
serialize results.

Invoked by grader.py as:

    python3 differential_runner.py \\
      --source_dir /workspace/code \\
      --module_path verl/trainer/ppo/core_algos.py \\
      --function_name agg_loss \\
      --test_cases /workspace/test_input.py \\
      --output /workspace/results.json

test_input.py must define a module-level ``test_cases`` dict with keys
``normal`` / ``edge`` / ``error``. Each case is:

    {"inputs": kwargs, "note": str}

If test_input.py exports ``get_callable()``, its return value is used as the
target callable instead of ``getattr(module, function_name)`` — this is how
methods (and any callable needing construction/binding) are supplied.

No imports from the memory package — this file is copied into the per-grade
workspace and executed in isolation.
"""
import argparse
import asyncio
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import re
import signal
import sys
import time
import traceback
import types
from pathlib import Path


# Forbid bytecode caches inside the source tree. Prevents a grading run from
# caching an old implementation's .pyc and a later run picking it up.
sys.dont_write_bytecode = True


_PER_CASE_TIMEOUT_S = 30
_MAX_STRUCTURED_ARRAY_VALUES = 16
_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+(?![A-Za-z0-9])")
_DISPLAY_SUFFIX_RE = re.compile(r"\s+\([^)]*\)$")
_CLASS_DISPLAY_SUFFIX_RE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<class>[A-Za-z_][A-Za-z0-9_]*)\)$"
)
_STUB_MODULE_PREFIXES = {
    "agno",
    "aiohttp",
    "aiofiles",
    "app",
    "chartographer",
    "chromadb",
    "colored",
    "deepscaler",
    "diffusers",
    "docx",
    "firebase_admin",
    "fire",
    "firecrawl",
    "fitz",
    "ddgs",
    "general_perf",
    "google",
    "google.cloud.storage",
    "google.cloud.exceptions",
    "google.cloud.aiplatform",
    "google.cloud.aiplatform_v1beta1",
    "google.genai",
    "google.generativeai",
    "hydra",
    "inference",
    "langchain_chroma",
    "langchain_community",
    "langchain_core",
    "langchain_openai",
    "langchain_text_splitters",
    "langchain",
    "latex2sympy2",
    "lightning",
    "lightning.fabric.plugins.precision",
    "lightrag",
    "load_runstep",
    "llama_parse",
    "mammoth",
    "markdownify",
    "markdown",
    "matplotlib",
    "mcp.server.fastmcp",
    "megatron",
    "mlir_tensorrt",
    "minio",
    "mysql",
    "modelopt",
    "numpy.lib.function_base",
    "omegaconf",
    "onnx",
    "nvtripy",
    "onnxruntime",
    "open_r1",
    "onnx2torch",
    "opt_tf",
    "pathvalidate",
    "poprt",
    "pptx",
    "puremagic",
    "raganything",
    "sandbox_fusion",
    "serpapi",
    "sentence_transformers",
    "smolagents",
    "speech_recognition",
    "sqlalchemy",
    "streamlit",
    "tensorflow",
    "tensorflow.core",
    "tenacity",
    "tensorrt",
    "tiktoken",
    "tomllib",
    "timeout_decorator",
    "utils",
    "vllm",
    "verl",
    "vertexai",
    "wikipediaapi",
    "word2number",
    "wolframalpha",
    "xai_sdk",
    "youtube_transcript_api",
}


class _CaseTimeout(Exception):
    pass


class _DummyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return cls


class _Dummy(metaclass=_DummyMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "__all__":
            return []
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Dummy
        setattr(self, name, value)
        return value


class _StubModuleLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Create lightweight modules for optional heavy deps missing in Docker.

    The runner only uses this for known external dependencies observed in the
    practice artifacts. Real modules still win when they are importable from the
    code tree or Docker image.
    """

    def find_spec(self, fullname, path=None, target=None):
        if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
            return None
        if _should_stub_module(fullname):
            return importlib.machinery.ModuleSpec(
                fullname, self, is_package=True
            )
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__file__ = "<stub {}>".format(spec.name)
        module.__package__ = spec.name
        module.__path__ = []
        return module

    def exec_module(self, module):
        return None


def _should_stub_module(fullname: str) -> bool:
    return any(
        fullname == prefix or fullname.startswith(prefix + ".")
        for prefix in _STUB_MODULE_PREFIXES
    )


def _install_stub_importer() -> None:
    if not any(isinstance(x, _StubModuleLoader) for x in sys.meta_path):
        sys.meta_path.insert(0, _StubModuleLoader())


def _on_alarm(signum, frame):
    raise _CaseTimeout("per-case timeout")


def _module_path_to_dotted(module_path: str) -> str:
    """`verl/trainer/ppo/core_algos.py` -> `verl.trainer.ppo.core_algos`."""
    p = module_path.replace("\\", "/").strip("/")
    if p.endswith(".py"):
        p = p[:-3]
    return p.replace("/", ".")


def _callable_name(function_name: str) -> str:
    """Map an artifact display name to the Python callable name."""
    match = _CLASS_DISPLAY_SUFFIX_RE.match(function_name)
    if match and "." not in match.group("name"):
        return "{}.{}".format(match.group("class"), match.group("name"))
    return _DISPLAY_SUFFIX_RE.sub("", function_name)


def _add_import_roots(source_dir: str, module_path: str) -> None:
    source = Path(source_dir)
    module_file = source / module_path.replace("\\", "/")
    roots = [
        source,
        source / "src",
        module_file.parent,
    ]
    if module_file.exists():
        for parent in module_file.parents:
            if parent == source:
                break
            roots.append(parent)
    for root in roots:
        if not root.exists():
            continue
        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def _load_module(source_dir: str, module_path: str):
    dotted = _module_path_to_dotted(module_path)
    module_file = Path(source_dir) / module_path.replace("\\", "/")
    try:
        module = importlib.import_module(dotted)
        loaded_file = getattr(module, "__file__", None)
        if module_file.exists() and loaded_file:
            try:
                if Path(loaded_file).resolve() != module_file.resolve():
                    raise ImportError(
                        "{!r} resolved to {}, not {}".format(
                            dotted, loaded_file, module_file
                        )
                    )
            except OSError:
                raise ImportError(
                    "{!r} resolved to {}, not {}".format(
                        dotted, loaded_file, module_file
                    )
                )
        return module
    except Exception as import_exc:
        if not module_file.exists():
            raise
        try:
            _ensure_parent_packages(source_dir, dotted)
            sys.modules.pop(dotted, None)
            spec = importlib.util.spec_from_file_location(dotted, module_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            raise import_exc


def _ensure_parent_packages(source_dir: str, dotted: str) -> None:
    source = Path(source_dir)
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name in sys.modules:
            continue
        pkg_path = source.joinpath(*parts[:i])
        if not pkg_path.is_dir():
            continue
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(pkg_path)]
        pkg.__package__ = name
        sys.modules[name] = pkg


def _load_test_input(test_input_path: str):
    """Import test_input.py and return (module, test_cases, get_callable_or_None)."""
    import unittest.mock

    spec = importlib.util.spec_from_file_location(
        "_pilot_test_input", test_input_path
    )
    module = importlib.util.module_from_spec(spec)
    module.MagicMock = unittest.mock.MagicMock
    module.AsyncMock = unittest.mock.AsyncMock
    module.mock = unittest.mock
    spec.loader.exec_module(module)
    test_cases = getattr(module, "test_cases", None)
    if not isinstance(test_cases, dict):
        raise RuntimeError(
            "{} must define `test_cases: dict[str, list[dict]]`".format(
                test_input_path
            )
        )
    for key in ("normal", "edge", "error"):
        if key not in test_cases:
            raise RuntimeError("test_cases missing key: {!r}".format(key))
    get_callable = getattr(module, "get_callable", None)
    return module, test_cases, get_callable


class _DotDict(dict):
    """Dict subclass that allows attribute access for config-like inputs."""

    def __getattr__(self, key):
        try:
            val = self[key]
            return _DotDict(val) if isinstance(val, dict) and not isinstance(val, _DotDict) else val
        except KeyError:
            raise AttributeError(key)


def _call_setup_self(setup_self, config=None):
    if config is None:
        try:
            return setup_self()
        except TypeError:
            return setup_self(config=None)
    return setup_self(config=config)


def _fallback_self(config=None):
    """Build a lightweight receiver for class methods when test_input omits one."""

    from unittest.mock import MagicMock

    mock_self = MagicMock()
    mock_self.config = _DotDict(config) if isinstance(config, dict) else config
    return mock_self


def _resolve_callable(module, function_name: str, test_module=None):
    """Resolve a module function or a supported class constructor.

    Instance methods prefer test_input.py's setup_self(), but fall back to a
    MagicMock receiver to match coverage_runner's auto-mock behavior.
    """
    function_name = _callable_name(function_name)
    fn = getattr(module, function_name, None)
    if fn is not None:
        return fn

    if "." not in function_name:
        return None

    class_name, method_name = function_name.split(".", 1)
    cls = getattr(module, class_name, None)
    if cls is None:
        return None

    if method_name == "__init__":
        def _constructor(**kwargs):
            cls(**kwargs)
            return None

        return _constructor

    method = getattr(cls, method_name, None)
    if method is None:
        return None
    setup_self = getattr(test_module, "setup_self", None) if test_module else None
    if setup_self is not None:
        return method.__get__(_call_setup_self(setup_self), cls)
    return method.__get__(_fallback_self(), cls)


def _resolve_case_callable(module, function_name: str, test_module, inputs: dict):
    """Resolve a callable for a single case.

    Some generated artifacts put `self` or constructor `config` in each case's
    inputs. For instance methods, use those per-case values to bind the method
    and remove them from the keyword arguments passed to the actual method.
    """
    callable_name = _callable_name(function_name)
    if "." not in callable_name:
        return None, inputs

    class_name, method_name = callable_name.split(".", 1)
    if method_name == "__init__":
        return None, inputs

    cls = getattr(module, class_name, None)
    if cls is None:
        return None, inputs
    method = getattr(cls, method_name, None)
    if method is None:
        return None, inputs

    call_inputs = dict(inputs)
    explicit_self = call_inputs.pop("self", None)
    config = call_inputs.pop("config", None)
    setup_self = getattr(test_module, "setup_self", None) if test_module else None
    if explicit_self is not None:
        receiver = explicit_self
    elif setup_self is not None:
        receiver = _call_setup_self(setup_self, config)
    else:
        receiver = _fallback_self(config)
    return method.__get__(receiver, cls), call_inputs


def _flatten(cases: dict):
    """Flatten dict-of-case-lists into [(index, category, case_dict), ...]."""
    out = []
    idx = 0
    for cat in ("normal", "edge", "error"):
        for case in cases.get(cat, []):
            out.append((idx, cat, case))
            idx += 1
    return out


def _is_tensor(x) -> bool:
    """Duck-typed torch.Tensor check that doesn't import torch eagerly."""
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(x, torch.Tensor)


def _is_numpy_array(x) -> bool:
    """Duck-typed numpy.ndarray check that avoids importing numpy eagerly."""
    return (
        type(x).__module__.startswith("numpy")
        and hasattr(x, "shape")
        and hasattr(x, "dtype")
        and hasattr(x, "tolist")
    )


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        try:
            total *= int(dim)
        except (TypeError, ValueError):
            return _MAX_STRUCTURED_ARRAY_VALUES + 1
    return total


def _serialize(value):
    """Recursively serialize a Python value to JSON-safe form."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if _is_tensor(value):
        return {
            "__type__": "tensor",
            "data": value.detach().cpu().tolist(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if _is_numpy_array(value):
        shape = [int(dim) for dim in value.shape]
        out = {
            "__type__": "ndarray",
            "shape": shape,
            "dtype": str(value.dtype),
        }
        if _numel(shape) <= _MAX_STRUCTURED_ARRAY_VALUES:
            out["data"] = value.tolist()
        return out
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_serialize(v) for v in value]}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    return {"__type__": "repr", "value": _ADDR_RE.sub("<ADDR>", repr(value))}


def _run_one(fn, case: dict, module=None, function_name=None, test_module=None,
             allow_case_binding=False) -> dict:
    inputs = case.get("inputs", {})
    case_fn = fn
    if allow_case_binding:
        resolved, resolved_inputs = _resolve_case_callable(
            module, function_name, test_module, inputs
        )
        if resolved is not None:
            case_fn = resolved
            inputs = resolved_inputs
    note = case.get("note", "") or ""
    inputs_serialized = {k: _serialize(v) for k, v in inputs.items()}

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(_PER_CASE_TIMEOUT_S)
    t0 = time.monotonic()
    try:
        result = case_fn(**inputs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        wall_ms = int((time.monotonic() - t0) * 1000)
        rec = {"status": "ok", "value": _serialize(result), "wall_ms": wall_ms}
    except _CaseTimeout:
        wall_ms = int((time.monotonic() - t0) * 1000)
        rec = {"status": "timeout", "wall_ms": wall_ms}
    except Exception as exc:
        wall_ms = int((time.monotonic() - t0) * 1000)
        rec = {
            "status": "exception",
            "exception_type": type(exc).__name__,
            "exception_msg": str(exc),
            "wall_ms": wall_ms,
        }
    finally:
        signal.alarm(0)

    rec["note"] = note
    rec["inputs"] = inputs_serialized
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--module_path", required=True)
    ap.add_argument("--function_name", required=True)
    ap.add_argument("--test_cases", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    source_dir = str(Path(args.source_dir).resolve())
    _add_import_roots(source_dir, args.module_path)
    _install_stub_importer()

    test_module, test_cases, get_callable = _load_test_input(args.test_cases)
    setup_environment = getattr(test_module, "setup_environment", None)
    if setup_environment is not None:
        setup_environment()

    module = None
    if get_callable is not None:
        fn = get_callable()
        if fn is None:
            raise RuntimeError(
                "get_callable() in {} returned None".format(args.test_cases)
            )
    else:
        dotted = _module_path_to_dotted(args.module_path)
        try:
            module = _load_module(source_dir, args.module_path)
        except Exception:
            sys.stderr.write(
                "[differential_runner] failed to import {!r} from {}\n".format(
                    dotted, source_dir
                )
            )
            traceback.print_exc(file=sys.stderr)
            raise

        fn = _resolve_callable(module, args.function_name, test_module)
        if fn is None and "." not in _callable_name(args.function_name):
            setup_self = getattr(test_module, "setup_self", None)
            matches = []
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if not hasattr(cls, args.function_name):
                    continue
                if getattr(cls, "__module__", None) != module.__name__:
                    continue
                method = getattr(cls, args.function_name)
                method_module = getattr(method, "__module__", module.__name__)
                if method_module != module.__name__:
                    continue
                try:
                    params = list(inspect.signature(method).parameters)
                except (TypeError, ValueError):
                    params = []
                if params and params[0] in {"self", "cls"}:
                    if setup_self is not None:
                        matches.append(method.__get__(setup_self(), cls))
                else:
                    matches.append(method)
            if len(matches) == 1:
                fn = matches[0]
        if fn is None:
            raise RuntimeError(
                "function {!r} not found in module {!r} "
                "(and test_input.py did not export get_callable)".format(
                    args.function_name, dotted
                )
            )

    flat = _flatten(test_cases)
    results = []
    for idx, category, case in flat:
        rec = _run_one(
            fn,
            case,
            module=module,
            function_name=args.function_name,
            test_module=test_module,
            allow_case_binding=get_callable is None,
        )
        rec["index"] = idx
        rec["category"] = category
        results.append(rec)

    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
