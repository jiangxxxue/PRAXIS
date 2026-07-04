"""Differential-testing grader.

A candidate completion is injected into a fresh copy of the code tree; the
differential runner executes the function on the oracle's test cases; the
results are deep-compared against the oracle. GT and empty-stub behavior is
covered by the same flow (see `_run_differential`'s `completion` argument).
"""
import ast
import json
import math
import re
import shutil
import tempfile
import textwrap
from pathlib import Path

from .config import (
    PROJECT_ROOT,
    get_docker_image,
    oracle_path_for_spec,
    test_input_path,
)
from .docker_runner import run_in_docker


_DIFF_RUNNER_PATH = Path(__file__).parent / "differential_runner.py"

_DOCKER_TIMEOUT_S = 900
_STDERR_TAIL_LIMIT = 4000
_MAX_FAILURES_SHOWN = 5
_RTOL = 1e-5
_ATOL = 1e-8
_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+(?![A-Za-z0-9])")
_MOCK_ID_RE = re.compile(r"<(AsyncMock|MagicMock|Mock)([^>]*) id='[0-9]+'>")
_DISPLAY_SUFFIX_RE = re.compile(r"\s+\([^)]*\)$")
_CLASS_DISPLAY_SUFFIX_RE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<class>[A-Za-z_][A-Za-z0-9_]*)\)$"
)


class _DifferentialError(Exception):
    """Raised when the Docker run couldn't produce usable results."""


def _callable_name(function_name: str) -> str:
    """Map an artifact display name to the Python callable name."""
    match = _CLASS_DISPLAY_SUFFIX_RE.match(function_name)
    if match and "." not in match.group("name"):
        return f"{match.group('class')}.{match.group('name')}"
    return _DISPLAY_SUFFIX_RE.sub("", function_name)


def _spec_callable_name(spec: dict) -> str:
    return _callable_name(spec.get("callable_name") or spec["function_name"])


def grade_completion(spec: dict, completion: str) -> dict:
    """Grade a candidate completion against the stored oracle.

    Returns a dict with keys ``passed``, ``pass_ratio``, ``infra_failure``,
    ``stderr_tail``, ``per_case``. ``per_case`` is a list of per-case records
    (index / category / note / status / failure_reason / inputs / expected / got)
    intended for structured consumption by practice prompts and distill.
    """
    from runner import _parse_impl_location

    rel, _, _ = _parse_impl_location(spec["implementation_location"])
    code_dir = (PROJECT_ROOT / spec["framework"]
                / "test_examples" / spec["example"] / "code")
    test_cases_path = test_input_path(
        spec["framework"],
        spec["example"],
        spec["function_name"],
    )
    oracle_path = oracle_path_for_spec(spec)

    oracle = json.loads(oracle_path.read_text())

    with tempfile.TemporaryDirectory(prefix="grade_result_") as tmp:
        output_path = Path(tmp) / "results.json"
        try:
            candidate = _run_differential(
                code_dir=code_dir,
                rel_file=rel,
                function_name=_spec_callable_name(spec),
                test_cases_path=test_cases_path,
                completion=completion or "",
                output_path=output_path,
                image=get_docker_image(spec["framework"]),
            )
        except _DifferentialError as exc:
            return {
                "passed": False,
                "pass_ratio": 0.0,
                "infra_failure": True,
                "stderr_tail": _truncate(f"INFRA FAILURE: {exc}"),
                "per_case": [],
            }

    comparison = _compare_results(oracle, candidate)
    report = _format_report(oracle, candidate, comparison)
    per_case = _build_per_case(oracle, candidate, comparison)
    return {
        "passed": comparison["passed"],
        "pass_ratio": comparison["pass_ratio"],
        "infra_failure": False,
        "stderr_tail": report,
        "per_case": per_case,
    }


def _build_per_case(oracle: list[dict], candidate: list[dict],
                    comparison: dict) -> list[dict]:
    """Join oracle + candidate + comparison into per-case records.

    Inputs and notes come from the oracle record (produced by differential_runner),
    so this runs purely on host-side JSON — no need to import test_input.py here.
    """
    failure_map = {f["index"]: f["reason"] for f in comparison["failures"]}
    out = []
    n = max(len(oracle), len(candidate))
    for i in range(n):
        o = oracle[i] if i < len(oracle) else None
        c = candidate[i] if i < len(candidate) else None
        src = o or c or {}
        idx = src.get("index", i)
        out.append({
            "index": idx,
            "category": src.get("category", "?"),
            "note": src.get("note", ""),
            "status": "fail" if idx in failure_map else "pass",
            "failure_reason": failure_map.get(idx),
            "inputs": (o or {}).get("inputs", {}),
            "expected": _summarize_outcome(o),
            "got": _summarize_outcome(c),
        })
    return out


def _summarize_outcome(rec: dict | None) -> dict:
    """Collapse a differential_runner record into a compact outcome summary."""
    if rec is None:
        return {"status": "missing"}
    status = rec.get("status")
    if status == "ok":
        return {"status": "ok", "value": rec.get("value")}
    if status == "exception":
        return {
            "status": "exception",
            "exception_type": rec.get("exception_type"),
            "exception_msg": rec.get("exception_msg"),
        }
    if status == "timeout":
        return {"status": "timeout"}
    return {"status": status}


def _run_differential(
    *,
    code_dir: Path,
    rel_file: str,
    function_name: str,
    test_cases_path: Path,
    completion: str | None,
    output_path: Path,
    image: str | None = None,
) -> list[dict]:
    """Execute differential_runner.py inside Docker.

    completion semantics:
      - None: no injection — run GT as-is. Used for oracle generation.
      - "":   re-stub with `raise NotImplementedError`. Used for empty-candidate smoke.
      - text: replace the function definition with `text`.

    Returns the parsed per-case list. Archives a copy of results.json to
    `output_path`.
    """
    code_dir = Path(code_dir)
    test_cases_path = Path(test_cases_path)

    with tempfile.TemporaryDirectory(prefix="diff_ws_") as tmp_str:
        tmp_ws = Path(tmp_str)
        ws_code = tmp_ws / "code"
        shutil.copytree(code_dir, ws_code, symlinks=True)

        target = ws_code / rel_file
        _apply_completion(target, _callable_name(function_name), completion)

        tc_name = f"test_cases{test_cases_path.suffix}"
        shutil.copy2(test_cases_path, tmp_ws / tc_name)
        shutil.copy2(_DIFF_RUNNER_PATH, tmp_ws / "differential_runner.py")

        mounts = [(str(tmp_ws), "/workspace")]
        env = {"PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [
            "python3", "/workspace/differential_runner.py",
            "--source_dir", "/workspace/code",
            "--module_path", rel_file.replace("\\", "/"),
            "--function_name", function_name,
            "--test_cases", f"/workspace/{tc_name}",
            "--output", "/workspace/results.json",
        ]
        cp = run_in_docker(
            cmd,
            mounts=mounts,
            image=image,
            env=env,
            workdir="/workspace",
            timeout=_DOCKER_TIMEOUT_S,
        )

        host_results = tmp_ws / "results.json"
        if not host_results.exists():
            raise _DifferentialError(
                f"differential_runner produced no output (rc={cp.returncode}).\n"
                f"stderr tail:\n{(cp.stderr or '')[-_STDERR_TAIL_LIMIT:]}"
            )

        results = json.loads(host_results.read_text())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(host_results, output_path)
        return results


def _find_target_node(tree: ast.AST, function_name: str):
    """Find a function or method node by simple or dotted name."""
    parts = function_name.split(".")
    if len(parts) == 2:
        class_name, method_name = parts
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name == method_name):
                        return item

        # Keep compatibility with older derived data that sometimes used a
        # module prefix rather than a class prefix.
        target_name = method_name
    else:
        target_name = function_name

    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == target_name):
            return node
    return None


def _apply_completion(target: Path, function_name: str,
                      completion: str | None) -> None:
    """Modify `target` in place to reflect a candidate completion.

    Uses AST to locate the function (or method) by name — walks the whole tree
    so methods nested inside ClassDefs are found too. Preserves the original
    indent level when splicing in the candidate, which matters for methods.
    """
    if completion is None:
        return

    from runner import _stub_one_function

    source = target.read_text()
    tree = ast.parse(source)
    target_node = _find_target_node(tree, function_name)
    if target_node is None:
        raise _DifferentialError(
            f"function {function_name!r} not found in {target}"
        )

    lines = source.splitlines(keepends=True)
    if target_node.decorator_list:
        start = target_node.decorator_list[0].lineno
    else:
        start = target_node.lineno
    end = target_node.end_lineno

    if not completion.strip():
        new_lines = _stub_one_function(lines, start, end)
    else:
        # Candidate is usually written at column 0; if the target sits inside
        # a class, re-indent every non-empty line to match the original def.
        indent = " " * target_node.col_offset
        dedented = textwrap.dedent(completion).rstrip()
        if indent:
            reindented = "\n".join(
                (indent + line) if line.strip() else ""
                for line in dedented.splitlines()
            )
        else:
            reindented = dedented
        completion_text = reindented + "\n"
        new_lines = lines[: start - 1] + [completion_text] + lines[end:]
    target.write_text("".join(new_lines))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_results(oracle: list[dict], candidate: list[dict]) -> dict:
    n = max(len(oracle), len(candidate))
    failures = []
    matches = 0
    for i in range(n):
        o = oracle[i] if i < len(oracle) else None
        c = candidate[i] if i < len(candidate) else None
        ok, reason = _compare_one(o, c)
        if ok:
            matches += 1
        else:
            failures.append({"index": i, "reason": reason,
                             "oracle": o, "candidate": c})
    total = len(oracle) if len(oracle) > 0 else n
    return {
        "passed": len(failures) == 0 and len(candidate) == len(oracle),
        "pass_ratio": matches / max(total, 1),
        "failures": failures,
    }


def _compare_one(oracle, candidate) -> tuple[bool, str]:
    if oracle is None or candidate is None:
        return False, "missing-case"
    o_status = oracle.get("status")
    c_status = candidate.get("status")
    if o_status == "timeout" or c_status == "timeout":
        return False, "timeout"
    if o_status == "exception" and c_status == "exception":
        if oracle.get("exception_type") == candidate.get("exception_type"):
            return True, ""
        return (False, f"exception-type-mismatch: expected "
                       f"{oracle.get('exception_type')}, got "
                       f"{candidate.get('exception_type')}")
    if o_status == "exception" or c_status == "exception":
        return False, f"status-mismatch: oracle={o_status}, candidate={c_status}"
    if o_status == "ok" and c_status == "ok":
        if _deep_equal(oracle.get("value"), candidate.get("value")):
            return True, ""
        return False, "value-mismatch"
    return False, f"unknown-status: oracle={o_status}, candidate={c_status}"


def _deep_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        ta, tb = a.get("__type__"), b.get("__type__")
        if ta != tb:
            return False
        if ta == "tensor":
            if a.get("shape") != b.get("shape"):
                return False
            return _data_close(a.get("data"), b.get("data"))
        if ta == "ndarray":
            if a.get("shape") != b.get("shape") or a.get("dtype") != b.get("dtype"):
                return False
            if "data" in a or "data" in b:
                return "data" in a and "data" in b and _data_close(a.get("data"), b.get("data"))
            return True
        if ta == "tuple":
            items_a = a.get("items", [])
            items_b = b.get("items", [])
            if len(items_a) != len(items_b):
                return False
            return all(_deep_equal(x, y) for x, y in zip(items_a, items_b))
        if ta == "repr":
            return _stable_repr(a.get("value")) == _stable_repr(b.get("value"))
        if ta is None:
            if set(a.keys()) != set(b.keys()):
                return False
            return all(_deep_equal(a[k], b[k]) for k in a)
        return False
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            return _float_close(float(a), float(b))
        return a == b
    return a == b


def _stable_repr(value):
    if not isinstance(value, str):
        return value
    value = _ADDR_RE.sub("<ADDR>", value.replace("0xADDR", "<ADDR>"))
    return _MOCK_ID_RE.sub(r"<\1\2 id='<MOCK_ID>'>", value)


def _data_close(a, b) -> bool:
    """Element-wise tolerance compare for tensor `data` (possibly nested lists)."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_data_close(x, y) for x, y in zip(a, b))
    if isinstance(a, list) or isinstance(b, list):
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return _float_close(float(a), float(b))


def _float_close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=_RTOL, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Report formatting (stderr_tail shown to agent / distill LLM)
# ---------------------------------------------------------------------------

def _format_report(oracle, candidate, comparison) -> str:
    cat_total = {"normal": 0, "edge": 0, "error": 0}
    cat_match = {"normal": 0, "edge": 0, "error": 0}
    failure_indices = {f["index"] for f in comparison["failures"]}
    for i, o in enumerate(oracle):
        cat = o.get("category", "normal")
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if i not in failure_indices:
            cat_match[cat] = cat_match.get(cat, 0) + 1

    total = len(oracle)
    matches = total - len(comparison["failures"])
    parts = [
        f"DIFFERENTIAL TEST: {matches}/{total} passed "
        f"(normal: {cat_match['normal']}/{cat_total['normal']}, "
        f"edge: {cat_match['edge']}/{cat_total['edge']}, "
        f"error: {cat_match['error']}/{cat_total['error']})",
    ]

    for f in comparison["failures"][:_MAX_FAILURES_SHOWN]:
        i = f["index"]
        o = f["oracle"] or {}
        c = f["candidate"] or {}
        cat = o.get("category") or c.get("category") or "?"
        note = o.get("note") or c.get("note") or ""
        header = f"\nFAILED case {i} [{cat}]"
        if note:
            header += f" — {note}"
        parts.append(f"{header} ({f['reason']}):")
        parts.append(f"  Expected: {_fmt_outcome(o)}")
        parts.append(f"  Got:      {_fmt_outcome(c)}")

    if len(comparison["failures"]) > _MAX_FAILURES_SHOWN:
        parts.append(f"\n(showing first {_MAX_FAILURES_SHOWN} failures)")

    return _truncate("\n".join(parts))


def _fmt_outcome(rec: dict) -> str:
    status = rec.get("status", "?")
    if status == "exception":
        return rec.get("exception_type", "Exception")
    if status == "timeout":
        return "<timeout>"
    if status == "ok":
        return _fmt_value(rec.get("value"))
    return f"<{status}>"


def _fmt_value(v) -> str:
    if isinstance(v, dict) and v.get("__type__") == "tensor":
        shape = v.get("shape") or []
        dtype = v.get("dtype", "?")
        numel = 1
        for d in shape:
            numel *= d
        if numel <= 16:
            return f"tensor({v.get('data')!r})"
        return f"tensor(shape={shape}, dtype={dtype})"
    if isinstance(v, dict) and v.get("__type__") == "ndarray":
        shape = v.get("shape") or []
        dtype = v.get("dtype", "?")
        if "data" in v:
            return f"ndarray({v.get('data')!r}, dtype={dtype})"
        return f"ndarray(shape={shape}, dtype={dtype})"
    if isinstance(v, dict) and v.get("__type__") == "tuple":
        return f"({', '.join(_fmt_value(x) for x in v.get('items', []))})"
    if isinstance(v, list):
        if len(v) <= 6:
            return repr(v)
        return f"{v[:6]!r}... (+{len(v) - 6})"
    return repr(v)


def _truncate(text: str) -> str:
    if len(text) <= _STDERR_TAIL_LIMIT:
        return text
    return text[-_STDERR_TAIL_LIMIT:]
