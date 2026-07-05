"""Prompt construction helpers for the practice loop.

Also exposes ``format_failed_cases`` / ``format_case_block`` which distill
reuses to render the same per-case information.
"""
import math


_MAX_CASES_SHOWN = 5
_COMPACT_ELEMS = 16
_MAX_REPR_CHARS = 512


def build_practice_prompt(base_prompt: str, prior_attempts: list) -> str:
    """Append PREVIOUS_ATTEMPTS section to base prompt, if any."""
    if not prior_attempts:
        return base_prompt
    sections = []
    for i, att in enumerate(prior_attempts):
        if att.get("infra_failure"):
            result_label = "INFRA_FAILURE (not a code problem — infrastructure crashed)"
        elif att.get("passed"):
            result_label = "PASS"
        else:
            ratio = att.get("pass_ratio", 0.0)
            result_label = f"FAIL ({ratio:.0%} passed)"
        parts = [
            f"--- Attempt {i} ---",
            f"Code:\n```python\n{att['completion']}\n```",
            f"Result: {result_label}",
        ]
        failed = [c for c in att.get("per_case", []) if c.get("status") == "fail"]
        if failed:
            parts.append(format_failed_cases(failed))
        sections.append("\n".join(parts))
    return base_prompt + "\n\nPREVIOUS_ATTEMPTS:\n" + "\n".join(sections)


def format_failed_cases(cases: list, max_shown: int = _MAX_CASES_SHOWN) -> str:
    """Render a list of failing per-case records for prompt injection."""
    n = len(cases)
    header = (f"\nFAILED CASES ({n} total):" if n <= max_shown
              else f"\nFAILED CASES ({n} total, showing {max_shown}):")
    blocks = [format_case_block(c) for c in cases[:max_shown]]
    if n > max_shown:
        blocks.append(f"  ... (+{n - max_shown} more)")
    return header + "\n" + "\n".join(blocks)


def format_case_block(case: dict) -> str:
    """Render a single per-case record: category, note, reason, inputs, expected vs got."""
    reason = case.get("failure_reason") or "?"
    header = f"  Case {case['index']} [{case.get('category', '?')}]"
    note = case.get("note") or ""
    if note:
        header += f" — {note}"
    lines = [f"{header} ({reason}):", "    Input:"]
    for k, v in (case.get("inputs") or {}).items():
        lines.append(f"      {k} = {_fmt_input(v)}")
    lines.append(f"    Expected: {_fmt_outcome(case.get('expected', {}))}")
    lines.append(f"    Got:      {_fmt_outcome(case.get('got', {}))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Value formatters                                                            #
# --------------------------------------------------------------------------- #

def _fmt_input(v):
    """Compact repr for kwarg values; summarize large tensors/lists by shape."""
    if isinstance(v, dict) and v.get("__type__") in {"tensor", "tuple", "repr"}:
        return _fmt_value(v)
    if isinstance(v, list):
        if any(isinstance(x, dict) and x.get("__type__") in {"tensor", "tuple", "repr"}
               for x in v):
            shown = ", ".join(_fmt_value(x) for x in v[:3])
            suffix = f", ... (+{len(v) - 3})" if len(v) > 3 else ""
            return "[" + shown + suffix + "]"
        shape = _infer_shape(v)
        if shape is None:
            return _clip_repr(v)
        numel = math.prod(shape) if shape else 1
        if numel <= _COMPACT_ELEMS:
            return _clip_repr(v)
        dtype = "float" if _has_float(v) else "int"
        preview = _head_preview(v, 2)
        return f"<list shape={list(shape)} dtype={dtype} preview={preview}>"
    return _clip_repr(v)


def _infer_shape(v):
    if not isinstance(v, list):
        return ()
    if not v:
        return (0,)
    inner = [_infer_shape(x) for x in v]
    if any(s is None for s in inner):
        return None
    first = inner[0]
    if not all(s == first for s in inner):
        return None
    return (len(v),) + first


def _has_float(v):
    if isinstance(v, float):
        return True
    if isinstance(v, list):
        return any(_has_float(x) for x in v)
    return False


def _head_preview(v, n=2):
    if not isinstance(v, list):
        return repr(v)
    head = v[:n]
    if all(not isinstance(x, list) for x in head):
        rendered = repr(head)
        return rendered[:-1] + ", ...]" if len(v) > n else rendered
    rendered_head = [_head_preview(x, n) for x in head]
    suffix = ", ..." if len(v) > n else ""
    return "[" + ", ".join(rendered_head) + suffix + "]"


def _fmt_outcome(outcome: dict) -> str:
    status = outcome.get("status")
    if status == "ok":
        return _fmt_value(outcome.get("value"))
    if status == "exception":
        t = outcome.get("exception_type", "Exception")
        msg = outcome.get("exception_msg") or ""
        return f"{t}: {msg}" if msg else t
    if status == "timeout":
        return "<timeout>"
    if status == "missing":
        return "<missing>"
    return f"<{status}>"


def _fmt_value(v):
    if isinstance(v, dict) and v.get("__type__") == "tensor":
        shape = v.get("shape") or []
        data = v.get("data")
        numel = math.prod(shape) if shape else 1
        if numel <= 8:
            return f"tensor({data!r})"
        dtype = v.get("dtype", "?")
        return f"tensor(shape={list(shape)}, dtype={dtype})"
    if isinstance(v, dict) and v.get("__type__") == "repr":
        value = v.get("value", "")
        return _clip_text(str(value))
    if isinstance(v, dict) and v.get("__type__") == "tuple":
        items = v.get("items", [])
        return "(" + ", ".join(_fmt_value(x) for x in items) + ")"
    if isinstance(v, list):
        if len(v) <= 6:
            return _clip_repr(v)
        return f"{_clip_repr(v[:6])}... (+{len(v) - 6})"
    return _clip_repr(v)


def _clip_repr(v, max_chars: int = _MAX_REPR_CHARS) -> str:
    return _clip_text(repr(v), max_chars=max_chars)


def _clip_text(text: str, max_chars: int = _MAX_REPR_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"
