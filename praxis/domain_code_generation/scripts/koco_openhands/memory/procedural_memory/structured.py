"""Structured procedural-memory generation and indexing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from agent.sdk import _resolve_llm_model
from runner import _extract_function_from_file, _parse_impl_location

from .config import (
    PROJECT_ROOT,
    REPO_ROOT,
    output_profile_for_set,
    structured_knowledge_path,
    structured_per_function_dir,
    structured_per_function_path,
    trace_path_for_spec,
)
from .grader import _spec_callable_name
from .prompts import format_failed_cases


def distill_structured_trace(trace, *, model, api_key, base_url,
                             provider=None, api_version=None):
    """Distill one practice trace into structured JSONL memory entries."""
    spec = trace["spec"]
    gt_source = _ground_truth_source_for_spec(spec)

    system_msg = _structured_distill_system()
    user_msg = _build_structured_distill_user_message(trace, gt_source)
    entries = _call_json_entries(
        system_msg,
        user_msg,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        api_version=api_version,
        max_tokens=8192,
    )
    entries = [
        _normalize_entry(entry, spec, idx)
        for idx, entry in enumerate(entries, 1)
        if _entry_has_signal(entry)
    ]

    out_path = structured_per_function_path(
        spec["example"],
        spec["function_name"],
        spec["derived_set"],
        framework=spec.get("framework"),
    )
    _write_jsonl(out_path, entries)
    print(f"distill-structured: wrote {len(entries)} entry(ies) to {out_path}")
    return entries


def consolidate_structured_example(example, *, derived_set, framework=None):
    """Merge per-function structured entries into one profile/example JSONL."""
    pf_dir = structured_per_function_dir(example, derived_set, framework)
    entries = []
    if pf_dir.exists():
        for path in sorted(pf_dir.glob("*.jsonl")):
            entries.extend(_read_jsonl(path))

    entries = _dedupe_entries(entries)
    out_path = structured_knowledge_path(example, derived_set, framework)
    _write_jsonl(out_path, entries)
    print(f"consolidate-structured: wrote {len(entries)} entry(ies) to {out_path}")
    return entries


def _structured_distill_system():
    return """\
You distill practice attempts into structured PracticeMemory entries.

Return strict JSON only:

{
  "entries": [
    {
      "trigger": "When this memory applies: concrete code pattern, API use, shape contract, error contract, or framework situation.",
      "content": "The transferable implementation constraint or convention. Include the anti-pattern and correct pattern when useful.",
      "evidence": {
        "key_diff": "Grounded evidence from iter labels, case_idx labels, or GT design. Mention source behavior and what changed.",
        "passed_iterations": [0],
        "failed_iterations": [1]
      },
      "confidence": {
        "score": 0.0,
        "rationale": "Brief explanation of why this knowledge deserves this reliability score."
      }
    }
  ]
}

Rules:
- Emit only genuinely transferable lessons. If there are none, return {"entries":[]}.
- Do not include Markdown fences or commentary.
- trigger must be specific enough that a future agent can decide whether to apply the memory.
- evidence must cite iter/case_idx labels or the GT design, not vague claims.
- confidence.score is the final reliability estimate for this extracted knowledge
  entry, not the practice pass rate and not confidence in your wording.
- confidence.rationale must briefly identify the strongest supporting evidence
  and the main remaining uncertainty behind the score.
- The score controls how much a future system should trust and use the knowledge.
  Knowledge below the downstream confidence threshold may be withheld entirely,
  so score conservatively and do not inflate uncertain lessons.
- Judge reliability from all of the following together:
  1. factual correctness against the ground-truth implementation;
  2. direct support from the cited attempts, case results, and key differences;
  3. precision of the trigger about when the knowledge applies;
  4. consistency across attempts, especially whether a successful implementation
     validates the claimed lesson.
- A PASS does not automatically make every inferred lesson reliable. A failed
  attempt can support an anti-pattern only when the failure evidence clearly
  establishes the connection. Generic advice, style preferences, speculation,
  or claims that go beyond the GT and trace must receive low confidence or be
  omitted.
- Calibrate confidence.score in [0, 1]:
  - 0.90-1.00: directly and unambiguously supported by GT and strong trace evidence;
    safe to trust when the trigger matches.
  - 0.75-0.89: reliable and well supported, with only minor uncertainty.
  - 0.60-0.74: useful but supported by limited evidence or a somewhat broad trigger;
    apply with caution.
  - 0.40-0.59: materially uncertain, weakly supported, or difficult to generalize;
    future use should be cautious and it may be filtered out.
  - 0.00-0.39: unsupported, contradicted, speculative, or not meaningfully
    transferable; normally omit the entry instead of emitting it.
"""


def _build_structured_distill_user_message(trace, gt_source):
    spec = trace["spec"]
    parts = [
        f"Function: {spec['function_name']}",
        f"Framework: {spec['framework']}",
        f"Example: {spec['example']}",
        f"Profile: {spec['derived_set']}",
        "\nGround-truth implementation:"
        f"\n```python\n{gt_source}\n```",
        f"\n{len(trace['iterations'])} practice attempts follow. "
        "`iter N` is the iteration index; `case_idx` values are stable.\n",
    ]
    for idx, it in enumerate(trace["iterations"]):
        grade = it.get("grade", {})
        if grade.get("infra_failure"):
            status = "INFRA_FAILURE"
        elif grade.get("passed"):
            status = "PASS"
        else:
            status = f"FAIL ({grade.get('pass_ratio', 0.0):.0%} passed)"
        block = [f"--- iter {idx} ---", f"Status: {status}"]
        per_case = grade.get("per_case") or []
        if per_case:
            block.append(
                "Passing case_idx: "
                + repr([c["index"] for c in per_case if c.get("status") == "pass"])
            )
            block.append(
                "Failing case_idx: "
                + repr([c["index"] for c in per_case if c.get("status") == "fail"])
            )
        block.append(f"Code:\n```python\n{it.get('completion', '')}\n```")
        failed = [c for c in per_case if c.get("status") == "fail"]
        if failed:
            block.append(format_failed_cases(failed))
        parts.append("\n".join(block))
    return "\n".join(parts)


def _call_json_entries(system_msg, user_msg, *, model, api_key, base_url,
                       max_tokens, provider=None, api_version=None):
    import litellm

    kwargs = {
        "model": _resolve_llm_model(model, base_url),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "api_key": api_key,
        "api_base": base_url,
    }
    if api_version:
        kwargs["api_version"] = api_version
    resp = litellm.completion(**kwargs)
    content = (resp.choices[0].message.content or "").strip()
    payload = _parse_json_payload(content)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise RuntimeError("structured distill returned non-list entries")
    return entries


def _parse_json_payload(content):
    candidates = [content]
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match and match.group(0) != content:
        candidates.append(match.group(0))

    last_error = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    try:
        from json_repair import loads as repair_json_loads
    except ImportError:
        if last_error is not None:
            raise last_error
        raise

    for candidate in candidates:
        try:
            return repair_json_loads(candidate)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("structured distill returned empty JSON content")


def _normalize_entry(entry, spec, idx):
    trigger = str(entry.get("trigger", "")).strip()
    content = str(entry.get("content", "")).strip()
    evidence_in = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
    confidence_in = entry.get("confidence") if isinstance(entry.get("confidence"), dict) else {}

    passed, failed = _iteration_status_counts(spec)
    score = _normalize_confidence_score(confidence_in.get("score"))

    normalized = {
        "id": _entry_id(
            output_profile_for_set(spec["derived_set"]),
            spec["example"],
            spec["function_name"],
            trigger,
            content,
        ),
        "profile": output_profile_for_set(spec["derived_set"]),
        "framework": spec["framework"],
        "example": spec["example"],
        "source_function": spec["function_name"],
        "callable_name": _spec_callable_name(spec),
        "implementation_location": spec["implementation_location"],
        "trigger": trigger,
        "content": content,
        "evidence": {
            "trace_path": _repo_relative_str(trace_path_for_spec(spec)),
            "passed_iterations": _int_list(evidence_in.get("passed_iterations")),
            "failed_iterations": _int_list(evidence_in.get("failed_iterations")),
            "key_diff": str(evidence_in.get("key_diff", "")).strip(),
        },
        "confidence": {
            "practice_attempts": passed + failed,
            "passes": passed,
            "failures": failed,
            "score": score,
            "rationale": str(confidence_in.get("rationale", "")).strip(),
        },
    }
    if not normalized["evidence"]["passed_iterations"]:
        normalized["evidence"]["passed_iterations"] = _passed_iterations_from_trace_path(
            trace_path_for_spec(spec)
        )
    if not normalized["evidence"]["failed_iterations"]:
        normalized["evidence"]["failed_iterations"] = _failed_iterations_from_trace_path(
            trace_path_for_spec(spec)
        )
    return normalized


def _normalize_confidence_score(value):
    """Return a conservative unit score for model-produced confidence."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return 0.0
    return round(max(0.0, min(1.0, float(value))), 4)


def _ground_truth_source_for_spec(spec):
    location = str(spec.get("implementation_location") or "").strip()
    rel, _, _ = _parse_impl_location(location)
    if not rel:
        raise ValueError(f"cannot parse implementation_location: {location!r}")

    code_root = (
        PROJECT_ROOT / spec["framework"] / "test_examples" / spec["example"] / "code"
    ).resolve()
    source_path = (code_root / rel).resolve()
    try:
        source_path.relative_to(code_root)
    except ValueError as exc:
        raise ValueError(
            f"implementation_location escapes the example code directory: {location!r}"
        ) from exc

    gt_source = _extract_function_from_file(
        str(source_path),
        _spec_callable_name(spec),
    )
    if not gt_source:
        raise ValueError(
            "cannot extract GT source for "
            f"{spec.get('framework')}/{spec.get('example')}/"
            f"{spec.get('function_name')}"
        )
    return gt_source


def _repo_relative_str(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _iteration_status_counts(spec):
    trace_path = trace_path_for_spec(spec)
    trace = json.loads(trace_path.read_text())
    passed = 0
    failed = 0
    for it in trace.get("iterations", []):
        grade = it.get("grade") or {}
        if grade.get("infra_failure"):
            continue
        if grade.get("passed"):
            passed += 1
        else:
            failed += 1
    return passed, failed


def _passed_iterations_from_trace_path(trace_path):
    return _iterations_from_trace_path(trace_path, passed=True)


def _failed_iterations_from_trace_path(trace_path):
    return _iterations_from_trace_path(trace_path, passed=False)


def _iterations_from_trace_path(trace_path, *, passed):
    trace = json.loads(Path(trace_path).read_text())
    values = []
    for idx, it in enumerate(trace.get("iterations", [])):
        grade = it.get("grade") or {}
        if grade.get("infra_failure"):
            continue
        if bool(grade.get("passed")) is passed:
            values.append(idx)
    return values


def _int_list(value):
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, int):
            result.append(item)
    return result


def _entry_has_signal(entry):
    return bool(str(entry.get("trigger", "")).strip() and str(entry.get("content", "")).strip())


def _entry_id(profile, example, source_function, trigger, content):
    digest = hashlib.sha1(
        "\n".join([profile, example, source_function, trigger, content]).encode()
    ).hexdigest()[:12]
    return f"{profile}:{example}:{source_function}:{digest}"


def _dedupe_entries(entries):
    seen = set()
    unique = []
    for entry in entries:
        key = (
            _norm_text(entry.get("trigger", "")),
            _norm_text(entry.get("content", "")),
            entry.get("source_function", ""),
            entry.get("original_profile") or entry.get("profile", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _norm_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _write_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(_clean_json_text(entry), ensure_ascii=False) + "\n")


def _clean_json_text(value):
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [_clean_json_text(item) for item in value]
    if isinstance(value, dict):
        return {
            _clean_json_text(key): _clean_json_text(item)
            for key, item in value.items()
        }
    return value


def _read_jsonl(path):
    entries = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
