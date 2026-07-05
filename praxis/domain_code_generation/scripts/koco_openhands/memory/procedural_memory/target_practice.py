"""Distill target-function practice memory from OpenHands inference traces.

Target practice consumes completed OpenHands inference artifacts instead of
oracle-based practice runs.  It imports infer/eval output plus agent logs,
distills only passed target traces, and writes structured procedural knowledge
under ``memory/derived/procedural_knowledge``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from agent.sdk import _resolve_llm_model

from .config import (
    SCRIPTS_DIR,
    STRUCTURED_KNOWLEDGE_DIR,
    structured_knowledge_path,
    structured_per_function_dir,
)
from .structured import _dedupe_entries, _parse_json_payload, _write_jsonl

PROVIDER_CHOICES = ("openai", "azure_openai", "openrouter")
DEFAULT_PROFILE = "target"
DEFAULT_SOURCE_SUBDIR = ""
MAX_PROMPT_CHARS = 12000
MAX_COMPLETION_CHARS = 12000
MAX_TOOL_ITEMS = 40
MAX_TRAJECTORY_STEPS = 80
MAX_THOUGHT_CHARS = 1800
MAX_ACTION_CHARS = 2000
MAX_OBSERVATION_CHARS = 4000
MAX_KNOWLEDGE_CHARS = 1200
TRACE_DETAIL_CHOICES = ("summary", "compact", "full")
TARGET_TRACES_DIR = STRUCTURED_KNOWLEDGE_DIR / "_target_traces"


def cmd_import(args) -> None:
    """Import infer output, eval result, and agent logs into target traces."""

    infer_dir = _resolve_source_dir(args)
    result_dir = Path(args.result_dir) if args.result_dir else infer_dir
    if not infer_dir.exists():
        sys.exit(f"ERROR: infer dir not found: {infer_dir}")
    if args.require_result and not result_dir.exists():
        sys.exit(f"ERROR: result dir not found: {result_dir}")

    output_paths = sorted(infer_dir.glob("algorithm_methods_data_*_output.jsonl"))
    if args.example:
        output_paths = [p for p in output_paths if _example_from_output_path(p) == args.example]
    if not output_paths:
        sys.exit(f"ERROR: no *_output.jsonl files found in {infer_dir}")

    imported = skipped = missing_result = failed = 0
    for output_path in output_paths:
        example = _example_from_output_path(output_path)
        if not example:
            continue
        result_path = result_dir / output_path.name.replace("_output.jsonl", "_result.jsonl")
        results_by_function, results_by_index = _load_result_records(result_path)
        if args.require_result and not result_path.exists():
            print(f"[missing-result] {args.framework}/{example}: {result_path}")
            missing_result += 1
            continue

        for idx, output_record in enumerate(_read_jsonl(output_path)):
            function = output_record.get("function_name", "")
            if not function or (args.function and function != args.function):
                skipped += 1
                continue
            result_record = results_by_function.get(function) or results_by_index.get(idx) or {}
            if args.require_result and not result_record:
                print(f"[missing-result] {args.framework}/{example}/{function}")
                missing_result += 1
                continue
            trace_path = target_trace_path(args.profile, args.framework, example, function)
            if trace_path.exists() and not args.force:
                skipped += 1
                continue
            try:
                trace = build_target_trace(
                    profile=args.profile,
                    framework=args.framework,
                    example=example,
                    function=function,
                    output_record=output_record,
                    result_record=result_record,
                    infer_dir=infer_dir,
                    output_path=output_path,
                    result_path=result_path if result_path.exists() else None,
                    source_index=idx,
                    trace_detail=args.trace_detail,
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
                imported += 1
                print(f"[imported] {args.framework}/{example}/{function}")
            except Exception as exc:
                failed += 1
                print(f"[error] {args.framework}/{example}/{function}: {exc}")

    print(
        "target-practice import: "
        f"imported={imported} skipped={skipped} "
        f"missing_result={missing_result} failed={failed}"
    )


def cmd_import_multi(args) -> None:
    """Import multiple infer/eval attempts into one trace per function."""

    attempts = _resolve_attempt_dirs(args)
    if not attempts:
        sys.exit("ERROR: at least one --attempt-dir or --attempt-model-dir is required")

    imported = skipped = missing_result = failed = 0
    traces_by_key: dict[tuple[str, str], list[dict]] = {}
    first_index_by_key: dict[tuple[str, str], int] = {}

    for attempt_index, attempt in enumerate(attempts):
        label = attempt["label"]
        infer_dir = attempt["infer_dir"]
        result_dir = attempt["result_dir"]
        if not infer_dir.exists():
            sys.exit(f"ERROR: attempt infer dir not found: {infer_dir}")
        if args.require_result and not result_dir.exists():
            sys.exit(f"ERROR: attempt result dir not found: {result_dir}")

        output_paths = sorted(infer_dir.glob("algorithm_methods_data_*_output.jsonl"))
        if args.example:
            output_paths = [p for p in output_paths if _example_from_output_path(p) == args.example]
        for output_path in output_paths:
            example = _example_from_output_path(output_path)
            if not example:
                continue
            result_path = result_dir / output_path.name.replace("_output.jsonl", "_result.jsonl")
            results_by_function, results_by_index = _load_result_records(result_path)
            if args.require_result and not result_path.exists():
                missing_result += 1
                continue
            for idx, output_record in enumerate(_read_jsonl(output_path)):
                function = output_record.get("function_name", "")
                if not function or (args.function and function != args.function):
                    skipped += 1
                    continue
                result_record = results_by_function.get(function) or results_by_index.get(idx) or {}
                if args.require_result and not result_record:
                    missing_result += 1
                    continue
                try:
                    single_trace = build_target_trace(
                        profile=args.profile,
                        framework=args.framework,
                        example=example,
                        function=function,
                        output_record=output_record,
                        result_record=result_record,
                        infer_dir=infer_dir,
                        output_path=output_path,
                        result_path=result_path if result_path.exists() else None,
                        source_index=idx,
                        trace_detail=args.trace_detail,
                    )
                    key = (example, function)
                    traces_by_key.setdefault(key, []).append(
                        _single_trace_to_attempt(single_trace, label, attempt_index)
                    )
                    first_index_by_key.setdefault(key, idx)
                except Exception as exc:
                    failed += 1
                    print(f"[error] attempt={label} {args.framework}/{example}/{function}: {exc}")

    for (example, function), attempts_for_function in sorted(traces_by_key.items()):
        trace_path = target_trace_path(args.profile, args.framework, example, function)
        if trace_path.exists() and not args.force:
            skipped += 1
            continue
        trace = build_multi_attempt_trace(
            profile=args.profile,
            framework=args.framework,
            example=example,
            function=function,
            attempts=attempts_for_function,
            source_index=first_index_by_key.get((example, function), 0),
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
        imported += 1
        print(f"[imported-multi] {args.framework}/{example}/{function} attempts={len(attempts_for_function)}")

    print(
        "target-practice import-multi: "
        f"imported={imported} skipped={skipped} "
        f"missing_result={missing_result} failed={failed}"
    )


def cmd_distill(args) -> None:
    """Distill passed target traces into per-function PracticeMemory JSONL."""

    traces = _select_trace_paths(args.profile, args.framework, args.example, args.function)
    if not traces:
        sys.exit("ERROR: no target trace.json files found; run import first")

    wrote = skipped = failed = not_passed = 0
    for trace_path in traces:
        trace = json.loads(trace_path.read_text())
        spec = trace.get("spec") or {}
        if not _trace_passed(trace):
            not_passed += 1
            print(f"[skip-not-passed] {spec.get('framework')}/{spec.get('example')}/{spec.get('function_name')}")
            continue
        out_path = target_per_function_path(
            args.profile,
            spec["framework"],
            spec["example"],
            spec["function_name"],
        )
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            entries = distill_target_trace(
                trace,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                provider=args.provider,
                api_version=args.api_version,
            )
            _write_jsonl(out_path, entries)
            wrote += 1
            print(f"[distilled] {spec['framework']}/{spec['example']}/{spec['function_name']} -> {len(entries)}")
        except Exception as exc:
            failed += 1
            print(f"[error] {spec.get('framework')}/{spec.get('example')}/{spec.get('function_name')}: {exc}")

    print(
        "target-practice distill: "
        f"wrote={wrote} skipped={skipped} not_passed={not_passed} failed={failed}"
    )


def cmd_consolidate(args) -> None:
    examples = _select_examples(args.profile, args.framework, args.example)
    if not examples:
        sys.exit("ERROR: no per-function memories found; run distill first")
    total = 0
    for example in examples:
        entries = []
        per_dir = target_per_function_dir(args.profile, args.framework, example)
        for path in sorted(per_dir.glob("*.jsonl")):
            entries.extend(_read_jsonl(path))
        entries = _dedupe_entries(entries)
        out_path = structured_knowledge_path(example, args.profile, args.framework)
        _write_jsonl(out_path, entries)
        total += len(entries)
        print(f"[consolidated] {args.framework}/{example}: {len(entries)} -> {out_path}")
    print(f"target-practice consolidate: examples={len(examples)} entries={total}")


def build_target_trace(
    *,
    profile: str,
    framework: str,
    example: str,
    function: str,
    output_record: dict,
    result_record: dict,
    infer_dir: Path,
    output_path: Path,
    result_path: Path | None,
    source_index: int,
    trace_detail: str = "compact",
) -> dict:
    log_dir = _agent_log_dir(infer_dir, example, function)
    task_prompt_path = log_dir / "task_prompt.txt"
    tool_trace_path = log_dir / "tool_trace.jsonl"
    sdk_events_path = log_dir / "sdk_events.json"
    tool_rows = _read_jsonl(tool_trace_path) if tool_trace_path.exists() else []
    completion = _first_completion(output_record) or _first_completion(result_record)
    eval_summary = _eval_summary(result_record)
    infer_summary = _infer_summary(output_record)
    return {
        "schema_version": 1,
        "kind": "target_openhands_infer_trace",
        "spec": {
            "profile": profile,
            "framework": framework,
            "example": example,
            "function_name": function,
            "implementation_location": output_record.get("implementation_location", ""),
            "function_signature": output_record.get("function_signature", ""),
            "source_index": source_index,
        },
        "source": {
            "output_file": _repo_relative(output_path),
            "result_file": _repo_relative(result_path) if result_path else "",
            "agent_log_dir": _repo_relative(log_dir) if log_dir.exists() else "",
            "source_model_dir": infer_dir.parent.name,
            "source_subdir": infer_dir.name,
        },
        "infer": {**infer_summary, "completion": completion},
        "eval": eval_summary,
        "trajectory": {
            "task_prompt_path": _repo_relative(task_prompt_path) if task_prompt_path.exists() else "",
            "sdk_events_path": _repo_relative(sdk_events_path) if sdk_events_path.exists() else "",
            "tool_trace_path": _repo_relative(tool_trace_path) if tool_trace_path.exists() else "",
            "task_prompt_excerpt": _read_text(task_prompt_path, MAX_PROMPT_CHARS),
            **summarize_tool_trace(tool_rows, trace_detail=trace_detail),
        },
        "source_record": _compact_source_record(output_record, result_record),
    }


def build_multi_attempt_trace(*, profile: str, framework: str, example: str,
                              function: str, attempts: list[dict], source_index: int) -> dict:
    attempts = sorted(attempts, key=lambda item: item.get("attempt_index", 0))
    first = attempts[0] if attempts else {}
    spec = dict(first.get("spec") or {})
    spec.update({
        "profile": profile,
        "framework": framework,
        "example": example,
        "function_name": function,
        "source_index": source_index,
    })
    pass_ratios = [
        item.get("eval", {}).get("pass_ratio")
        for item in attempts
        if isinstance(item.get("eval", {}).get("pass_ratio"), (int, float))
    ]
    num_passed = sum(1 for item in attempts if item.get("eval", {}).get("passed") is True)
    return {
        "schema_version": 2,
        "kind": "target_openhands_multi_attempt_trace",
        "spec": spec,
        "attempts": attempts,
        "summary": {
            "num_attempts": len(attempts),
            "num_passed": num_passed,
            "num_failed": len(attempts) - num_passed,
            "best_pass_ratio": max(pass_ratios) if pass_ratios else None,
            "pass_ratios": pass_ratios,
        },
    }


def _single_trace_to_attempt(trace: dict, label: str, attempt_index: int) -> dict:
    return {
        "attempt_index": attempt_index,
        "attempt_label": label,
        "spec": trace.get("spec") or {},
        "source": trace.get("source") or {},
        "infer": trace.get("infer") or {},
        "eval": trace.get("eval") or {},
        "trajectory": trace.get("trajectory") or {},
        "completion": (trace.get("infer") or {}).get("completion", ""),
    }


def distill_target_trace(trace: dict, *, model: str, api_key: str, base_url: str,
                         provider: str | None = None, api_version: str | None = None) -> list[dict]:
    is_multi = trace.get("kind") == "target_openhands_multi_attempt_trace"
    system_msg = _multi_attempt_distill_system() if is_multi else _target_distill_system()
    user_msg = _multi_attempt_distill_user(trace) if is_multi else _target_distill_user(trace)
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
    return [
        _normalize_multi_attempt_entry(entry, trace, idx)
        if is_multi else _normalize_target_entry(entry, trace, idx)
        for idx, entry in enumerate(entries, 1)
        if _entry_has_signal(entry)
    ]


def summarize_tool_trace(rows: list[dict], trace_detail: str = "compact") -> dict:
    terminal_commands = []
    files_viewed = []
    files_edited = []
    graph_hits = []
    observed_hits = []
    errors = []
    trajectory_steps = []
    important_events = []
    for idx, row in enumerate(rows):
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        tool_name = row.get("tool_name") or ""
        if tool_name == "terminal":
            command = str(action.get("command") or "").strip()
            if command:
                _append_unique(terminal_commands, command, MAX_TOOL_ITEMS)
        if tool_name == "file_editor":
            command = str(action.get("command") or "").strip()
            path = str(action.get("path") or "").strip()
            if command == "view" and path:
                _append_unique(files_viewed, path, MAX_TOOL_ITEMS)
            elif command and path:
                _append_unique(files_edited, f"{command}: {path}", MAX_TOOL_ITEMS)
        if row.get("has_graph_knowledge"):
            graph_hits.append(_knowledge_hit(row))
        if row.get("has_observed_knowledge") or row.get("has_practice_knowledge"):
            observed_hits.append(_knowledge_hit(row))
        if row.get("is_error"):
            errors.append({
                "tool_name": tool_name,
                "action": _clip_text(json.dumps(action, ensure_ascii=False), 600),
                "observation": _clip_text(str(row.get("observation_text") or ""), 1000),
            })
        if trace_detail in {"compact", "full"} and len(trajectory_steps) < MAX_TRAJECTORY_STEPS:
            step = _compact_tool_step(row, idx)
            trajectory_steps.append(step)
            if _is_important_step(step):
                important_events.append(_important_event(step))
    summary = {
        "tool_observations": len(rows),
        "terminal_commands": terminal_commands,
        "files_viewed": files_viewed,
        "files_edited": files_edited,
        "graph_knowledge_hits": graph_hits[:MAX_TOOL_ITEMS],
        "observed_or_practice_hits": observed_hits[:MAX_TOOL_ITEMS],
        "tool_errors": errors[:MAX_TOOL_ITEMS],
    }
    if trace_detail in {"compact", "full"}:
        summary["trajectory_steps"] = trajectory_steps
        summary["important_events"] = important_events[:MAX_TOOL_ITEMS]
    if trace_detail == "full":
        summary["raw_tool_trace_excerpt"] = [_clip_tool_row(row) for row in rows[:MAX_TRAJECTORY_STEPS]]
    return summary


def target_trace_path(profile: str, framework: str, example: str, function: str) -> Path:
    return target_trace_dir(profile, framework, example, function) / "trace.json"


def target_trace_dir(profile: str, framework: str, example: str, function: str) -> Path:
    return TARGET_TRACES_DIR / profile / framework / example / _safe_component(function)


def target_per_function_dir(profile: str, framework: str, example: str) -> Path:
    return structured_per_function_dir(example, profile, framework)


def target_per_function_path(profile: str, framework: str, example: str, function: str) -> Path:
    return target_per_function_dir(profile, framework, example) / f"{_safe_component(function)}.jsonl"


def _target_distill_system() -> str:
    return """\
You distill passed target-function OpenHands inference trajectories into structured PracticeMemory.

Use only the task, final completion, compact step-level trajectory, token/iteration statistics, and evaluation result.

Return strict JSON only:

{
  "entries": [
    {
      "trigger": "Concrete future situation where this memory applies.",
      "content": "Transferable implementation lesson grounded in the passed target attempt.",
      "evidence": {
        "key_observations": ["Specific signals from eval or trajectory."],
        "key_risk": "Main constraint, failure mode avoided, or validated behavior."
      },
      "confidence": {"score": 0.0}
    }
  ]
}

Rules:
- Emit only lessons that can help a future agent implement related functions.
- This trace has passed evaluation; describe validated implementation/API conventions.
- Prefer specific, actionable lessons over generic advice.
- Reference actual APIs, files, commands, errors, branch conditions, tensor shapes, imports, or test signals when present.
- If there is no useful signal, return {"entries":[]}.
- Do not invent APIs, file contents, test cases, or ground truth.
- confidence.score must be in [0, 1].
"""


def _target_distill_user(trace: dict) -> str:
    spec = trace["spec"]
    infer = trace.get("infer") or {}
    eval_info = trace.get("eval") or {}
    traj = trace.get("trajectory") or {}
    source = trace.get("source") or {}
    parts = [
        f"Function: {spec.get('function_name', '')}",
        f"Framework: {spec.get('framework', '')}",
        f"Example: {spec.get('example', '')}",
        f"Profile: {spec.get('profile', '')}",
        f"Implementation location: {spec.get('implementation_location', '')}",
        "",
        "Evaluation summary:",
        json.dumps(eval_info, ensure_ascii=False, indent=2),
        "",
        "Inference metrics:",
        json.dumps({
            k: infer.get(k)
            for k in ("status", "agent_iterations", "tool_observations", "llm_calls", "total_tokens")
        }, ensure_ascii=False, indent=2),
        "",
        "Source paths:",
        json.dumps(source, ensure_ascii=False, indent=2),
        "",
        "Task prompt excerpt:",
        _clip_text(str(traj.get("task_prompt_excerpt") or ""), MAX_PROMPT_CHARS),
        "",
        "Trajectory summary:",
        json.dumps({
            "terminal_commands": traj.get("terminal_commands", []),
            "files_viewed": traj.get("files_viewed", []),
            "files_edited": traj.get("files_edited", []),
            "graph_knowledge_hits": traj.get("graph_knowledge_hits", []),
            "observed_or_practice_hits": traj.get("observed_or_practice_hits", []),
            "tool_errors": traj.get("tool_errors", []),
        }, ensure_ascii=False, indent=2),
        "",
        "Important trajectory events:",
        json.dumps(traj.get("important_events", []), ensure_ascii=False, indent=2),
        "",
        "Compact step-level trajectory:",
        json.dumps(traj.get("trajectory_steps", []), ensure_ascii=False, indent=2),
        "",
        "Final completion:",
        "```python",
        _clip_text(str(infer.get("completion") or ""), MAX_COMPLETION_CHARS),
        "```",
    ]
    return "\n".join(parts)


def _multi_attempt_distill_system() -> str:
    return """\
You distill multi-attempt target-function OpenHands traces into structured PracticeMemory.
Only passed attempts should support positive implementation lessons.
Return strict JSON with an entries list. Do not invent facts.
"""


def _multi_attempt_distill_user(trace: dict) -> str:
    spec = trace["spec"]
    attempts = trace.get("attempts") or []
    passed_attempts = [attempt for attempt in attempts if attempt.get("eval", {}).get("passed") is True]
    payload = {
        "function": spec.get("function_name"),
        "framework": spec.get("framework"),
        "example": spec.get("example"),
        "summary": trace.get("summary"),
        "passed_attempts": passed_attempts,
    }
    return _clip_text(json.dumps(payload, ensure_ascii=False, indent=2), 100000)


def _normalize_target_entry(entry: dict, trace: dict, idx: int) -> dict:
    spec = trace["spec"]
    infer = trace.get("infer") or {}
    eval_info = trace.get("eval") or {}
    evidence_in = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
    confidence_in = entry.get("confidence") if isinstance(entry.get("confidence"), dict) else {}
    trigger = str(entry.get("trigger", "")).strip()
    content = str(entry.get("content", "")).strip()
    score = confidence_in.get("score")
    if not isinstance(score, (int, float)):
        score = 0.8
    score = max(0.0, min(1.0, float(score)))
    profile = spec["profile"]
    function = spec["function_name"]
    trace_path = target_trace_path(profile, spec["framework"], spec["example"], function)
    return {
        "id": _entry_id(profile, spec["example"], function, trigger, content),
        "profile": profile,
        "framework": spec["framework"],
        "example": spec["example"],
        "source_function": function,
        "implementation_location": spec.get("implementation_location", ""),
        "function_signature": spec.get("function_signature", ""),
        "trigger": trigger,
        "content": content,
        "evidence": {
            "trace_path": _repo_relative(trace_path),
            "eval_passed": eval_info.get("passed"),
            "pass_ratio": eval_info.get("pass_ratio"),
            "source_status": infer.get("status", ""),
            "agent_iterations": infer.get("agent_iterations", 0),
            "llm_calls": infer.get("llm_calls", 0),
            "total_tokens": infer.get("total_tokens", 0),
            "key_observations": _str_list(evidence_in.get("key_observations")),
            "key_risk": str(evidence_in.get("key_risk", "")).strip(),
        },
        "confidence": {
            "practice_attempts": 1,
            "passes": 1,
            "failures": 0,
            "score": round(score, 4),
        },
    }


def _normalize_multi_attempt_entry(entry: dict, trace: dict, idx: int) -> dict:
    single = _normalize_target_entry(entry, {**trace, "infer": {}, "eval": {"passed": True, "pass_ratio": trace.get("summary", {}).get("best_pass_ratio")}}, idx)
    single["evidence"]["multi_attempt"] = True
    single["evidence"]["num_attempts"] = trace.get("summary", {}).get("num_attempts", 0)
    single["confidence"]["practice_attempts"] = trace.get("summary", {}).get("num_attempts", 0)
    single["confidence"]["passes"] = trace.get("summary", {}).get("num_passed", 0)
    single["confidence"]["failures"] = trace.get("summary", {}).get("num_failed", 0)
    return single


def _call_json_entries(system_msg: str, user_msg: str, *, model: str, api_key: str,
                       base_url: str, max_tokens: int, provider: str | None = None,
                       api_version: str | None = None) -> list[dict]:
    import litellm

    kwargs = {
        "model": _resolve_llm_model(model, base_url),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        "api_key": api_key,
        "api_base": base_url,
    }
    if api_version:
        kwargs["api_version"] = api_version
    resp = litellm.completion(**kwargs)
    payload = _parse_json_payload((resp.choices[0].message.content or "").strip())
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise RuntimeError("target distill returned non-list entries")
    return entries


def _parse_json_payload(content: str) -> dict:
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
    raise RuntimeError("target distill returned empty JSON content")


def _trace_passed(trace: dict) -> bool:
    if trace.get("kind") == "target_openhands_multi_attempt_trace":
        return int(trace.get("summary", {}).get("num_passed") or 0) > 0
    return trace.get("eval", {}).get("passed") is True


def _resolve_source_dir(args) -> Path:
    if args.infer_dir:
        return Path(args.infer_dir)
    if not args.source_model_dir:
        sys.exit("ERROR: --source-model-dir is required when --infer-dir is not set")
    base = SCRIPTS_DIR / "data" / args.framework / "openhands" / args.source_model_dir
    return base / args.source_subdir if args.source_subdir else base


def _resolve_attempt_dirs(args) -> list[dict]:
    attempts = []
    for idx, raw in enumerate(args.attempt_dir or []):
        label, infer_dir, result_dir = _parse_attempt_dir_spec(raw, idx)
        attempts.append({"label": label, "infer_dir": infer_dir, "result_dir": result_dir})
    for idx, raw in enumerate(args.attempt_model_dir or [], start=len(attempts)):
        label, model_dir, subdir = _parse_attempt_model_dir_spec(raw, idx, args.source_subdir)
        infer_dir = SCRIPTS_DIR / "data" / args.framework / "openhands" / model_dir
        if subdir:
            infer_dir = infer_dir / subdir
        attempts.append({"label": label, "infer_dir": infer_dir, "result_dir": infer_dir})
    return attempts


def _parse_attempt_dir_spec(raw: str, index: int) -> tuple[str, Path, Path]:
    parts = raw.split("=", 1)
    if len(parts) == 2:
        label, rest = parts[0].strip(), parts[1].strip()
    else:
        label, rest = f"attempt{index}", raw.strip()
    dirs = rest.split(":", 1)
    infer_dir = Path(dirs[0])
    result_dir = Path(dirs[1]) if len(dirs) == 2 else infer_dir
    return label or f"attempt{index}", infer_dir, result_dir


def _parse_attempt_model_dir_spec(raw: str, index: int, default_source_subdir: str) -> tuple[str, str, str]:
    parts = raw.split("=", 1)
    if len(parts) == 2:
        label, rest = parts[0].strip(), parts[1].strip()
    else:
        label, rest = f"attempt{index}", raw.strip()
    model_parts = rest.split(":", 1)
    return label or f"attempt{index}", model_parts[0], model_parts[1] if len(model_parts) == 2 else default_source_subdir


def _load_result_records(path: Path) -> tuple[dict[str, dict], dict[int, dict]]:
    by_function = {}
    by_index = {}
    if not path.exists():
        return by_function, by_index
    for idx, record in enumerate(_read_jsonl(path)):
        function = record.get("function_name", "")
        if function:
            by_function[function] = record
        by_index[idx] = record
    return by_function, by_index


def _infer_summary(record: dict) -> dict:
    metrics = record.get("agent_metrics_summary") if isinstance(record.get("agent_metrics_summary"), dict) else {}
    return {
        "status": record.get("status", ""),
        "agent_iterations": _int_value(record.get("agent_iterations", metrics.get("agent_iterations"))),
        "tool_observations": _int_value(record.get("tool_observations", metrics.get("tool_observations"))),
        "llm_calls": _int_value(record.get("llm_calls", metrics.get("llm_calls"))),
        "total_tokens": _int_value(record.get("total_tokens", metrics.get("total_tokens"))),
    }


def _eval_summary(record: dict) -> dict:
    grade = record.get("grade") if isinstance(record.get("grade"), dict) else {}
    if grade:
        passed = bool(grade.get("passed"))
        return {"available": True, "passed": passed, "pass_ratio": float(grade.get("pass_ratio") or 0.0)}
    results = record.get("results") if isinstance(record.get("results"), list) else []
    pass_ratios = record.get("pass_ratios") if isinstance(record.get("pass_ratios"), list) else []
    passed = bool(results[0]) if results else None
    if pass_ratios:
        pass_ratio = float(pass_ratios[0] or 0.0)
    elif passed is None:
        pass_ratio = None
    else:
        pass_ratio = 1.0 if passed else 0.0
    return {"available": bool(record), "passed": passed, "pass_ratio": pass_ratio, "results": results, "pass_ratios": pass_ratios, "status": record.get("status", "")}


def _compact_source_record(output_record: dict, result_record: dict) -> dict:
    keys = ["function_name", "function_signature", "implementation_location", "detailed_description", "overview", "input_parameters", "output"]
    payload = {key: output_record.get(key) for key in keys if key in output_record}
    if "prompt" in output_record:
        payload["prompt"] = output_record["prompt"]
    if result_record:
        payload["eval_record_keys"] = sorted(result_record.keys())
    return payload


def _agent_log_dir(infer_dir: Path, example: str, function: str) -> Path:
    direct = infer_dir / "agent_logs" / example / function
    if direct.exists():
        return direct
    base = infer_dir / "agent_logs" / example
    if not base.exists():
        return direct
    safe = _safe_component(function)
    for candidate in base.iterdir():
        if candidate.is_dir() and candidate.name == safe:
            return candidate
    return direct


def _select_trace_paths(profile: str, framework: str, example: str = "", function: str = "") -> list[Path]:
    root = TARGET_TRACES_DIR / profile / framework
    if example:
        root = root / example
    if not root.exists():
        return []
    paths = sorted(root.glob("**/trace.json"))
    if function:
        paths = [path for path in paths if _trace_function_name(path) == function]
    return paths


def _trace_function_name(path: Path) -> str:
    try:
        return json.loads(path.read_text()).get("spec", {}).get("function_name", "")
    except Exception:
        return ""


def _select_examples(profile: str, framework: str, example: str = "") -> list[str]:
    if example:
        return [example] if target_per_function_dir(profile, framework, example).exists() else []
    root = STRUCTURED_KNOWLEDGE_DIR / profile / framework
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "per_function").exists())


def _example_from_output_path(path: Path) -> str:
    name = path.name
    prefix = "algorithm_methods_data_"
    suffix = "_output.jsonl"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return ""


def _first_completion(record: dict) -> str:
    completions = record.get("completions")
    if isinstance(completions, list) and completions:
        return completions[0] or ""
    completion = record.get("completion")
    return completion if isinstance(completion, str) else ""


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compact_tool_step(row: dict, index: int) -> dict:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    observation = str(row.get("observation_text") or "")
    knowledge = str(row.get("knowledge_text") or "")
    step = {
        "index": index,
        "tool_name": row.get("tool_name", ""),
        "thought": _clip_text(str(row.get("thought_text") or "").strip(), MAX_THOUGHT_CHARS),
        "action": _clip_text(json.dumps(action, ensure_ascii=False), MAX_ACTION_CHARS),
        "observation_excerpt": _clip_text(observation.strip(), MAX_OBSERVATION_CHARS),
        "is_error": bool(row.get("is_error")),
    }
    if knowledge:
        step["knowledge_excerpt"] = _clip_text(knowledge.strip(), MAX_KNOWLEDGE_CHARS)
    for key in ("has_graph_knowledge", "has_practice_knowledge", "has_observed_knowledge"):
        if key in row:
            step[key] = bool(row.get(key))
    return step


def _is_important_step(step: dict) -> bool:
    if step.get("is_error") or step.get("has_graph_knowledge") or step.get("has_practice_knowledge"):
        return True
    observation = str(step.get("observation_excerpt") or "").lower()
    action_text = str(step.get("action") or "")
    if step.get("tool_name") == "file_editor" and any(cmd in action_text for cmd in ("str_replace", "insert", "create")):
        return True
    return any(marker in observation for marker in ("traceback", "error", "failed", "assert", "exception", "pytest"))


def _important_event(step: dict) -> dict:
    return {
        "index": step.get("index"),
        "tool_name": step.get("tool_name", ""),
        "action": _clip_text(str(step.get("action") or ""), 800),
        "observation_excerpt": _clip_text(str(step.get("observation_excerpt") or ""), 2500),
        "knowledge_excerpt": _clip_text(str(step.get("knowledge_excerpt") or ""), MAX_KNOWLEDGE_CHARS) if step.get("knowledge_excerpt") else "",
        "is_error": bool(step.get("is_error")),
    }


def _clip_tool_row(row: dict) -> dict:
    clipped = {}
    for key in ("event_index", "tool_name", "thought_text", "action", "observation_text", "knowledge_text", "has_graph_knowledge", "has_practice_knowledge", "has_observed_knowledge", "is_error"):
        if key not in row:
            continue
        value = row.get(key)
        clipped[key] = _clip_text(value, MAX_OBSERVATION_CHARS) if isinstance(value, str) else value
    return clipped


def _knowledge_hit(row: dict) -> dict:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    return {"tool_name": row.get("tool_name", ""), "action": _clip_text(json.dumps(action, ensure_ascii=False), 500), "knowledge_text": _clip_text(str(row.get("knowledge_text") or ""), MAX_KNOWLEDGE_CHARS)}


def _append_unique(items: list[str], value: str, limit: int) -> None:
    if len(items) < limit and value not in items:
        items.append(value)


def _entry_has_signal(entry: dict) -> bool:
    return bool(str(entry.get("trigger", "")).strip() and str(entry.get("content", "")).strip())


def _entry_id(profile: str, example: str, source_function: str, trigger: str, content: str) -> str:
    digest = hashlib.sha1("\n".join([profile, example, source_function, trigger, content]).encode()).hexdigest()[:12]
    return f"{profile}:{example}:{source_function}:{digest}"


def _safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(value).strip()).strip(" .") or "function"
    digest = hashlib.sha1(str(value).encode()).hexdigest()[:8]
    return f"{text[:120]}__{digest}"


def _repo_relative(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(SCRIPTS_DIR.parent.parent.parent).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _read_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    try:
        return _clip_text(path.read_text(encoding="utf-8", errors="replace"), max_chars)
    except OSError:
        return ""


def _clip_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_value(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _env_file_values() -> dict[str, str]:
    values = {}
    for env_file in (SCRIPTS_DIR / ".env", SCRIPTS_DIR / ".env.local"):
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values[name.strip()] = value.split("#", 1)[0].strip()
    return values


def _infer_provider(base_url: str) -> str:
    if "openrouter.ai" in base_url:
        return "openrouter"
    if "openai.azure.com" in base_url or "services.ai.azure.com" in base_url:
        return "azure_openai"
    if "api.openai.com" in base_url or "token-plan.cn-beijing.maas.aliyuncs.com" in base_url:
        return "openai"
    return ""


def _default_api_key_env(provider: str, base_url: str) -> str:
    provider = provider or _infer_provider(base_url)
    if "token-plan.cn-beijing.maas.aliyuncs.com" in base_url:
        return "BAILIAN_TOKEN_PLAN_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "azure_openai":
        return "AZURE_OPENAI_API_KEY"
    return "OPENROUTER_API_KEY"


def _load_api_key(*, api_key: str, api_key_env: str | None, provider: str, base_url: str) -> str:
    if api_key:
        return api_key
    env_name = api_key_env or _default_api_key_env(provider, base_url)
    env_values = _env_file_values()
    key = os.environ.get(env_name, "").strip() or env_values.get(env_name, "").strip()
    if key:
        return key
    sys.exit(f"ERROR: {env_name} is not set; pass --api-key or export it")


def _add_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=DEFAULT_PROFILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
    parser.add_argument("--api-version", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="build target trace.json files")
    _add_profile_arg(p_import)
    p_import.add_argument("--framework", required=True)
    p_import.add_argument("--source-model-dir", default="")
    p_import.add_argument("--source-subdir", default=DEFAULT_SOURCE_SUBDIR)
    p_import.add_argument("--infer-dir", default="")
    p_import.add_argument("--result-dir", default="")
    p_import.add_argument("--example", default="")
    p_import.add_argument("--function", default="")
    p_import.add_argument("--require-result", action="store_true")
    p_import.add_argument("--trace-detail", choices=TRACE_DETAIL_CHOICES, default="compact")
    p_import.add_argument("--force", action="store_true")
    p_import.set_defaults(func=cmd_import)

    p_import_multi = sub.add_parser("import-multi", help="build multi-attempt target trace.json files")
    _add_profile_arg(p_import_multi)
    p_import_multi.add_argument("--framework", required=True)
    p_import_multi.add_argument("--attempt-model-dir", action="append", default=[])
    p_import_multi.add_argument("--attempt-dir", action="append", default=[])
    p_import_multi.add_argument("--source-subdir", default=DEFAULT_SOURCE_SUBDIR)
    p_import_multi.add_argument("--example", default="")
    p_import_multi.add_argument("--function", default="")
    p_import_multi.add_argument("--require-result", action="store_true")
    p_import_multi.add_argument("--trace-detail", choices=TRACE_DETAIL_CHOICES, default="compact")
    p_import_multi.add_argument("--force", action="store_true")
    p_import_multi.set_defaults(func=cmd_import_multi)

    p_distill = sub.add_parser("distill", help="distill passed target traces")
    _add_profile_arg(p_distill)
    p_distill.add_argument("--framework", required=True)
    p_distill.add_argument("--example", default="")
    p_distill.add_argument("--function", default="")
    p_distill.add_argument("--model", required=True)
    p_distill.add_argument("--force", action="store_true")
    p_distill.set_defaults(func=cmd_distill)

    p_consolidate = sub.add_parser("consolidate", help="merge per-function target memories")
    _add_profile_arg(p_consolidate)
    p_consolidate.add_argument("--framework", required=True)
    p_consolidate.add_argument("--example", default="")
    p_consolidate.set_defaults(func=cmd_consolidate)

    args = parser.parse_args()
    args.provider = args.provider or _infer_provider(args.base_url)
    if args.command == "distill":
        args.api_key = _load_api_key(api_key=args.api_key, api_key_env=args.api_key_env, provider=args.provider, base_url=args.base_url)
    args.func(args)


if __name__ == "__main__":
    main()
