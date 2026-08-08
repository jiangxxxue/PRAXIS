"""Coverage-feedback regeneration loop.

Runs after Stage 4 (coverage): for functions below the coverage threshold,
feeds uncovered lines and execution errors back to the LLM agent so it can
regenerate improved test inputs.  Loops until coverage meets the threshold
or max retries are exhausted.
"""

import glob
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from agent.sdk import run_sdk_agent
from memory.config import (
    code_dir,
    candidates_path,
    coverage_result_path,
    feedback_log_path,
    feedback_status_path,
    observed_knowledge_path,
    test_input_path,
)
from memory.observed_memory.status import (
    fingerprint,
    is_non_retryable_error,
    is_transient_error,
    load_status,
    normalized_error_signature,
    reusable_terminal_status,
    write_status,
)
from memory.observed_memory.quality import (
    coverage_has_execution_failure,
    coverage_quality_score,
    coverage_ready,
    stage2_case_gate,
)
from memory.observed_memory.validate_test_input import validate_test_input
from memory.observed_memory.workspace import (
    benchmark_target_locations,
    build_single_stub_workspace,
)
from runner import _parse_impl_location

# Max retries when feedback-generated test_input fails validation
_MAX_FEEDBACK_VALIDATION_RETRIES = 2
_ENVIRONMENT_FAILURE_MARKERS = (
    "ModuleNotFoundError:",
    "ImportError:",
    "cannot import name",
    "No module named",
    "not a package",
)


class FeedbackResults(dict):
    complete = False


def _feedback_budget(
    max_iterations: int,
    max_retries: int,
) -> dict[str, int]:
    return {
        "max_iterations": max_iterations,
        "max_retries": max_retries,
    }


def _feedback_iteration_budgets(
    max_iterations: int,
    terminal_max_iterations: int,
    max_attempts: int,
) -> list[int]:
    terminal = max(max_iterations, terminal_max_iterations)
    tiers = [max_iterations]
    if max_iterations < terminal:
        tiers.append(min(max(max_iterations, 180), terminal))
    if tiers[-1] < terminal:
        tiers.append(terminal)
    return list(dict.fromkeys(tiers))[:max_attempts]


def _read_feedback_history(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _write_feedback_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _coverage_fingerprint_value(coverage: dict) -> str:
    metadata = coverage.get("_praxis_coverage")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("fingerprint") or "")


def _relevant_feedback_history(
    history: list[dict],
    coverage: dict,
) -> list[dict]:
    current_fingerprint = _coverage_fingerprint_value(coverage)
    if not current_fingerprint:
        return []
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("coverage_fingerprint_after") == current_fingerprint:
            return history[max(0, index - 1):index + 1]
    return []


def _business_logic_candidate(candidate: dict) -> bool:
    rationale = str(candidate.get("rationale") or "").strip().lower()
    return not (
        rationale.startswith("indegree=")
        or rationale.startswith("outdegree=")
    )


def _selective_third_attempt_reasons(
    candidate: dict,
    coverage: dict,
    recent_attempts: list[dict],
) -> list[str]:
    reasons = []
    line_coverage = float(coverage.get("line_coverage") or 0.0)
    gate = stage2_case_gate(coverage) or {}
    latest = recent_attempts[-1] if recent_attempts else {}
    if (
        float(latest.get("coverage_after") or 0.0)
        - float(latest.get("coverage_before") or 0.0)
        >= 0.05
    ):
        reasons.append("recent coverage gain >= 5pp")
    if line_coverage >= 0.65:
        reasons.append("coverage >= 65%")
    if latest.get("execution_failure_fixed"):
        reasons.append("execution failure fixed")
    if (
        int(gate.get("normal_cases") or 0) >= 4
        and float(gate.get("normal_edge_success_ratio") or 0.0) >= 0.7
    ):
        reasons.append("case gate is close")
    if _business_logic_candidate(candidate):
        reasons.append("business-logic candidate with another progress signal")
    if reasons == ["business-logic candidate with another progress signal"]:
        return []
    return reasons


def _repeated_no_progress(attempts: list[dict]) -> bool:
    if len(attempts) < 2:
        return False
    previous, latest = attempts[-2:]
    signatures = [
        str(item.get("error_signature") or "")
        for item in (previous, latest)
    ]
    return (
        all(signatures)
        and signatures[0] == signatures[1]
        and not previous.get("quality_improved")
        and not latest.get("quality_improved")
    )


def _feedback_fingerprint(
    candidate: dict,
    framework: str,
    example: str,
    model: str,
    base_url: str,
    coverage_threshold: float,
) -> str:
    from memory.observed_memory.coverage_orch import _coverage_execution_context

    func_name = candidate["function_name"]
    rel_path, _line_start, _line_end = _parse_impl_location(
        candidate["implementation_location"]
    )
    return fingerprint(
        {
            "stage": "feedback",
            "framework": framework,
            "example": example,
            "candidate": candidate,
            "model": model,
            "base_url": base_url,
            "coverage_threshold": coverage_threshold,
            "coverage_execution_context": _coverage_execution_context(
                framework,
                False,
                example=example,
            ),
        },
        [
            code_dir(framework, example) / rel_path,
            test_input_path(framework, example, func_name),
            coverage_result_path(framework, example, func_name),
            observed_knowledge_path(framework, example),
            Path(__file__).resolve().parent / "prompts" / "regenerate.md",
            Path(__file__).resolve().parent / "coverage_runner.py",
            Path(__file__).resolve().parent / "quality.py",
            Path(__file__),
            Path(validate_test_input.__code__.co_filename),
            Path(build_single_stub_workspace.__code__.co_filename),
        ],
    )


def _suppress_verbose_logging():
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    for name in ("litellm", "httpx", "httpcore", "openai", "httpcore.http11"):
        logging.getLogger(name).setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def _find_file_in_workspace(ws: str, pattern: str) -> str | None:
    matches = glob.glob(os.path.join(ws, "**", pattern), recursive=True)
    return matches[0] if matches else None


def _extract_python_block(content: str) -> str | None:
    import re
    pattern = r"```(?:python|py)\s*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return max(matches, key=len).strip() if matches else None


# ── source annotation ──────────────────────────────────────────────────────

def _annotate_source_with_coverage(
    source_lines: list[str],
    line_start: int,
    line_end: int,
    missing_lines: list[int],
    covered_lines: list[int],
) -> str:
    """Return source with line numbers and [MISSING]/[COVERED] tags."""
    missing_set = set(missing_lines)
    covered_set = set(covered_lines)
    result = []
    for i in range(line_start - 1, min(line_end, len(source_lines))):
        lineno = i + 1
        if lineno in missing_set:
            tag = "MISSING "
        elif lineno in covered_set:
            tag = "COVERED "
        else:
            tag = "        "
        result.append(f"  {lineno:4d} [{tag}] {source_lines[i].rstrip()}")
    return "\n".join(result)


# ── formatters ──────────────────────────────────────────────────────────────

def _format_coverage_stats(coverage_data: dict) -> str:
    lc = coverage_data.get("line_coverage", 0)
    n_cov = coverage_data.get("num_covered_lines", 0)
    n_tot = coverage_data.get("total_executable_lines", 0)
    n_err = coverage_data.get("num_execution_errors", 0)
    return f"{lc:.1%} ({n_cov}/{n_tot} lines), {n_err} execution errors"


def _format_execution_errors(coverage_data: dict) -> str:
    errors = coverage_data.get("execution_errors", [])
    if not errors:
        return "No execution errors."
    lines = []
    for e in errors:
        lines.append(
            f"  - [{e.get('category', '?')} #{e.get('index', '?')}] "
            f"{e.get('error_type', 'Error')}: {e.get('error', '')[:300]}"
        )
    return "\n".join(lines)


def _format_missing_lines(missing_lines: list[int]) -> str:
    if not missing_lines:
        return "All lines covered!"
    # Compact ranges for readability
    ranges = []
    start = prev = missing_lines[0]
    for ln in missing_lines[1:]:
        if ln == prev + 1:
            prev = ln
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = ln
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"Lines: {', '.join(ranges)} ({len(missing_lines)} lines)"


def _is_environment_blocked(coverage_data: dict) -> bool:
    """Return whether coverage failed before the test input could exercise code."""
    if not coverage_data.get("is_execution_failure", False):
        return False
    errors = coverage_data.get("execution_errors", [])
    return any(
        marker in str(error.get("error", ""))
        for error in errors
        for marker in _ENVIRONMENT_FAILURE_MARKERS
    )


# ── prompt building ─────────────────────────────────────────────────────────

def _build_feedback_prompt(
    func_name: str,
    impl_loc: str,
    source_lines: list[str],
    line_start: int,
    line_end: int,
    coverage_data: dict,
    current_test_input: str,
    static_mem: str | None,
    attempt: int,
    max_retries: int,
    last_error: str = "",
) -> str:
    """Assemble the full regeneration prompt."""
    is_error_only = coverage_data.get("is_execution_failure", False)
    missing = coverage_data.get("missing_lines", [])
    covered = coverage_data.get("covered_lines", [])

    annotated = _annotate_source_with_coverage(
        source_lines, line_start, line_end, missing, covered,
    )

    prompt_text = (
        Path(__file__).resolve().parent
        / "prompts"
        / "regenerate.md"
    ).read_text(encoding="utf-8")
    prompt_text = prompt_text.replace("{function_name}", func_name)
    prompt_text = prompt_text.replace("{implementation_location}", impl_loc)
    prompt_text = prompt_text.replace("{annotated_source}", annotated)

    if is_error_only:
        # Focus on fixing execution errors
        error_info = _format_execution_errors(coverage_data)
        prompt_text = prompt_text.replace("{missing_lines_summary}", "N/A (execution failed)")
        prompt_text = prompt_text.replace(
            "{execution_errors_summary}",
            f"**CRITICAL: The test_input FAILED to execute.** Fix the error first:\n\n{error_info}",
        )
        prompt_text = prompt_text.replace("{coverage_stats}", "EXECUTION FAILURE — test_input crashed")
    else:
        prompt_text = prompt_text.replace(
            "{missing_lines_summary}", _format_missing_lines(missing),
        )
        prompt_text = prompt_text.replace(
            "{execution_errors_summary}", _format_execution_errors(coverage_data),
        )
        prompt_text = prompt_text.replace("{coverage_stats}", _format_coverage_stats(coverage_data))
    prompt_text = prompt_text.replace("{current_test_input}", current_test_input)
    prompt_text = prompt_text.replace(
        "{iteration_info}",
        f"This is regeneration attempt {attempt}/{max_retries}.",
    )

    if last_error:
        prompt_text = prompt_text.replace(
            "{previous_error_feedback}",
            f"\n## Previous Attempt Error (MUST FIX)\n"
            f"The test_input you produced last time caused this error when executed:\n\n"
            f"```\n{last_error}\n```\n\n"
            f"Analyze WHY this error occurred and make sure your new test_input does NOT repeat it.\n",
        )
    else:
        prompt_text = prompt_text.replace("{previous_error_feedback}", "")

    if static_mem:
        prompt_text += f"\n\n## Reference: Project Knowledge Summary\n\n{static_mem}"

    return prompt_text


# ── single agent call ───────────────────────────────────────────────────────

def _regenerate_one_candidate(
    candidate: dict,
    framework: str,
    example: str,
    coverage_data: dict,
    source_lines: list[str],
    line_start: int,
    line_end: int,
    model: str,
    api_key: str,
    base_url: str,
    max_iterations: int,
    attempt: int,
    max_retries: int,
    debug: bool = False,
    last_error: str = "",
) -> dict:
    """Run agent for a single candidate to regenerate test_input."""
    func_name = candidate["function_name"]
    impl_loc = candidate["implementation_location"]

    # Load current test_input for reference
    ti_path = test_input_path(framework, example, func_name)
    current_test_input = ti_path.read_text(encoding="utf-8") if ti_path.exists() else "# (no existing test_input)"

    # Load static memory
    sm_path = observed_knowledge_path(framework, example)
    static_mem = sm_path.read_text(encoding="utf-8") if sm_path.exists() else None

    # Build prompt
    prompt_text = _build_feedback_prompt(
        func_name, impl_loc, source_lines, line_start, line_end,
        coverage_data, current_test_input, static_mem, attempt, max_retries, last_error,
    )

    # Debug: dump prompt for inspection
    if debug:
        prompt_dump = Path("debug_feedback") / framework / example / func_name / f"attempt_{attempt}_prompt.md"
        prompt_dump.parent.mkdir(parents=True, exist_ok=True)
        prompt_dump.write_text(prompt_text, encoding="utf-8")
        print(f"    DEBUG: prompt saved ({len(prompt_text)} chars) -> {prompt_dump}", flush=True)

    # Workspace
    rel_path, start, end = _parse_impl_location(impl_loc)
    code_root = str(code_dir(framework, example))

    tmp_dir_ctx: tempfile.TemporaryDirectory | None = None
    if debug:
        import shutil
        debug_dir = Path("debug_feedback") / framework / example / func_name / f"attempt_{attempt}"
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = str(debug_dir)
    else:
        tmp_dir_ctx = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_ctx.__enter__()

    try:
        paths = build_single_stub_workspace(
            code_root,
            benchmark_target_locations(framework, example),
            rel_path,
            start,
            end,
            tmp_dir,
        )

        events, status = run_sdk_agent(
            prompt=prompt_text,
            workspace=paths["workspace"],
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_iterations=max_iterations,
            corpus_dirs=[paths["code"]],
        )

        # Save events in debug mode
        if debug:
            events_file = Path(tmp_dir) / "sdk_events.json"
            events_data = []
            for ev in events:
                ev_dict = {"type": type(ev).__name__}
                for attr in ("content", "thought", "action", "args"):
                    val = getattr(ev, attr, None)
                    if val:
                        ev_dict[attr] = str(val)[:500]
                events_data.append(ev_dict)
            events_file.write_text(
                json.dumps(events_data, ensure_ascii=False, indent=2), encoding="utf-8",
            )

        # Extract test_input from workspace
        ws = paths["workspace"]
        result = {"function_name": func_name, "status": str(status)}

        test_file = os.path.join(ws, f"{func_name}_test_input.py")
        if not os.path.exists(test_file):
            test_file = _find_file_in_workspace(ws, f"*{func_name}*test_input*.py")
            if not test_file:
                test_file = _find_file_in_workspace(ws, "*test_input*.py")

        if test_file and os.path.exists(test_file):
            result["test_input"] = Path(test_file).read_text(encoding="utf-8")
            print(f"    test_input extracted ({os.path.relpath(test_file, ws)})", flush=True)
        else:
            # Fallback: extract from events
            for event in reversed(events):
                content = getattr(event, "content", None) or ""
                if "test_cases" in content and ("normal" in content or "edge" in content):
                    py_block = _extract_python_block(content)
                    if py_block:
                        result["test_input"] = py_block
                        break
            if "test_input" not in result:
                print(f"    test_input NOT FOUND in workspace or events", flush=True)

    finally:
        if tmp_dir_ctx:
            tmp_dir_ctx.__exit__(None, None, None)

    return result


# ── single-function feedback loop ───────────────────────────────────────────

def _run_feedback_for_one(
    func_name: str,
    candidate: dict,
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str,
    max_iterations: int,
    coverage_threshold: float,
    max_retries: int,
    terminal_budget: dict[str, int],
    gt_locations: dict | None,
    debug: bool = False,
) -> tuple[list[dict], dict]:
    """Run one feedback loop and return its iteration log and persistent state."""
    from memory.observed_memory.coverage_orch import run_coverage_single

    cov_path = coverage_result_path(framework, example, func_name)
    status_path = feedback_status_path(framework, example, func_name)
    log_path = feedback_log_path(framework, example, func_name)
    iteration_budgets = _feedback_iteration_budgets(
        max_iterations,
        terminal_budget["max_iterations"],
        terminal_budget["max_retries"],
    )
    history = _read_feedback_history(log_path)
    if not cov_path.exists():
        print(f"  SKIP {func_name}: no coverage data")
        state = write_status(
            status_path,
            status="unrunnable",
            fingerprint_value=_feedback_fingerprint(
                candidate,
                framework,
                example,
                model,
                base_url,
                coverage_threshold,
            ),
            budget=terminal_budget,
            model=model,
            reason="coverage data missing",
        )
        return [], state

    impl_loc = candidate["implementation_location"]
    rel_path, line_start, line_end = _parse_impl_location(impl_loc)
    if gt_locations and func_name in gt_locations:
        entry = gt_locations[func_name]
        rel_path, line_start, line_end = entry[0], entry[1], entry[2]

    source_file = code_dir(framework, example) / rel_path
    if not source_file.exists():
        print(f"  SKIP {func_name}: source file not found ({rel_path})")
        state = write_status(
            status_path,
            status="unrunnable",
            fingerprint_value=_feedback_fingerprint(
                candidate,
                framework,
                example,
                model,
                base_url,
                coverage_threshold,
            ),
            budget=terminal_budget,
            model=model,
            reason=f"source file not found: {rel_path}",
        )
        return [], state

    source_lines = source_file.read_text(encoding="utf-8").splitlines()

    iteration_log = []
    last_error = ""
    terminal_reason = ""
    latest_error_transient = False
    non_retryable_error = False
    environment_blocked = False
    early_stop_reason = ""

    for attempt, attempt_budget in enumerate(iteration_budgets, 1):
        # Load current coverage
        if not cov_path.exists():
            print(f"  [{func_name}] coverage file disappeared, stopping")
            terminal_reason = "coverage file disappeared"
            break

        current_coverage = json.loads(cov_path.read_text(encoding="utf-8"))
        current_is_error = coverage_has_execution_failure(current_coverage)
        current_raw_execution_failure = bool(
            current_coverage.get("is_execution_failure")
        )
        current_lc = current_coverage.get("line_coverage", 0) or 0
        environment_blocked = _is_environment_blocked(current_coverage)
        relevant_history = _relevant_feedback_history(
            history,
            current_coverage,
        )

        if coverage_ready(current_coverage, coverage_threshold):
            print(f"  [{func_name}] coverage {current_lc:.1%} >= threshold {coverage_threshold:.1%}, done")
            break

        if environment_blocked and (attempt > 1 or relevant_history):
            print(
                f"  [{func_name}] environment import failure persisted after one repair "
                "attempt; skipping further LLM retries",
                flush=True,
            )
            terminal_reason = "persistent environment import failure"
            early_stop_reason = "persistent_environment_failure"
            break

        if attempt >= 3:
            if _repeated_no_progress(relevant_history):
                terminal_reason = (
                    "same deterministic error repeated without quality gain"
                )
                early_stop_reason = "repeated_no_progress"
                print(
                    f"  [{func_name}] {terminal_reason}; stopping before "
                    f"adaptive attempt {attempt}",
                    flush=True,
                )
                break
            reasons = _selective_third_attempt_reasons(
                candidate,
                current_coverage,
                relevant_history,
            )
            if not reasons:
                terminal_reason = "not eligible for selective terminal feedback"
                early_stop_reason = "low_expected_yield"
                print(
                    f"  [{func_name}] {terminal_reason}; stopping",
                    flush=True,
                )
                break
            print(
                f"  [{func_name}] selective attempt reasons: "
                f"{', '.join(reasons)}",
                flush=True,
            )

        missing_count = len(current_coverage.get("missing_lines", []))
        error_count = current_coverage.get("num_execution_errors", 0)
        label = "FIX ERROR" if current_is_error else f"coverage={current_lc:.1%}"
        print(
            f"  [{func_name}] attempt {attempt}/{len(iteration_budgets)}: "
            f"{label}, missing={missing_count} lines, errors={error_count}, "
            f"max_iterations={attempt_budget}",
            flush=True,
        )

        # Regenerate
        started = time.monotonic()
        result = {}
        try:
            result = _regenerate_one_candidate(
                candidate, framework, example, current_coverage,
                source_lines, line_start, line_end,
                model, api_key, base_url, attempt_budget,
                attempt, len(iteration_budgets), debug, last_error,
            )
        except Exception as exc:
            print(f"  [{func_name}] agent call failed: {exc}", flush=True)
            last_error = str(exc)
            latest_error_transient = is_transient_error(last_error)
            non_retryable_error = is_non_retryable_error(last_error)

        new_lc = current_lc
        test_input_saved = False
        quality_improved = False
        execution_failure_fixed = False
        current_score = coverage_quality_score(
            current_coverage,
            coverage_threshold,
        )
        final_coverage_for_attempt = current_coverage
        if result:
            last_error = ""

        if result.get("test_input"):
            # Validate before saving
            code_root = str(code_dir(framework, example))
            errors = validate_test_input(
                result["test_input"],
                code_root,
                framework,
                implementation_location=impl_loc,
            )
            if errors:
                print(f"  [{func_name}] validation failed, feeding back to agent")
                last_error = "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)
                result["status"] = "validation_failed"
                terminal_reason = last_error
            else:
                # Save and re-run coverage.
                ti_path = test_input_path(framework, example, func_name)
                old_test_input = (
                    ti_path.read_text(encoding="utf-8")
                    if ti_path.exists()
                    else ""
                )
                old_coverage_text = cov_path.read_text(encoding="utf-8")
                ti_path.write_text(result["test_input"], encoding="utf-8")

                try:
                    new_cov = run_coverage_single(
                        framework, example, func_name,
                        per_test=False, capture_output=True,
                        gt_locations=gt_locations,
                    )
                except Exception as exc:
                    print(
                        f"  [{func_name}] coverage re-run FAILED: {exc}, "
                        "reverting",
                    )
                    last_error = str(exc)
                    new_cov = None

                if new_cov:
                    new_lc = float(new_cov.get("line_coverage") or 0.0)
                    new_score = coverage_quality_score(
                        new_cov,
                        coverage_threshold,
                    )
                    quality_improved = new_score > current_score
                    execution_failure_fixed = (
                        current_raw_execution_failure
                        and not bool(new_cov.get("is_execution_failure"))
                    )
                    if quality_improved:
                        final_coverage_for_attempt = new_cov
                        test_input_saved = True
                        print(
                            f"  [{func_name}] quality improved: "
                            f"coverage {current_lc:.1%} -> {new_lc:.1%}",
                            flush=True,
                        )
                        if coverage_has_execution_failure(new_cov):
                            last_error = "Stage 2 quality gate still not met"
                            terminal_reason = last_error
                    else:
                        print(
                            f"  [{func_name}] no Stage 2 quality gain; "
                            "reverting without another coverage run",
                            flush=True,
                        )
                        last_error = "No Stage 2 quality improvement"
                        terminal_reason = last_error
                else:
                    last_error = last_error or "Coverage re-run failed"
                    terminal_reason = last_error

                if not test_input_saved:
                    if old_test_input:
                        ti_path.write_text(old_test_input, encoding="utf-8")
                    else:
                        ti_path.unlink(missing_ok=True)
                    cov_path.write_text(old_coverage_text, encoding="utf-8")
                    final_coverage_for_attempt = current_coverage
                    new_lc = current_lc
        else:
            if result:
                print(f"  [{func_name}] agent did not produce test_input")
                last_error = "Agent did not produce a valid test_input file"
                terminal_reason = last_error

        latest_error_transient = is_transient_error(last_error)
        non_retryable_error = is_non_retryable_error(last_error)
        record = {
            "attempt": len(history) + 1,
            "stage_attempt": attempt,
            "max_iterations": attempt_budget,
            "duration_seconds": round(time.monotonic() - started, 3),
            "coverage_before": current_lc,
            "coverage_after": new_lc,
            "agent_status": str(result.get("status", "unknown")),
            "has_test_input": "test_input" in result,
            "test_input_saved": test_input_saved,
            "quality_improved": quality_improved,
            "execution_failure_fixed": execution_failure_fixed,
            "last_error": last_error,
            "error_signature": normalized_error_signature(last_error),
            "coverage_fingerprint_before": _coverage_fingerprint_value(
                current_coverage
            ),
            "coverage_fingerprint_after": _coverage_fingerprint_value(
                final_coverage_for_attempt
            ),
            "stage2_case_gate_before": stage2_case_gate(current_coverage),
            "stage2_case_gate_after": stage2_case_gate(
                final_coverage_for_attempt
            ),
            "timestamp": datetime.now().isoformat(),
        }
        iteration_log.append(record)
        history.append(record)
        _write_feedback_history(log_path, history)

        current_after = json.loads(cov_path.read_text(encoding="utf-8"))
        if coverage_ready(current_after, coverage_threshold):
            print(f"  [{func_name}] Stage 2 quality gate met, done")
            break
        if non_retryable_error:
            terminal_reason = last_error
            early_stop_reason = "non_retryable_request"
            print(f"  [{func_name}] non-retryable request error; stopping")
            break
    else:
        print(
            f"  [{func_name}] adaptive attempts "
            f"({len(iteration_budgets)}) exhausted"
        )

    final_coverage = json.loads(cov_path.read_text(encoding="utf-8"))
    final_is_failure = coverage_has_execution_failure(final_coverage)
    final_lc = float(final_coverage.get("line_coverage") or 0.0)
    final_environment_blocked = _is_environment_blocked(final_coverage)
    coverage_metadata = final_coverage.get("_praxis_coverage")
    coverage_fingerprint = (
        coverage_metadata.get("fingerprint")
        if isinstance(coverage_metadata, dict)
        else None
    )
    if coverage_ready(final_coverage, coverage_threshold):
        status = "success"
        terminal_reason = f"coverage {final_lc:.1%} meets threshold"
    elif latest_error_transient and not non_retryable_error:
        status = "retryable"
        terminal_reason = terminal_reason or last_error or "transient agent failure"
    elif final_environment_blocked:
        status = "unrunnable"
        terminal_reason = terminal_reason or "persistent environment failure"
    else:
        status = "exhausted"
        terminal_reason = (
            terminal_reason
            or last_error
            or "adaptive feedback budget exhausted"
        )

    state = write_status(
        status_path,
        status=status,
        fingerprint_value=_feedback_fingerprint(
            candidate,
            framework,
            example,
            model,
            base_url,
            coverage_threshold,
        ),
        budget=terminal_budget,
        model=model,
        reason=terminal_reason,
        details={
            "line_coverage": final_lc,
            "is_execution_failure": bool(final_is_failure),
            "stage2_case_gate": stage2_case_gate(final_coverage),
            "attempts": len(history),
            "latest_run_attempts": len(iteration_log),
            "coverage_fingerprint": coverage_fingerprint,
            "adaptive_policy": "balanced-v1",
            "early_stop_reason": early_stop_reason,
        },
    )
    return iteration_log, state


# ── outer orchestrator ──────────────────────────────────────────────────────

def run_feedback_loop(
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 100,
    terminal_max_iterations: int | None = None,
    coverage_threshold: float = 0.8,
    max_retries: int = 3,
    terminal_max_retries: int | None = None,
    function_name: str | None = None,
    concurrency: int = 1,
    debug: bool = False,
    force: bool = False,
) -> dict:
    """Run coverage-feedback regeneration for all (or one) candidate functions.

    Returns dict mapping function_name -> list of iteration records.
    """
    from memory.observed_memory.coverage_orch import _load_gt_locations_from_jsonl

    _suppress_verbose_logging()

    cands_path = candidates_path(framework, example)
    if not cands_path.exists():
        raise RuntimeError(
            f"Candidates file not found: {cands_path}\n"
            f"Run 'memory select' first."
        )

    candidates = json.loads(cands_path.read_text(encoding="utf-8"))

    # Filter to functions that need regeneration:
    # - Functions with coverage below threshold
    # - Functions with execution failures (error-only coverage JSON)
    eligible = []
    results = FeedbackResults()
    terminal_budget = _feedback_budget(
        terminal_max_iterations or max_iterations,
        terminal_max_retries or max_retries,
    )
    for c in candidates:
        fn = c["function_name"]
        if function_name and fn != function_name:
            continue
        cov_path = coverage_result_path(framework, example, fn)
        if not cov_path.exists():
            print(f"  SKIP {fn}: no coverage data (run 'memory coverage' first)")
            continue
        cov_data = json.loads(cov_path.read_text(encoding="utf-8"))
        is_failure = coverage_has_execution_failure(cov_data)
        current_lc = cov_data.get("line_coverage", 0)

        if coverage_ready(cov_data, coverage_threshold):
            print(f"  SKIP {fn}: coverage {current_lc:.1%} already >= threshold")
            current_status = reusable_terminal_status(
                feedback_status_path(framework, example, fn),
                expected_fingerprint=_feedback_fingerprint(
                    c,
                    framework,
                    example,
                    model,
                    base_url,
                    coverage_threshold,
                ),
                min_budget=terminal_budget,
            )
            if not current_status or current_status.get("status") != "success":
                coverage_metadata = cov_data.get("_praxis_coverage")
                coverage_fingerprint = (
                    coverage_metadata.get("fingerprint")
                    if isinstance(coverage_metadata, dict)
                    else None
                )
                current_status = write_status(
                    feedback_status_path(framework, example, fn),
                    status="success",
                    fingerprint_value=_feedback_fingerprint(
                        c,
                        framework,
                        example,
                        model,
                        base_url,
                        coverage_threshold,
                    ),
                    budget=terminal_budget,
                    model=model,
                    reason="validated existing Stage 2-ready coverage",
                    details={
                        "line_coverage": float(current_lc or 0.0),
                        "is_execution_failure": False,
                        "stage2_case_gate": stage2_case_gate(cov_data),
                        "attempts": len(
                            _read_feedback_history(
                                feedback_log_path(
                                    framework,
                                    example,
                                    fn,
                                )
                            )
                        ),
                        "latest_run_attempts": 0,
                        "coverage_fingerprint": coverage_fingerprint,
                        "adaptive_policy": "balanced-v1",
                        "early_stop_reason": "",
                    },
                )
            continue
        if not force:
            existing_status = load_status(
                feedback_status_path(framework, example, fn)
            )
            existing_budget = (
                existing_status.get("budget")
                if isinstance(existing_status, dict)
                else {}
            )
            existing_details = (
                existing_status.get("details")
                if isinstance(existing_status, dict)
                else {}
            )
            coverage_metadata = cov_data.get("_praxis_coverage")
            current_coverage_fingerprint = (
                coverage_metadata.get("fingerprint")
                if isinstance(coverage_metadata, dict)
                else None
            )
            if (
                isinstance(existing_status, dict)
                and existing_status.get("status")
                in {"exhausted", "unrunnable"}
                and existing_status.get("model") == model
                and isinstance(existing_budget, dict)
                and int(existing_budget.get("max_iterations") or 0)
                >= terminal_budget["max_iterations"]
                and int(existing_budget.get("max_retries") or 0)
                >= terminal_budget["max_retries"]
                and isinstance(existing_details, dict)
                and current_coverage_fingerprint
                and existing_details.get("coverage_fingerprint")
                == current_coverage_fingerprint
            ):
                status = write_status(
                    feedback_status_path(framework, example, fn),
                    status=existing_status["status"],
                    fingerprint_value=_feedback_fingerprint(
                        c,
                        framework,
                        example,
                        model,
                        base_url,
                        coverage_threshold,
                    ),
                    budget=terminal_budget,
                    model=model,
                    reason=existing_status.get("reason", ""),
                    details={
                        **existing_details,
                        "adaptive_policy": "balanced-v1",
                        "migrated_terminal_state": True,
                    },
                )
                print(
                    f"  SKIP {fn}: migrated terminal feedback status "
                    f"{status['status']}",
                    flush=True,
                )
                results[fn] = {
                    "log": [],
                    "state": status,
                }
                continue
            status = reusable_terminal_status(
                feedback_status_path(framework, example, fn),
                expected_fingerprint=_feedback_fingerprint(
                    c,
                    framework,
                    example,
                    model,
                    base_url,
                    coverage_threshold,
                ),
                min_budget=terminal_budget,
            )
            if status:
                print(
                    f"  SKIP {fn}: reusable feedback status "
                    f"{status['status']}",
                    flush=True,
                )
                results[fn] = {
                    "log": [],
                    "state": status,
                }
                continue

        if is_failure:
            print(f"  [{fn}] execution failure — will attempt fix")
        eligible.append(c)

    if not eligible:
        print("  No functions need regeneration.")
        results.complete = True
        return results

    print(f"  {len(eligible)} function(s) below {coverage_threshold:.0%} threshold")
    print(
        "  adaptive iteration budgets="
        f"{_feedback_iteration_budgets(max_iterations, terminal_budget['max_iterations'], terminal_budget['max_retries'])}, "
        f"concurrency={concurrency}",
        flush=True,
    )

    gt_locations = _load_gt_locations_from_jsonl(framework, example)

    def _worker(candidate):
        fn = candidate["function_name"]
        log, state = _run_feedback_for_one(
            fn, candidate, framework, example,
            model, api_key, base_url, max_iterations,
            coverage_threshold, max_retries, terminal_budget,
            gt_locations, debug,
        )
        return fn, log, state

    if concurrency <= 1:
        for i, candidate in enumerate(eligible, 1):
            fn = candidate["function_name"]
            print(f"\n--- [{i}/{len(eligible)}] {fn} ---", flush=True)
            _, log, state = _worker(candidate)
            results[fn] = {"log": log, "state": state}
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, c): c for c in eligible}
            for future in as_completed(futures):
                fn, log, state = future.result()
                results[fn] = {"log": log, "state": state}
                print(f"  [{fn}] feedback loop done", flush=True)

    # Summary
    total = len(results)
    improved = sum(
        1 for log in results.values()
        if log["log"]
        and log["log"][-1].get("coverage_after", 0)
        > log["log"][0].get("coverage_before", 0)
    )
    error_fixed = sum(
        1 for log in results.values()
        if log["log"]
        and log["log"][-1].get("test_input_saved")
        and not log["log"][-1].get("last_error")
    )
    threshold_met = sum(
        1 for result in results.values()
        if result["state"].get("status") == "success"
    )
    state_counts = {}
    for result in results.values():
        state = result["state"].get("status", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
    results.complete = all(
        result["state"].get("status")
        in {"success", "exhausted", "unrunnable"}
        for result in results.values()
    )
    print(
        f"\n  Feedback loop done: {total} processed, "
        f"{improved} improved, {error_fixed} fixed, {threshold_met} met threshold; "
        f"states={state_counts}",
        flush=True,
    )
    return results
