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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from agent.sdk import run_sdk_agent
from memory.config import (
    code_dir,
    candidates_path,
    coverage_result_path,
    feedback_log_path,
    observed_knowledge_path,
    test_input_path,
)
from memory.observed_memory.validate_test_input import validate_test_input
from memory.observed_memory.workspace import build_single_stub_workspace
from runner import _parse_impl_location

# Max retries when feedback-generated test_input fails validation
_MAX_FEEDBACK_VALIDATION_RETRIES = 2


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
        paths = build_single_stub_workspace(code_root, {}, rel_path, start, end, tmp_dir)

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
    gt_locations: dict | None,
    debug: bool = False,
) -> list[dict]:
    """Run the feedback loop for a single function. Returns iteration log."""
    from memory.observed_memory.coverage_orch import run_coverage_single

    cov_path = coverage_result_path(framework, example, func_name)
    if not cov_path.exists():
        # This shouldn't happen (caller filters), but handle gracefully
        print(f"  SKIP {func_name}: no coverage data")
        return []

    # Load initial coverage data
    initial_coverage = json.loads(cov_path.read_text(encoding="utf-8"))
    is_error_only = initial_coverage.get("is_execution_failure", False)

    # Load source code once
    impl_loc = candidate["implementation_location"]
    rel_path, line_start, line_end = _parse_impl_location(impl_loc)
    if gt_locations and func_name in gt_locations:
        entry = gt_locations[func_name]
        rel_path, line_start, line_end = entry[0], entry[1], entry[2]

    source_file = code_dir(framework, example) / rel_path
    if not source_file.exists():
        print(f"  SKIP {func_name}: source file not found ({rel_path})")
        return []

    source_lines = source_file.read_text(encoding="utf-8").splitlines()

    iteration_log = []
    last_error = ""

    for attempt in range(1, max_retries + 1):
        # Load current coverage
        if not cov_path.exists():
            print(f"  [{func_name}] coverage file disappeared, stopping")
            break

        current_coverage = json.loads(cov_path.read_text(encoding="utf-8"))
        current_is_error = current_coverage.get("is_execution_failure", False)
        current_lc = current_coverage.get("line_coverage", 0) or 0

        if not current_is_error and current_lc >= coverage_threshold:
            print(f"  [{func_name}] coverage {current_lc:.1%} >= threshold {coverage_threshold:.1%}, done")
            break

        missing_count = len(current_coverage.get("missing_lines", []))
        error_count = current_coverage.get("num_execution_errors", 0)
        label = "FIX ERROR" if current_is_error else f"coverage={current_lc:.1%}"
        print(
            f"  [{func_name}] attempt {attempt}/{max_retries}: "
            f"{label}, missing={missing_count} lines, errors={error_count}",
            flush=True,
        )

        # Regenerate
        try:
            result = _regenerate_one_candidate(
                candidate, framework, example, current_coverage,
                source_lines, line_start, line_end,
                model, api_key, base_url, max_iterations,
                attempt, max_retries, debug, last_error,
            )
        except Exception as exc:
            print(f"  [{func_name}] agent call failed: {exc}", flush=True)
            last_error = str(exc)
            import random as _random
            import time as _time
            _wait = _random.uniform(5, 15)
            print(f"  [{func_name}] waiting {_wait:.1f}s before next attempt...", flush=True)
            _time.sleep(_wait)
            iteration_log.append({
                "attempt": attempt,
                "coverage_before": current_lc,
                "coverage_after": current_lc,
                "agent_status": f"error: {exc}",
                "has_test_input": False,
                "test_input_saved": False,
                "timestamp": datetime.now().isoformat(),
            })
            continue

        new_lc = current_lc
        test_input_saved = False
        last_error = ""

        if result.get("test_input"):
            # Validate before saving
            code_root = str(code_dir(framework, example))
            errors = validate_test_input(result["test_input"], code_root, framework)
            if errors:
                print(f"  [{func_name}] validation failed, feeding back to agent")
                last_error = "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)
                iteration_log.append({
                    "attempt": attempt,
                    "coverage_before": current_lc,
                    "coverage_after": current_lc,
                    "agent_status": "validation_failed",
                    "has_test_input": True,
                    "test_input_saved": False,
                    "last_error": last_error,
                    "timestamp": datetime.now().isoformat(),
                })
                continue  # retry with error feedback

            # Save and re-run coverage
            ti_path = test_input_path(framework, example, func_name)
            old_test_input = ti_path.read_text(encoding="utf-8") if ti_path.exists() else ""
            ti_path.write_text(result["test_input"], encoding="utf-8")

            try:
                new_cov = run_coverage_single(
                    framework, example, func_name,
                    per_test=False, capture_output=True,
                    gt_locations=gt_locations,
                )
            except Exception as exc:
                print(f"  [{func_name}] coverage re-run FAILED: {exc}, reverting")
                last_error = str(exc)
                ti_path.write_text(old_test_input, encoding="utf-8")
                new_cov = None

            if new_cov:
                new_is_error = new_cov.get("is_execution_failure", False)
                new_lc = new_cov.get("line_coverage", 0) or 0

                if new_is_error:
                    print(f"  [{func_name}] still failing after regen")
                    last_error = "Still failed to execute"
                else:
                    print(f"  [{func_name}] after regen: {new_lc:.1%} (was {current_lc:.1%})")

                    if not current_is_error and new_lc < current_lc:
                        print(
                            f"  [{func_name}] regressed {current_lc:.1%} -> {new_lc:.1%}, reverting"
                        )
                        ti_path.write_text(old_test_input, encoding="utf-8")
                        last_error = f"Coverage regressed from {current_lc:.1%} to {new_lc:.1%}"
                        run_coverage_single(
                            framework, example, func_name,
                            per_test=False, capture_output=True,
                            gt_locations=gt_locations,
                        )
                        new_lc = current_lc
                    else:
                        test_input_saved = True
            else:
                print(f"  [{func_name}] coverage re-run FAILED, reverting")
                ti_path.write_text(old_test_input, encoding="utf-8")
                new_lc = current_lc
        else:
            print(f"  [{func_name}] agent did not produce test_input")
            last_error = "Agent did not produce a valid test_input file"

        iteration_log.append({
            "attempt": attempt,
            "coverage_before": current_lc,
            "coverage_after": new_lc,
            "agent_status": str(result.get("status", "unknown")),
            "has_test_input": "test_input" in result,
            "test_input_saved": test_input_saved,
            "last_error": last_error,
            "timestamp": datetime.now().isoformat(),
        })

        if not current_is_error:
            if new_lc >= coverage_threshold:
                print(f"  [{func_name}] coverage {new_lc:.1%} >= threshold, done")
                break
        else:
            if test_input_saved:
                print(f"  [{func_name}] execution error FIXED, coverage now {new_lc:.1%}")
                break
    else:
        print(f"  [{func_name}] max retries ({max_retries}) exhausted")

    # Save iteration log
    log_path = feedback_log_path(framework, example, func_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(iteration_log, indent=2), encoding="utf-8")

    return iteration_log


# ── outer orchestrator ──────────────────────────────────────────────────────

def run_feedback_loop(
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 100,
    coverage_threshold: float = 0.5,
    max_retries: int = 3,
    function_name: str | None = None,
    concurrency: int = 1,
    debug: bool = False,
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
    for c in candidates:
        fn = c["function_name"]
        if function_name and fn != function_name:
            continue
        cov_path = coverage_result_path(framework, example, fn)
        if not cov_path.exists():
            print(f"  SKIP {fn}: no coverage data (run 'memory coverage' first)")
            continue
        cov_data = json.loads(cov_path.read_text(encoding="utf-8"))
        is_failure = cov_data.get("is_execution_failure", False)
        current_lc = cov_data.get("line_coverage", 0)

        if is_failure:
            print(f"  [{fn}] execution failure — will attempt fix")
            eligible.append(c)
        elif current_lc >= coverage_threshold:
            print(f"  SKIP {fn}: coverage {current_lc:.1%} already >= threshold")
        else:
            eligible.append(c)

    if not eligible:
        print("  No functions need regeneration.")
        return {}

    print(f"  {len(eligible)} function(s) below {coverage_threshold:.0%} threshold")
    print(f"  max_retries={max_retries}, concurrency={concurrency}", flush=True)

    gt_locations = _load_gt_locations_from_jsonl(framework, example)

    results = {}

    def _worker(candidate):
        fn = candidate["function_name"]
        return fn, _run_feedback_for_one(
            fn, candidate, framework, example,
            model, api_key, base_url, max_iterations,
            coverage_threshold, max_retries, gt_locations, debug,
        )

    if concurrency <= 1:
        for i, candidate in enumerate(eligible, 1):
            fn = candidate["function_name"]
            print(f"\n--- [{i}/{len(eligible)}] {fn} ---", flush=True)
            _, log = _worker(candidate)
            results[fn] = log
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, c): c for c in eligible}
            for future in as_completed(futures):
                fn, log = future.result()
                results[fn] = log
                print(f"  [{fn}] feedback loop done", flush=True)

    # Summary
    total = len(results)
    improved = sum(
        1 for log in results.values()
        if log and log[-1].get("coverage_after", 0) > log[0].get("coverage_before", 0)
    )
    error_fixed = sum(
        1 for log in results.values()
        if log and log[-1].get("test_input_saved") and not log[-1].get("last_error")
    )
    threshold_met = sum(
        1 for log in results.values()
        if log and log[-1].get("coverage_after", 0) >= coverage_threshold
    )
    print(
        f"\n  Feedback loop done: {total} processed, "
        f"{improved} improved, {error_fixed} fixed, {threshold_met} met threshold",
        flush=True,
    )
    return results
