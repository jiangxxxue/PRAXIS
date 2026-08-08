"""Stage 3: Agent generates requirement docs and test inputs for each candidate."""

import glob
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from agent.sdk import run_sdk_agent
from memory.config import (
    code_dir,
    candidates_path,
    generate_log_path,
    generate_status_path,
    requirement_path,
    test_input_path,
    observed_knowledge_path,
)
from memory.observed_memory.status import (
    fingerprint,
    is_non_retryable_error,
    is_transient_error,
    reusable_terminal_status,
    write_status,
)
from memory.observed_memory.validate_test_input import validate_test_input
from memory.observed_memory.workspace import (
    benchmark_target_locations,
    build_single_stub_workspace,
)
from runner import _parse_impl_location

# Max retries when test_input validation fails
_MAX_VALIDATION_RETRIES = 2


class GenerateResults(dict):
    complete = False


def _generate_budget(max_iterations: int, max_attempts: int) -> dict[str, int]:
    return {
        "max_iterations": max_iterations,
        "max_attempts": max_attempts,
    }


def _generate_iteration_budgets(
    max_iterations: int,
    terminal_max_iterations: int,
) -> list[int]:
    terminal = max(max_iterations, terminal_max_iterations)
    budgets = [max_iterations]
    if max_iterations < terminal:
        budgets.append(min(max(max_iterations, 160), terminal))
    if budgets[-1] < terminal:
        budgets.append(terminal)
    return list(dict.fromkeys(budgets))


def _read_attempt_log(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _write_attempt_log(path: Path, attempts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(attempts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _generation_retry_instructions(result: dict) -> str:
    errors = result.get("validation_errors") or []
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        return (
            "\n\n## PREVIOUS ATTEMPT ERRORS (FIX THESE)\n\n"
            f"{details}\n\n"
            "Produce the requested test_input.py and fix every error above."
        )
    return (
        "\n\n## PREVIOUS ATTEMPT DID NOT PRODUCE THE REQUIRED FILE\n\n"
        "You must write a valid *_test_input.py file before finishing."
    )


def _generate_fingerprint(
    candidate: dict,
    framework: str,
    example: str,
    model: str,
    base_url: str,
) -> str:
    rel_path, _start, _end = _parse_impl_location(
        candidate["implementation_location"]
    )
    source_path = code_dir(framework, example) / rel_path
    prompt_path = Path(__file__).resolve().parent / "prompts" / "generate.md"
    static_memory_path = observed_knowledge_path(framework, example)
    return fingerprint(
        {
            "stage": "generate",
            "framework": framework,
            "example": example,
            "candidate": candidate,
            "model": model,
            "base_url": base_url,
        },
        [
            source_path,
            prompt_path,
            static_memory_path,
            Path(__file__),
            Path(validate_test_input.__code__.co_filename),
            Path(build_single_stub_workspace.__code__.co_filename),
        ],
    )


def _suppress_verbose_logging():
    """Suppress SDK/litellm verbose logging that produces blank lines."""
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    for name in ("litellm", "httpx", "httpcore", "openai", "httpcore.http11"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Also suppress the DeprecationWarning from SDK
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def _find_file_in_workspace(ws: str, pattern: str) -> str | None:
    """Recursively search workspace for a file matching glob pattern."""
    matches = glob.glob(os.path.join(ws, "**", pattern), recursive=True)
    return matches[0] if matches else None


def _extract_python_block(content: str) -> str | None:
    """Extract Python code from markdown code fences in agent output."""
    import re
    # Match ```python ... ``` or ```py ... ``` blocks
    pattern = r"```(?:python|py)\s*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        # Return the longest match (most likely the full file)
        return max(matches, key=len).strip()
    return None


def _run_agent_and_extract(
    func_name: str,
    impl_loc: str,
    rel_path: str,
    start: int,
    end: int,
    code_root: str,
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str,
    max_iterations: int,
    debug: bool,
    extra_instructions: str = "",
) -> dict:
    """Run the SDK agent once and extract output files. Returns a result dict."""
    tmp_dir_ctx: tempfile.TemporaryDirectory | None = None
    if debug:
        debug_dir = Path("debug_generate") / framework / example / func_name
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

        prompt_text = (
            Path(__file__).resolve().parent
            / "prompts"
            / "generate.md"
        ).read_text(encoding="utf-8")
        prompt_text = prompt_text.replace("{function_name}", func_name)
        prompt_text = prompt_text.replace("{implementation_location}", impl_loc)
        if extra_instructions:
            prompt_text += extra_instructions

        sm_path = observed_knowledge_path(framework, example)
        if sm_path.exists():
            static_mem = sm_path.read_text(encoding="utf-8")
            prompt_text += f"\n\n## Reference: Project Knowledge Summary\n\n{static_mem}"

        events, status = run_sdk_agent(
            prompt=prompt_text,
            workspace=paths["workspace"],
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_iterations=max_iterations,
            corpus_dirs=[paths["code"]],
        )

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
            events_file.write_text(json.dumps(events_data, ensure_ascii=False, indent=2), encoding="utf-8")

        ws = paths["workspace"]
        result = {"function_name": func_name, "status": str(status)}

        # Extract requirement
        req_file = os.path.join(ws, f"{func_name}_requirement.md")
        if not os.path.exists(req_file):
            req_file = _find_file_in_workspace(ws, f"*{func_name}*requirement*.md")
            if not req_file:
                req_file = _find_file_in_workspace(ws, "*requirement*.md")
        if req_file and os.path.exists(req_file):
            result["requirement"] = Path(req_file).read_text(encoding="utf-8")
        else:
            for event in reversed(events):
                content = getattr(event, "content", None) or ""
                if "## Function Overview" in content and func_name in content:
                    result["requirement"] = content
                    break

        # Extract test_input
        test_file = os.path.join(ws, f"{func_name}_test_input.py")
        if not os.path.exists(test_file):
            test_file = _find_file_in_workspace(ws, f"*{func_name}*test_input*.py")
            if not test_file:
                test_file = _find_file_in_workspace(ws, "*test_input*.py")
        if test_file and os.path.exists(test_file):
            result["test_input"] = Path(test_file).read_text(encoding="utf-8")
        else:
            for event in reversed(events):
                content = getattr(event, "content", None) or ""
                if "test_cases" in content and ("normal" in content or "edge" in content):
                    py_block = _extract_python_block(content)
                    if py_block:
                        result["test_input"] = py_block
                        break

        return result
    finally:
        if tmp_dir_ctx:
            tmp_dir_ctx.__exit__(None, None, None)


def _process_one_candidate(
    candidate: dict,
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str,
    max_iterations: int,
    debug: bool = False,
    max_validation_retries: int = _MAX_VALIDATION_RETRIES,
    initial_extra_instructions: str = "",
) -> dict:
    """Run agent for a single candidate function with validation and retry.

    If the generated test_input fails validation (syntax error, mock.patch
    target not found, missing imports), the error is fed back to the agent
    for a retry. Up to _MAX_VALIDATION_RETRIES retries.
    """
    func_name = candidate["function_name"]
    impl_loc = candidate["implementation_location"]
    rel_path, start, end = _parse_impl_location(impl_loc)
    if not rel_path:
        raise RuntimeError(f"Cannot parse implementation_location: {impl_loc}")

    code_root = str(code_dir(framework, example))

    extra = initial_extra_instructions
    for attempt in range(1 + max_validation_retries):
        result = _run_agent_and_extract(
            func_name, impl_loc, rel_path, start, end, code_root,
            framework, example, model, api_key, base_url,
            max_iterations, debug, extra_instructions=extra,
        )

        # Print extraction results
        if "requirement" in result:
            print(f"  [{func_name}] requirement OK", flush=True)
        else:
            print(f"  [{func_name}] requirement NOT FOUND", flush=True)

        if "test_input" in result:
            print(f"  [{func_name}] test_input extracted ({len(result['test_input'])} chars)", flush=True)
        else:
            print(f"  [{func_name}] test_input NOT FOUND", flush=True)

        print(f"  [{func_name}] agent_status={result['status']}", flush=True)

        # Validate every extracted test_input before it can be persisted.
        if "test_input" in result:
            errors = validate_test_input(
                result["test_input"],
                code_root,
                framework,
                implementation_location=impl_loc,
            )
            if not errors:
                break  # Valid, done

            if attempt >= max_validation_retries:
                result.pop("test_input", None)
                result["failure_reason"] = "validation retries exhausted"
                result["validation_errors"] = errors
                print(f"  [{func_name}] validation failed after final attempt", flush=True)
                print("    Errors: " + "\n    ".join(errors))
                break

            # Build retry instructions
            error_list = "\n".join(f"- {e}" for e in errors)
            extra = (
                f"\n\n## PREVIOUS ATTEMPT ERRORS (FIX THESE)\n\n"
                f"Your last test_input had these validation problems:\n\n"
                f"{error_list}\n\n"
                f"Instructions:\n"
                f"- Fix ALL of the above issues in your new test_input.py\n"
                f"- For mock.patch targets: the target module/function MUST exist. "
                f"Use the terminal to grep/search for the exact function/class name before writing mock.patch.\n"
                f"- For imports: only use stdlib modules or packages known to be installed "
                f"in the evaluation environment.\n"
            )
            print(f"  [{func_name}] validation failed (attempt {attempt+1}), retrying...")
            print(f"    Errors: {error_list}")
        else:
            result["failure_reason"] = "agent did not produce test_input"
            break

    return result


def run_generate(
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 100,
    terminal_max_iterations: int | None = None,
    concurrency: int = 1,
    debug: bool = False,
    force: bool = False,
) -> dict:
    """Stage 3: Generate requirement + test input for each candidate.

    Returns dict mapping function_name -> {"requirement": Path, "test_input": Path}.
    """
    _suppress_verbose_logging()

    cands_path = candidates_path(framework, example)
    if not cands_path.exists():
        raise RuntimeError(
            f"Candidates file not found: {cands_path}\n"
            f"Run 'memory select' first."
        )

    candidates = json.loads(cands_path.read_text(encoding="utf-8"))

    print(f"  Generating for {len(candidates)} candidates (concurrency={concurrency})")
    print(f"  max_iterations={max_iterations}, debug={debug}", flush=True)

    iteration_budgets = _generate_iteration_budgets(
        max_iterations,
        terminal_max_iterations or max_iterations,
    )
    terminal_budget = _generate_budget(
        iteration_budgets[-1],
        len(iteration_budgets),
    )
    print(f"  adaptive iteration budgets={iteration_budgets}", flush=True)
    results = GenerateResults()

    def _worker(candidate):
        func_name = candidate["function_name"]
        status_path = generate_status_path(framework, example, func_name)
        log_path = generate_log_path(framework, example, func_name)
        fingerprint_value = _generate_fingerprint(
            candidate,
            framework,
            example,
            model,
            base_url,
        )
        ti_path = test_input_path(framework, example, func_name)
        existing_input_invalid = False

        if not force and ti_path.is_file() and ti_path.stat().st_size > 0:
            existing_errors = validate_test_input(
                ti_path.read_text(encoding="utf-8"),
                str(code_dir(framework, example)),
                framework,
                implementation_location=candidate["implementation_location"],
            )
            if not existing_errors:
                status = write_status(
                    status_path,
                    status="success",
                    fingerprint_value=fingerprint_value,
                    budget=terminal_budget,
                    model=model,
                    reason="validated existing test_input",
                    details={"adaptive_policy": "balanced-v1"},
                )
                print(
                    f"  SKIP {func_name}: validated existing test_input",
                    flush=True,
                )
                return func_name, {"_state": status}
            existing_input_invalid = True
            print(
                f"  [{func_name}] existing test_input failed current "
                "validation; regenerating",
                flush=True,
            )

        if not force and not existing_input_invalid:
            status = reusable_terminal_status(
                status_path,
                expected_fingerprint=fingerprint_value,
                min_budget=terminal_budget,
            )
            if status:
                if status["status"] == "success" and not (
                    ti_path.is_file() and ti_path.stat().st_size > 0
                ):
                    status = None
            if status:
                print(
                    f"  SKIP {func_name}: reusable generate status "
                    f"{status['status']}",
                    flush=True,
                )
                return func_name, {"_state": status}

        rel_path, _start, _end = _parse_impl_location(
            candidate["implementation_location"]
        )
        source_path = code_dir(framework, example) / rel_path
        if not rel_path or not source_path.is_file():
            status = write_status(
                status_path,
                status="unrunnable",
                fingerprint_value=fingerprint_value,
                budget=terminal_budget,
                model=model,
                reason=f"source file not found: {rel_path or '<unknown>'}",
            )
            return func_name, {"_state": status}

        attempt_log = [] if force else _read_attempt_log(log_path)
        extra_instructions = ""
        result = {}
        last_error = ""
        latest_error_transient = False
        non_retryable = False
        run_attempts = 0

        for attempt, attempt_budget in enumerate(iteration_budgets, 1):
            run_attempts += 1
            started = time.monotonic()
            print(
                f"  [{func_name}] adaptive attempt {attempt}/"
                f"{len(iteration_budgets)}: max_iterations={attempt_budget}",
                flush=True,
            )
            try:
                result = _process_one_candidate(
                    candidate,
                    framework,
                    example,
                    model,
                    api_key,
                    base_url,
                    attempt_budget,
                    debug=debug,
                    max_validation_retries=0,
                    initial_extra_instructions=extra_instructions,
                )
                last_error = str(result.get("failure_reason") or "")
                if not last_error and result.get("validation_errors"):
                    last_error = "Validation errors: " + "; ".join(
                        str(error) for error in result["validation_errors"]
                    )
            except Exception as exc:
                result = {}
                last_error = str(exc)

            transient = is_transient_error(last_error)
            non_retryable = is_non_retryable_error(last_error)
            latest_error_transient = transient
            attempt_log.append({
                "attempt": len(attempt_log) + 1,
                "stage_attempt": attempt,
                "max_iterations": attempt_budget,
                "duration_seconds": round(time.monotonic() - started, 3),
                "has_requirement": "requirement" in result,
                "has_test_input": "test_input" in result,
                "validation_errors": result.get("validation_errors", []),
                "error": last_error,
                "transient": transient,
                "timestamp": datetime.now().isoformat(),
            })
            _write_attempt_log(log_path, attempt_log)

            if result.get("test_input"):
                break
            if non_retryable:
                print(
                    f"  [{func_name}] non-retryable request error; stopping",
                    flush=True,
                )
                break
            extra_instructions = _generation_retry_instructions(result)

        if result.get("test_input"):
            terminal_status = "success"
        elif latest_error_transient and not non_retryable:
            terminal_status = "retryable"
        else:
            terminal_status = "exhausted"
        _save_result(framework, example, func_name, result)
        fingerprint_value = _generate_fingerprint(
            candidate,
            framework,
            example,
            model,
            base_url,
        )
        status = write_status(
            status_path,
            status=terminal_status,
            fingerprint_value=fingerprint_value,
            budget=terminal_budget,
            model=model,
            reason=last_error,
            details={
                "agent_status": str(result.get("status", "unknown")),
                "has_requirement": "requirement" in result,
                "has_test_input": "test_input" in result,
                "validation_errors": result.get("validation_errors", []),
                "adaptive_policy": "balanced-v1",
                "attempts": len(attempt_log),
                "latest_run_attempts": run_attempts,
            },
        )
        result["_state"] = status
        return func_name, result

    if concurrency <= 1:
        for i, candidate in enumerate(candidates, 1):
            print(f"\n--- [{i}/{len(candidates)}] {candidate['function_name']} ---", flush=True)
            func_name, result = _worker(candidate)
            results[func_name] = result
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, c): c for c in candidates}
            for future in as_completed(futures):
                func_name, result = future.result()
                results[func_name] = result
                print(f"  [{func_name}] done", flush=True)

    # Summary
    ok = sum(1 for r in results.values() if "requirement" in r and "test_input" in r)
    partial = sum(1 for r in results.values() if "requirement" in r or "test_input" in r) - ok
    fail = len(results) - ok - partial
    state_counts = {}
    for result in results.values():
        state = result.get("_state", {}).get("status", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
    results.complete = bool(results) and all(
        result.get("_state", {}).get("status")
        in {"success", "exhausted", "unrunnable"}
        for result in results.values()
    )
    print(
        f"\n  Stage 3 done: {ok} complete, {partial} partial, {fail} failed; "
        f"states={state_counts}",
        flush=True,
    )
    return results


def _save_result(framework: str, example: str, func_name: str, result: dict):
    """Save requirement and test_input to derived/.

    Also saves a copy using the leaf name (without class prefix) so the
    coverage stage can find the file regardless of naming convention.
    """
    leaf_name = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name

    if "requirement" in result:
        out = requirement_path(framework, example, func_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["requirement"], encoding="utf-8")

    if "test_input" in result:
        out = test_input_path(framework, example, func_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["test_input"], encoding="utf-8")
        # Also save leaf-name copy for coverage stage fallback
        if leaf_name != func_name:
            out_leaf = test_input_path(framework, example, leaf_name)
            out_leaf.parent.mkdir(parents=True, exist_ok=True)
            out_leaf.write_text(result["test_input"], encoding="utf-8")
