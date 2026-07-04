"""Stage 3: Agent generates requirement docs and test inputs for each candidate."""

import glob
import json
import logging
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.sdk import run_sdk_agent
from memory.config import (
    code_dir,
    candidates_path,
    requirement_path,
    test_input_path,
    observed_knowledge_path,
)
from memory.observed_memory.validate_test_input import validate_test_input
from memory.observed_memory.workspace import build_single_stub_workspace
from runner import _parse_impl_location

# Max retries when test_input validation fails
_MAX_VALIDATION_RETRIES = 2


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
            code_root, {}, rel_path, start, end, tmp_dir
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

    extra = ""
    for attempt in range(1 + _MAX_VALIDATION_RETRIES):
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

        # Validate test_input if we have one
        if "test_input" in result and attempt < 1 + _MAX_VALIDATION_RETRIES:
            errors = validate_test_input(
                result["test_input"], code_root, framework,
            )
            if not errors:
                break  # Valid, done

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
            break

    return result


def run_generate(
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 100,
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

    # Skip candidates that already have test_input (unless --force)
    if not force:
        to_process = []
        for c in candidates:
            ti = test_input_path(framework, example, c["function_name"])
            if ti.exists():
                print(f"  SKIP {c['function_name']}: test_input already exists")
            else:
                to_process.append(c)
        skipped = len(candidates) - len(to_process)
        if skipped:
            print(f"  Skipped {skipped} candidates with existing test_input")
        candidates = to_process

    print(f"  Generating for {len(candidates)} candidates (concurrency={concurrency})")
    print(f"  max_iterations={max_iterations}, debug={debug}", flush=True)

    results = {}

    def _worker(candidate):
        return candidate["function_name"], _process_one_candidate(
            candidate, framework, example, model, api_key, base_url,
            max_iterations, debug=debug,
        )

    if concurrency <= 1:
        for i, candidate in enumerate(candidates, 1):
            print(f"\n--- [{i}/{len(candidates)}] {candidate['function_name']} ---", flush=True)
            func_name, result = _worker(candidate)
            results[func_name] = result
            _save_result(framework, example, func_name, result)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, c): c for c in candidates}
            for future in as_completed(futures):
                func_name, result = future.result()
                results[func_name] = result
                _save_result(framework, example, func_name, result)
                print(f"  [{func_name}] done", flush=True)

    # Summary
    ok = sum(1 for r in results.values() if "requirement" in r and "test_input" in r)
    partial = sum(1 for r in results.values() if "requirement" in r or "test_input" in r) - ok
    fail = len(results) - ok - partial
    print(f"\n  Stage 3 done: {ok} complete, {partial} partial, {fail} failed", flush=True)
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
