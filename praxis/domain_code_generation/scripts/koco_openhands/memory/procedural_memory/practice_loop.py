import json
import os
import sys
import tempfile
from pathlib import Path

from runner import _parse_impl_location

from .config import SCRIPTS_DIR, trace_path_for_spec
from .workspace import build_practice_ws
from .harness import run_practice_attempt
from .grader import grade_completion

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def run_practice(spec, *, K=8, model, api_key, base_url,
                 provider=None, base_model="", api_version=None,
                 max_agent_iterations=None,
                 **_unused):
    """Run up to K practice iterations: re-stub → agent → extract → grade → record.

    Stops early on the first PASS. Single workspace in temp directory,
    re-stubbed between iterations.
    """
    trace = {
        "spec": spec,
        "iterations": [],
        "K_requested": K,
        "llm": {
            "agent": "openhands",
            "provider": provider or "",
            "model": model,
            "base_model": base_model or "",
            "base_url": base_url,
            "api_version": api_version or "",
        },
    }
    prior_attempts = []
    stopped_early = False

    trace_path = trace_path_for_spec(spec)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _persist():
        trace["non_empty_count"] = sum(1 for it in trace["iterations"]
                                       if it["completion"])
        trace["iterations_ran"] = len(trace["iterations"])
        trace["stopped_early"] = stopped_early
        trace_path.write_text(json.dumps(trace, indent=2, default=str))

    try:
        for i in range(K):
            # Each attempt receives a fresh workspace. An agent that violates
            # the prompt and edits another file cannot influence later attempts.
            with tempfile.TemporaryDirectory(prefix="koco_practice_") as tmp_dir:
                workspace_root = Path(tmp_dir) / "ws"
                repo_paths = build_practice_ws(spec, workspace_root)

                rel, _, _ = _parse_impl_location(spec["implementation_location"])
                stub_file = os.path.join(repo_paths["code"], rel)
                print(f"practice: iter {i + 1}/{K} — running agent ...")
                attempt = run_practice_attempt(
                    spec, i, prior_attempts,
                    model=model, api_key=api_key, base_url=base_url,
                    provider=provider, api_version=api_version,
                    repo_paths=repo_paths,
                    max_iterations=max_agent_iterations,
                )
                completion_len = len(attempt["completion"] or "")
                print(f"practice: iter {i + 1}/{K} — completion={completion_len} chars, "
                      f"grading ...")
                grade = grade_completion(spec, attempt["completion"])
                if grade.get("infra_failure"):
                    status = "INFRA_FAILURE"
                elif grade.get("passed"):
                    status = "PASS"
                else:
                    status = "FAIL"
                print(f"practice: iter {i + 1}/{K} — {status}")
                iteration = {**attempt, "grade": grade}
                trace["iterations"].append(iteration)
                _persist()

                prior_attempts.append({
                    "completion": attempt["completion"],
                    "passed": grade.get("passed", False),
                    "infra_failure": grade.get("infra_failure", False),
                    "pass_ratio": grade.get("pass_ratio", 0.0),
                    "per_case": grade.get("per_case", []),
                })

                if grade.get("passed"):
                    print(f"practice: first PASS at iter {i + 1}/{K} — stopping early")
                    stopped_early = True
                    break
    finally:
        _persist()

    return trace
