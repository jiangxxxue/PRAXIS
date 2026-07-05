import os

from runner import (
    _parse_impl_location,
    _extract_function_from_file,
    _extract_from_events,
    build_prompt,
)
from agent.sdk import run_sdk_agent

from .grader import _callable_name, _spec_callable_name
from .prompts import build_practice_prompt


def run_practice_attempt(spec, iteration, prior_attempts,
                         *, model, api_key, base_url, repo_paths,
                         provider=None, api_version=None,
                         max_iterations=None):
    """Run one practice attempt on an existing workspace.

    Workspace is NOT created here — caller passes repo_paths from a
    pre-built practice workspace and handles re-stubbing between iterations.
    """
    rel, start, end = _parse_impl_location(spec["implementation_location"])
    stub_file = os.path.join(repo_paths["code"], rel)

    base_prompt = build_prompt(
        spec["source_record"], spec["framework"], repo_paths,
        stub_file=stub_file, stub_line=start,
    )
    prompt = build_practice_prompt(base_prompt, prior_attempts)

    if max_iterations is None:
        max_iterations = int(os.environ.get("KOCO_PRACTICE_MAX_ITERATIONS", "100"))
    events, status = run_sdk_agent(
        prompt=prompt,
        workspace=repo_paths["workspace"],
        model=model, api_key=api_key, base_url=base_url,
        max_iterations=max_iterations,
        corpus_dirs=[repo_paths["knowledge_corpus"], repo_paths["code"]],
    )

    callable_name = _spec_callable_name(spec)
    impl = _extract_function_from_file(stub_file, callable_name)
    if not impl:
        impl = _extract_from_events(events, callable_name)
    if not impl and callable_name != spec["function_name"]:
        impl = _extract_from_events(events, spec["function_name"])

    return {
        "iteration": iteration,
        "completion": impl or "",
        "sdk_status": status.value if status else "unknown",
    }
