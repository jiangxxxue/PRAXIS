#!/usr/bin/env python3
"""Procedural memory pipeline — plumbing pilot CLI.

Entry point: memory_pilot.py (same directory). See its docstring for usage.
"""
import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

from .config import (
    DEFAULT_DERIVED_SET,
    PROJECT_ROOT,
    SCRIPTS_DIR,
    TRACES_DIR,
    candidates_path,
    get_docker_image,
    oracle_path_for_spec,
    requirement_path,
    test_input_path,
    trace_function_dir,
    trace_path_for_spec,
)
from .grader import _run_differential, grade_completion
from .practice_loop import run_practice
from .structured import (
    consolidate_structured_example,
    distill_structured_trace,
)

from runner import _extract_function_from_file, _parse_impl_location


# Fixed system framing for all practice prompts. The per-function requirement.md
# from observed memory is threaded through as the user message by build_prompt.
_GLOBAL_SYSTEM = (
    "You are a senior Python engineer. Implement the function described below."
)
_DISPLAY_SUFFIX_RE = re.compile(r"\s+\([^)]*\)$")
_CLASS_DISPLAY_SUFFIX_RE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<class>[A-Za-z_][A-Za-z0-9_]*)\)$"
)
PROVIDER_CHOICES = ("openai", "azure_openai", "openrouter")


def _callable_display_name(function_name: str) -> str:
    match = _CLASS_DISPLAY_SUFFIX_RE.match(function_name)
    if match and "." not in match.group("name"):
        return f"{match.group('class')}.{match.group('name')}"
    return _DISPLAY_SUFFIX_RE.sub("", function_name)


def _callable_name_for_location(src_path: Path, function_name: str, start: int) -> str:
    """Resolve a bare function name to Class.method when the location is a method."""

    try:
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return function_name

    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    candidates = []
    leaf = _callable_display_name(function_name).split(".")[-1]
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= start <= end and node.name == leaf:
            candidates.append(node)
    if not candidates:
        return function_name

    candidates.sort(key=lambda item: (item.end_lineno - item.lineno, item.lineno))
    target = candidates[0]
    parent = parents.get(target)
    if isinstance(parent, ast.ClassDef):
        return f"{parent.name}.{target.name}"
    return target.name


# --------------------------------------------------------------------------- #
# init                                                                        #
# --------------------------------------------------------------------------- #

def cmd_init(args):
    """Build per-function pilot_spec from observed memory and run oracle.

    Reads observed_knowledge/{framework}/{example}/candidates.json and, for
    each candidate function (or only --function if given), reads its
    requirement and test_input, runs GT in Docker to produce the oracle, and
    writes pilot_spec.json under derived/procedural_knowledge/_traces/.
    """
    derived_set = args.derived_set
    candidate_file = candidates_path(args.framework, args.example)
    if not candidate_file.exists():
        sys.exit(f"ERROR: {candidate_file} not found")

    candidates = json.loads(candidate_file.read_text())
    if args.function:
        candidates = [c for c in candidates if c["function_name"] == args.function]
        if not candidates:
            sys.exit(
                f"ERROR: function {args.function!r} not found in {candidate_file}"
            )
    elif not candidates:
        print(f"init: no candidates in {candidate_file}; skipping")
        return

    out_dir = trace_function_dir(
        args.framework,
        args.example,
        "_placeholder",
        derived_set,
    ).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    successes, failures = [], []
    for c in candidates:
        func = c["function_name"]
        print(f"\n=== init[{derived_set}]: {func} ===")
        try:
            _init_one(
                args.framework,
                args.example,
                c,
                out_dir,
                derived_set,
            )
            successes.append(func)
        except Exception as exc:
            print(f"init: {func} FAILED — {exc}")
            failures.append((func, str(exc)))

    print(f"\ninit: {len(successes)} succeeded, {len(failures)} failed")
    for func, msg in failures:
        print(f"  - {func}: {msg}")
    if not successes:
        sys.exit(1)


def _init_one(framework: str, example: str, candidate: dict,
              out_dir: Path, derived_set: str) -> None:
    func = candidate["function_name"]
    impl_loc = _normalize_impl_location(candidate["implementation_location"])

    req_md_path = requirement_path(framework, example, func)
    cases_path = test_input_path(framework, example, func)
    for p in (req_md_path, cases_path):
        if not p.exists():
            raise RuntimeError(f"missing: {p}")

    rel, start, _ = _parse_impl_location(impl_loc)
    if rel is None:
        raise RuntimeError(
            f"cannot parse implementation_location: {impl_loc!r}"
        )
    code_dir = PROJECT_ROOT / framework / "test_examples" / example / "code"
    src_path = code_dir / rel
    if not src_path.exists():
        raise RuntimeError(f"source file not found: {src_path}")
    callable_name = _callable_name_for_location(src_path, func, start)

    requirement_md = req_md_path.read_text()

    func_dir = out_dir / func
    func_dir.mkdir(parents=True, exist_ok=True)

    print(f"init: {func} — running oracle in Docker ...")
    oracle_path = func_dir / "oracle.json"
    oracle = _run_differential(
        code_dir=code_dir,
        rel_file=rel,
        function_name=callable_name,
        test_cases_path=cases_path,
        completion=None,
        output_path=oracle_path,
        image=get_docker_image(framework),
    )
    _validate_oracle(oracle, func)

    spec = {
        "function_name": func,
        "callable_name": callable_name,
        "implementation_location": impl_loc,
        "framework": framework,
        "example": example,
        "derived_set": derived_set,
        "source_record": {
            "function_name": func,
            "prompt": [
                {"role": "system", "content": _GLOBAL_SYSTEM},
                {"role": "user",   "content": requirement_md},
            ],
        },
    }
    spec_path = func_dir / "pilot_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"init: {func} — wrote {func_dir.relative_to(TRACES_DIR.parent)}")


def _normalize_impl_location(loc: str) -> str:
    """derived uses 'code/foo.py:21-24'; _parse_impl_location wants 'code/foo.py:line 21-24'."""
    head, sep, tail = loc.rpartition(":")
    if not sep or tail.startswith("line "):
        return loc
    return f"{head}:line {tail}"


def _validate_oracle(oracle: list, function_name: str) -> None:
    if not isinstance(oracle, list) or not oracle:
        raise RuntimeError("oracle is empty or not a list")

    counts = {"normal": 0, "edge": 0, "error": 0}
    ok_count = {"normal": 0, "edge": 0, "error": 0}
    for rec in oracle:
        cat = rec.get("category", "normal")
        counts[cat] = counts.get(cat, 0) + 1
        if rec.get("status") == "ok":
            ok_count[cat] = ok_count.get(cat, 0) + 1

    print(f"init: {function_name} — oracle counts — "
          f"normal={counts['normal']} (ok={ok_count['normal']}), "
          f"edge={counts['edge']} (ok={ok_count['edge']}), "
          f"error={counts['error']} (ok={ok_count['error']})")

    if counts["normal"] < 5:
        raise RuntimeError(f"not enough normal cases ({counts['normal']} < 5)")

    n_strong = counts["normal"] + counts["edge"]
    ok_strong = ok_count["normal"] + ok_count["edge"]
    if n_strong > 0 and (ok_strong / n_strong) < 0.8:
        raise RuntimeError(
            f"normal+edge ok-ratio {ok_strong}/{n_strong} < 80% — "
            f"test_input.py has cases that error under GT"
        )


# --------------------------------------------------------------------------- #
# grade-smoke                                                                 #
# --------------------------------------------------------------------------- #

def cmd_grade_smoke(args):
    spec = _load_spec(args.framework, args.example, args.function, args.derived_set)
    oracle_path = oracle_path_for_spec(spec)
    n_cases = len(json.loads(oracle_path.read_text()))

    rel, _, _ = _parse_impl_location(spec["implementation_location"])
    gt_file = (PROJECT_ROOT / spec["framework"] / "test_examples"
               / spec["example"] / "code" / rel)
    gt_body = _extract_function_from_file(
        str(gt_file),
        _callable_display_name(spec.get("callable_name") or spec["function_name"]),
    )

    print(f"grade-smoke: spec → {spec['framework']}/{spec['example']}/"
          f"{spec['function_name']} ({n_cases} cases)")

    print("grade-smoke: [1/2] grading GT (must pass) ...")
    result = grade_completion(spec, gt_body)
    assert result["passed"] and not result.get("infra_failure"), \
        f"GT grading failed: {result}"
    print(f"grade-smoke: [1/2] GT passed — {n_cases}/{n_cases} cases")

    print("grade-smoke: [2/2] grading empty completion (must fail) ...")
    result2 = grade_completion(spec, "")
    assert not result2["passed"], f"Empty unexpectedly passed: {result2}"
    assert not result2.get("infra_failure"), f"Empty stub triggered infra failure: {result2}"
    empty_passes = sum(
        1 for case in result2.get("per_case", [])
        if case.get("status") == "pass"
    )
    print(
        "grade-smoke: [2/2] empty failed as expected — "
        f"{empty_passes}/{n_cases} cases matched"
    )

    print("grade-smoke: all gates passed ✓")


# --------------------------------------------------------------------------- #
# practice / structured memory                                                #
# --------------------------------------------------------------------------- #

def cmd_practice(args):
    spec = _load_spec(args.framework, args.example, args.function, args.derived_set)
    print(f"practice: spec → {spec['framework']}/{spec['example']}/"
          f"{spec['function_name']} [{spec['derived_set']}]")
    model_desc = f"agent=openhands provider={args.provider} model={args.model}"
    if args.base_model:
        model_desc += f" base_model={args.base_model}"
    print(f"practice: up to K={args.K} iterations with {model_desc} "
          f"(max_agent_iterations={args.max_iterations}, early-stop on first PASS) ...")
    trace = run_practice(
        spec, K=args.K, model=args.model,
        api_key=args.api_key, base_url=args.base_url,
        provider=args.provider,
        base_model=args.base_model,
        api_version=args.api_version,
        max_agent_iterations=args.max_iterations,
    )
    ran = trace["iterations_ran"]
    passed = sum(1 for it in trace["iterations"]
                 if it.get("grade", {}).get("passed"))
    trace_path = trace_path_for_spec(spec)
    early = " (stopped early)" if trace["stopped_early"] else ""
    print(f"practice: ran {ran}/{args.K} iterations{early} — "
          f"{trace['non_empty_count']}/{ran} non-empty, {passed}/{ran} passed")
    print(f"practice: wrote trace → {trace_path}")
    usable = [
        it for it in trace["iterations"]
        if it.get("completion") and not it.get("grade", {}).get("infra_failure")
    ]
    if not usable:
        raise RuntimeError(
            f"practice produced no usable attempt for {spec['function_name']}"
        )


def cmd_distill_structured(args):
    spec = _load_spec(args.framework, args.example, args.function, args.derived_set)
    trace_path = trace_path_for_spec(spec)
    trace = json.loads(trace_path.read_text())
    print(f"distill-structured: reading trace → {trace_path} "
          f"({len(trace.get('iterations', []))} iterations)")
    print(f"distill-structured: calling provider={args.provider} model={args.model} ...")
    entries = distill_structured_trace(
        trace,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        provider=args.provider,
        api_version=args.api_version,
    )
    if not entries:
        raise RuntimeError(
            f"distill-structured produced no knowledge for {spec['function_name']}"
        )


def cmd_consolidate_structured(args):
    print(f"consolidate-structured: profile={args.derived_set}, example={args.example} ...")
    consolidate_structured_example(
        args.example,
        derived_set=args.derived_set,
        framework=args.framework,
    )


def _load_spec(framework: str, example: str, function: str,
               derived_set: str = DEFAULT_DERIVED_SET) -> dict:
    p = trace_function_dir(framework, example, function, derived_set) / "pilot_spec.json"
    if not p.exists():
        sys.exit(f"ERROR: {p} not found; run init first")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# entrypoint                                                                  #
# --------------------------------------------------------------------------- #

def _env_file_values() -> dict[str, str]:
    values = {}
    for env_file in (SCRIPTS_DIR / ".env", SCRIPTS_DIR / ".env.local"):
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values[name.strip()] = value.split("#", 1)[0].strip()
    return values


def _infer_provider(base_url: str) -> str:
    if "openrouter.ai" in base_url:
        return "openrouter"
    if "openai.azure.com" in base_url or "services.ai.azure.com" in base_url:
        return "azure_openai"
    if (
        "api.openai.com" in base_url
        or "token-plan.cn-beijing.maas.aliyuncs.com" in base_url
    ):
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
    # Preserve the historical default for unknown/custom endpoints.
    return "OPENROUTER_API_KEY"


def _load_api_key(*, api_key: str, api_key_env: str | None,
                  provider: str, base_url: str) -> str:
    """Load a provider key from args, environment, or scripts/.env."""
    if api_key:
        return api_key

    env_name = api_key_env or _default_api_key_env(provider, base_url)
    placeholders = {
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here",
        "OPENAI_API_KEY": "sk-your-openai-key-here",
        "AZURE_OPENAI_API_KEY": "your-azure-openai-key-here",
        "BAILIAN_TOKEN_PLAN_API_KEY": "sk-sp-your-token-plan-key-here",
        "QWEN_TOKEN_PLAN_API_KEY": "sk-sp-your-token-plan-key-here",
    }

    env_values = _env_file_values()
    env_names = (env_name,)
    if env_name == "BAILIAN_TOKEN_PLAN_API_KEY":
        env_names = ("BAILIAN_TOKEN_PLAN_API_KEY", "QWEN_TOKEN_PLAN_API_KEY")
    for name in env_names:
        key = os.environ.get(name, "").strip()
        if not key:
            key = env_values.get(name, "").strip()
        if key and key != placeholders.get(name):
            return key

    env_file = SCRIPTS_DIR / ".env.local"
    sys.exit(
        f"ERROR: {env_name} is not set. Set it in the environment, pass "
        f"--api-key, or add it to {env_file}."
    )


def _add_derived_set_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--derived-set",
        "--profile",
        dest="derived_set",
        default=DEFAULT_DERIVED_SET,
        help=(
            "procedural-memory profile. Input always comes from observed "
            "memory; the profile only scopes traces and practice_knowledge "
            "outputs."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Memory pipeline pilot")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help=(
            "LLM provider. Defaults from --base-url; pass this explicitly for "
            "Azure/OpenAI/OpenRouter runs."
        ),
    )
    parser.add_argument(
        "--base-model",
        default="",
        help=(
            "underlying model/version for trace metadata only, e.g. "
            "gpt-5.5-2026-04-24 when --model is an Azure deployment name"
        ),
    )
    parser.add_argument(
        "--api-version",
        default="",
        help="Azure OpenAI legacy API version. Leave empty for Azure OpenAI v1 endpoints.",
    )
    parser.add_argument("--api-key", default="", help="LLM API key override")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "environment variable to read for the LLM API key. Defaults to "
            "OPENROUTER_API_KEY for OpenRouter, OPENAI_API_KEY for OpenAI, "
            "AZURE_OPENAI_API_KEY for Azure OpenAI, and "
            "BAILIAN_TOKEN_PLAN_API_KEY for token-plan OpenAI-compatible "
            "endpoints."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser(
        "init", help="build per-function pilot_spec + oracle from observed memory"
    )
    p_init.add_argument("--framework", required=True)
    p_init.add_argument("--example", required=True)
    _add_derived_set_arg(p_init)
    p_init.add_argument("--function", default=None,
                        help="only init this function (default: all in candidates.json)")
    p_init.set_defaults(func=cmd_init)

    p_gs = sub.add_parser("grade-smoke")
    p_gs.add_argument("--framework", required=True)
    p_gs.add_argument("--example", required=True)
    p_gs.add_argument("--function", required=True)
    _add_derived_set_arg(p_gs)
    p_gs.set_defaults(func=cmd_grade_smoke)

    p_pr = sub.add_parser("practice")
    p_pr.add_argument("--framework", required=True)
    p_pr.add_argument("--example", required=True)
    p_pr.add_argument("--function", required=True)
    _add_derived_set_arg(p_pr)
    p_pr.add_argument("--K", type=int, default=8,
                      help="max practice iterations; stops early on first PASS (default: 8)")
    p_pr.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.environ.get("KOCO_PRACTICE_MAX_ITERATIONS", "100")),
        help=(
            "max OpenHands agent turns per practice attempt. "
            "Defaults to KOCO_PRACTICE_MAX_ITERATIONS or 100."
        ),
    )
    p_pr.add_argument("--model", required=True)
    p_pr.set_defaults(func=cmd_practice)

    p_ds = sub.add_parser("distill-structured",
                          help="distill one trace into structured PracticeMemory JSONL")
    p_ds.add_argument("--framework", required=True)
    p_ds.add_argument("--example", required=True)
    p_ds.add_argument("--function", required=True)
    _add_derived_set_arg(p_ds)
    p_ds.add_argument("--model", required=True)
    p_ds.set_defaults(func=cmd_distill_structured)

    p_cs = sub.add_parser("consolidate-structured",
                          help="merge one profile's structured per-function notes into JSONL")
    p_cs.add_argument("--example", required=True)
    p_cs.add_argument("--framework", required=True)
    _add_derived_set_arg(p_cs)
    p_cs.set_defaults(func=cmd_consolidate_structured)

    args = parser.parse_args()
    args.provider = args.provider or _infer_provider(args.base_url)
    if args.command in {"practice", "distill-structured"}:
        args.api_key = _load_api_key(
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            provider=args.provider,
            base_url=args.base_url,
        )
    args.func(args)


if __name__ == "__main__":
    main()
