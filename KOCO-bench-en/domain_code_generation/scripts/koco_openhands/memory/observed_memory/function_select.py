"""Stage 2: Select candidate practice functions.

Supports multiple selection strategies via the ``strategy`` parameter:

- "llm"            — original LLM agent selection (default)
- "business_logic"  — LLM agent focused on domain business logic
- "indegree"        — programmatic OR LLM: functions most called by other project code
- "outdegree"       — programmatic OR LLM: functions that call the most project-internal code
"""

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from agent.sdk import run_sdk_agent
from memory.config import code_dir, candidates_path
from memory.observed_memory.workspace import build_explore_workspace
from runner import _collect_gt_locations, load_jsonl

# Map strategy name → prompt template name (LLM-only strategies)
_LLM_STRATEGIES = {
    "llm": "select",
    "business_logic": "select_business_logic",
}

# Programmatic strategies (no LLM, pure script)
_PROGRAMMATIC_STRATEGIES = {"indegree", "outdegree"}


def _run_llm_select(
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str,
    max_iterations: int,
    strategy: str,
) -> list[dict]:
    """Run an LLM agent to select candidate functions.

    Returns selected candidate entries.
    """
    from memory.config import ensure_input_data, SCRIPTS_DIR

    data_file = SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"
    if not data_file.exists():
        if not ensure_input_data(framework, example):
            raise RuntimeError(f"Failed to generate input data for {example}")

    records = load_jsonl(str(data_file))
    gt_locations = _collect_gt_locations(records)

    code_root = str(code_dir(framework, example))
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = build_explore_workspace(code_root, gt_locations, tmp_dir)

        prompt_name = _LLM_STRATEGIES[strategy]
        prompt = (
            Path(__file__).resolve().parent
            / "prompts"
            / f"{prompt_name}.md"
        ).read_text(encoding="utf-8")
        events, status = run_sdk_agent(
            prompt=prompt,
            workspace=paths["workspace"],
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_iterations=max_iterations,
            corpus_dirs=[paths["code"]],
        )

        # Extract candidate_functions.json from workspace
        candidates_file = os.path.join(paths["workspace"], "candidate_functions.json")
        if os.path.exists(candidates_file):
            raw = Path(candidates_file).read_text(encoding="utf-8")
        else:
            # Fallback: try to extract JSON from events
            raw = None
            for event in reversed(events):
                content = getattr(event, "content", None) or ""
                if "function_name" in content and "implementation_location" in content:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start >= 0 and end > start:
                        raw = content[start:end]
                        break
            if raw is None:
                raise RuntimeError(
                    f"Agent did not produce candidate_functions.json "
                    f"(status={status})"
                )

    candidates = json.loads(raw)
    if not isinstance(candidates, list):
        raise RuntimeError("Invalid candidates format: expected list")
    return candidates


def _run_programmatic_select(
    framework: str,
    example: str,
    strategy: str,
    top_k: int | None = None,
) -> list[dict]:
    """Run a programmatic call-graph strategy (no LLM needed)."""
    from memory.config import ensure_input_data, SCRIPTS_DIR
    from memory.observed_memory.select_callgraph import select_by_strategy

    data_file = SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"
    if not data_file.exists():
        if not ensure_input_data(framework, example):
            raise RuntimeError(f"Failed to generate input data for {example}")

    code_root = str(code_dir(framework, example))
    return select_by_strategy(
        strategy, str(data_file), code_root,
        top_k=top_k, framework=framework, example=example,
    )


def _filter_by_scope(
    candidates: list[dict], code_root: str, framework: str, example: str
) -> list[dict]:
    """Remove candidates whose implementation_location falls outside instance scope dirs."""
    from memory.observed_memory.select_callgraph import determine_scope_dirs
    from runner import _parse_impl_location

    scope_dirs = determine_scope_dirs(code_root, example, framework)
    if scope_dirs is None:
        # No scope determined — keep all candidates
        return candidates

    filtered = []
    skipped = []
    for c in candidates:
        loc = c.get("implementation_location", "")
        rel_path, start, end = _parse_impl_location(loc)
        if not rel_path:
            filtered.append(c)
            continue
        rel_norm = rel_path.replace("\\", "/")
        in_scope = any(rel_norm.startswith(d + "/") or rel_norm == d for d in scope_dirs)
        if in_scope:
            filtered.append(c)
        else:
            skipped.append(c["function_name"])

    if skipped:
        print(f"    Scope filter: removed {len(skipped)} framework functions: {skipped[:5]}"
              + ("..." if len(skipped) > 5 else ""))
    # If scope filter removed everything, fall back to keeping all candidates.
    # This happens when all GT functions live in the framework directory (e.g. DAPO).
    if not filtered and candidates:
        print(f"    Scope filter: all candidates removed — falling back to keep all {len(candidates)}")
        return candidates
    return filtered


def _filter_abstract_stubs(candidates: list[dict], code_root: str) -> list[dict]:
    """Remove abstract stubs (raise NotImplementedError / pass-only) from candidates."""
    import ast
    from runner import _parse_impl_location

    filtered = []
    for c in candidates:
        loc = c.get("implementation_location", "")
        rel_path, start, end = _parse_impl_location(loc)
        if not rel_path or not start or not end:
            filtered.append(c)
            continue

        full_path = os.path.join(code_root, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=full_path)
        except Exception:
            filtered.append(c)
            continue

        is_stub = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= start <= node.end_lineno:
                    body = node.body
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        body = body[1:]
                    if not body:
                        is_stub = True
                    elif (len(body) == 1 and isinstance(body[0], ast.Raise)
                          and isinstance(body[0].exc, ast.Name)
                          and body[0].exc.id == "NotImplementedError"):
                        is_stub = True
                    elif (len(body) == 1 and isinstance(body[0], ast.Raise)
                          and isinstance(body[0].exc, ast.Call)
                          and isinstance(body[0].exc.func, ast.Name)
                          and body[0].exc.func.id == "NotImplementedError"):
                        is_stub = True
                    elif len(body) == 1 and isinstance(body[0], ast.Pass):
                        is_stub = True
                    break

        if is_stub:
            print(f"    SKIP {c['function_name']}: abstract stub (raise NotImplementedError / pass)")
        else:
            filtered.append(c)

    skipped = len(candidates) - len(filtered)
    if skipped:
        print(f"    Filtered {skipped} abstract stub(s) from {len(candidates)} candidates")
    return filtered


def _leaf_name(name: str) -> str:
    """Return the final component of a possibly qualified function name."""

    return str(name or "").strip().split(".")[-1]


def _is_qualified_name(name: str) -> bool:
    return "." in str(name or "").strip()


def _normalized_location_key(location: str) -> str:
    """Normalize an implementation location for overlap matching."""

    from runner import _parse_impl_location

    rel_path, start, end = _parse_impl_location(str(location or ""))
    if not rel_path or not start or not end:
        return ""
    rel_path = rel_path.replace("\\", "/").lstrip("./")
    if rel_path.startswith("code/"):
        rel_path = rel_path[len("code/"):]
    return f"{rel_path}:{start}-{end}"


def _load_benchmark_targets(framework: str, example: str) -> list[dict]:
    """Load benchmark target records for the current example."""

    from memory.config import ensure_input_data, SCRIPTS_DIR

    data_file = SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"
    if not data_file.exists():
        if not ensure_input_data(framework, example):
            raise RuntimeError(f"Failed to generate input data for {example}")
    return load_jsonl(str(data_file))


def _filter_benchmark_targets(
    candidates: list[dict],
    framework: str,
    example: str,
) -> list[dict]:
    """Remove candidates that overlap the benchmark test targets.

    Practice knowledge must not include the same functions that are later used
    as benchmark targets.  Match locations first, then full names, then unique
    leaf names for cases where one side omits a class qualifier.
    """

    targets = _load_benchmark_targets(framework, example)
    target_names = {
        str(record.get("function_name", "")).strip()
        for record in targets
        if record.get("function_name")
    }
    leaf_counts = Counter(_leaf_name(name) for name in target_names)
    unique_target_leaf_names = {
        leaf for leaf, count in leaf_counts.items() if leaf and count == 1
    }
    target_qualified_by_leaf = {
        _leaf_name(name): _is_qualified_name(name)
        for name in target_names
    }
    target_locations = {
        key
        for key in (
            _normalized_location_key(record.get("implementation_location", ""))
            for record in targets
        )
        if key
    }

    filtered = []
    removed = []
    for candidate in candidates:
        name = str(candidate.get("function_name", "")).strip()
        loc_key = _normalized_location_key(candidate.get("implementation_location", ""))
        leaf = _leaf_name(name)
        leaf_overlap = (
            leaf in unique_target_leaf_names
            and (not _is_qualified_name(name) or not target_qualified_by_leaf.get(leaf, False))
        )
        overlaps = (
            bool(loc_key and loc_key in target_locations)
            or name in target_names
            or leaf_overlap
        )
        if overlaps:
            removed.append(name or candidate.get("implementation_location", "<unknown>"))
        else:
            filtered.append(candidate)

    if removed:
        print(
            f"    Benchmark target filter: removed {len(removed)} candidate(s): "
            f"{removed[:5]}" + ("..." if len(removed) > 5 else "")
        )
    return filtered


def _save_candidates(candidates: list[dict], framework: str, example: str) -> Path:
    """Validate and save candidates to the standard output path."""
    if len(candidates) == 0:
        raise RuntimeError("Selection produced 0 candidates")

    for c in candidates:
        if "function_name" not in c or "implementation_location" not in c:
            raise RuntimeError(f"Invalid candidate entry: {c}")

    out = candidates_path(framework, example)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _deduplicate(candidates: list[dict]) -> list[dict]:
    """Deduplicate candidates by implementation_location (file:line-range).

    Two candidates with the same file+line range are the same function,
    even if their function_name differs (e.g. 'load_documents' vs
    'MCPGeminiRAGClient.load_documents').
    """
    seen_names = set()
    seen_locs = set()
    result = []
    for c in candidates:
        name = c.get("function_name", "")
        loc = c.get("implementation_location", "")
        if not name:
            continue
        # Dedup by exact name
        if name in seen_names:
            continue
        # Dedup by location: same file+line = same function
        loc_key = loc
        if loc_key and loc_key in seen_locs:
            continue
        seen_names.add(name)
        if loc_key:
            seen_locs.add(loc_key)
        result.append(c)
    return result


def _parse_strategies(strategy: str) -> list[str]:
    """Parse strategy string into a list of strategy names.

    Accepts: "llm", "indegree", "all", "llm,indegree", etc.
    """
    valid = set(_LLM_STRATEGIES) | _PROGRAMMATIC_STRATEGIES
    if strategy == "all":
        return list(_LLM_STRATEGIES) + list(_PROGRAMMATIC_STRATEGIES)
    strategies = [s.strip() for s in strategy.split(",") if s.strip()]
    for s in strategies:
        if s not in valid:
            raise ValueError(f"Unknown strategy: {s}")
    return strategies


def run_select(
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 100,
    strategy: str = "llm",
    top_k: int | None = None,
) -> Path:
    """Stage 2: Select candidate functions for practice.

    Args:
        strategy: "llm", "business_logic" (LLM), "indegree", "outdegree" (script),
                  "all" (all 4), or comma-separated (e.g. "llm,indegree").
        top_k: Max candidates for programmatic strategies.

    Returns path to written observed-memory candidates.json.
    """
    strategies = _parse_strategies(strategy)
    all_candidates: list[dict] = []

    for i, strat in enumerate(strategies):
        if strat in _LLM_STRATEGIES:
            print(f"  Running strategy [{i+1}/{len(strategies)}]: {strat} (LLM)")
            candidates = _run_llm_select(
                framework, example, model, api_key, base_url, max_iterations, strat,
            )
        elif strat in _PROGRAMMATIC_STRATEGIES:
            print(f"  Running strategy [{i+1}/{len(strategies)}]: {strat} (script)")
            candidates = _run_programmatic_select(framework, example, strat, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strat}")

        print(f"    -> {len(candidates)} candidates from {strat}")
        all_candidates.extend(candidates)

    # Deduplicate across strategies
    before_dedup = len(all_candidates)
    all_candidates = _deduplicate(all_candidates)
    if before_dedup > len(all_candidates):
        print(f"  Deduplicated: {before_dedup} -> {len(all_candidates)} candidates")

    # Filter by instance scope (remove framework-internal candidates)
    code_root = str(code_dir(framework, example))
    all_candidates = _filter_by_scope(all_candidates, code_root, framework, example)

    # Filter out benchmark targets to avoid practice/test leakage.
    all_candidates = _filter_benchmark_targets(all_candidates, framework, example)

    # Filter abstract stubs before saving
    all_candidates = _filter_abstract_stubs(all_candidates, code_root)
    if not all_candidates:
        raise RuntimeError("All candidates were abstract stubs — nothing to practice")

    out = _save_candidates(all_candidates, framework, example)
    print(f"  Stage 2 done (strategies={strategies}): {len(all_candidates)} candidates -> {out}")
    return out
