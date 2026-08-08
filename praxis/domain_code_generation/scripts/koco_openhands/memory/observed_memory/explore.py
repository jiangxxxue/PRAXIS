"""Generate observed memory and a dependency graph for one benchmark example."""

import os
import tempfile
from pathlib import Path

from agent.sdk import run_sdk_agent
from memory.config import (
    code_dir,
    ensure_input_data,
    observed_knowledge_path,
)
from memory.observed_memory.build_dep_graph import build_dep_graph, save_dep_graph
from memory.observed_memory.workspace import (
    benchmark_target_locations,
    build_explore_workspace,
)

def _extract_from_events(events, filename):
    """Fallback: extract file content from SDK events (file_editor writes)."""
    for event in reversed(events):
        content = getattr(event, "content", None) or ""
        if filename in content and len(content) > 200:
            # Try to find the file content in the event
            return content
    return None


def run_explore(
    framework: str,
    example: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 100,
) -> Path:
    """Generate observed memory and dep graph for one example.

    Returns the run-scoped observed-knowledge output path.
    """
    if not ensure_input_data(framework, example):
        raise RuntimeError(f"Failed to generate input data for {example}")

    target_locations = benchmark_target_locations(framework, example)

    # Benchmark targets stay present as signatures/docstrings, but their bodies
    # are hidden before either the agent or graph builder can observe them.
    code_root = str(code_dir(framework, example))
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = build_explore_workspace(code_root, target_locations, tmp_dir)

        prompt = (
            Path(__file__).resolve().parent
            / "prompts"
            / "explore.md"
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

        # Extract observed_knowledge.md from workspace
        knowledge_file = os.path.join(paths["workspace"], "observed_knowledge.md")
        if os.path.exists(knowledge_file):
            content = Path(knowledge_file).read_text(encoding="utf-8")
        else:
            # Fallback: scan events
            content = _extract_from_events(events, "observed_knowledge")
            if content is None:
                raise RuntimeError(
                    f"Agent did not produce observed_knowledge.md "
                    f"(status={status})"
                )

        graph = build_dep_graph(paths["code"], framework, example)

    # Save to the run-scoped observed-knowledge directory.
    out = observed_knowledge_path(framework, example)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"  Stage 1 done: {out}")

    # Build program dependency graph (pure AST, no LLM)
    graph_path = save_dep_graph(graph, framework, example)
    stats = graph.get("_stats", {})
    print(f"  Dependency graph: {graph_path}")
    print(f"    {stats.get('num_nodes', '?')} nodes, "
          f"{stats.get('num_edges', '?')} edges "
          f"(calls={stats.get('call_edges', '?')}, "
          f"contains={stats.get('containment_edges', '?')}, "
          f"inherits={stats.get('inheritance_edges', '?')})")

    return out
