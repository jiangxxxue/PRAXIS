"""Shared helpers for injecting graph-derived node knowledge into tool outputs.

Used by ``knowledge_search``, ``file_editor`` and ``terminal`` wrappers so that
when a tool surfaces a code location, the matching dependency-graph node's
practice knowledge can be appended to the LLM-visible observation.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_NODES = 3
DEFAULT_MAX_ITEMS_PER_NODE = 5
DEFAULT_MAX_CONTENT_CHARS = 700


def make_graph_retriever(graph_knowledge_path: str | None):
    """Lazily build a ``GraphKnowledgeRetriever``; return None on failure/empty."""
    if not graph_knowledge_path:
        return None
    try:
        from memory.graph_knowledge_retriever import GraphKnowledgeRetriever

        return GraphKnowledgeRetriever(graph_knowledge_path)
    except Exception:
        return None


def inject_for_locations(
    retriever,
    locations: list[dict[str, Any]],
    *,
    title: str = "GRAPH KNOWLEDGE FOR MATCHED CODE LOCATIONS:",
    max_nodes: int = DEFAULT_MAX_NODES,
    max_items_per_node: int = DEFAULT_MAX_ITEMS_PER_NODE,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    knowledge_format: str = "trigger_content",
    seen_node_keys: set[str] | None = None,
) -> tuple[str, list[dict]]:
    """Match graph nodes for given code locations and return (formatted_text, structured_results).

    If ``seen_node_keys`` is provided, node_keys already in the set are filtered
    out of the returned results (cross-call dedup) and newly matched node_keys
    are added to it in-place.
    """
    if retriever is None or not locations:
        return "", []

    search = retriever.search_code_locations(
        locations=locations,
        max_nodes=max_nodes,
        max_items_per_node=max_items_per_node,
    )
    graph_results = search.get("results", [])

    if seen_node_keys is not None and graph_results:
        filtered: list[dict] = []
        for item in graph_results:
            node_key = item.get("node_key")
            if node_key and node_key in seen_node_keys:
                continue
            filtered.append(item)
        for item in filtered:
            node_key = item.get("node_key")
            if node_key:
                seen_node_keys.add(node_key)
        graph_results = filtered

    if not graph_results:
        return "", []

    from memory.graph_knowledge_retriever import format_knowledge_results

    formatted = format_knowledge_results(
        graph_results,
        title,
        max_content_chars=max_content_chars,
        knowledge_format=knowledge_format,
    )
    return formatted, graph_results
