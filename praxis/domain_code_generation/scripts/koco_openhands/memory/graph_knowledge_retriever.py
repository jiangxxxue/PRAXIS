"""Retrieval helpers for dependency-graph-mounted practice knowledge."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from memory.confidence import (
    configured_confidence_threshold,
    knowledge_confidence_score,
)


class GraphKnowledgeRetriever:
    """Read ``dep_graph.with_knowledge.json`` and retrieve exact node knowledge."""

    def __init__(
        self,
        graph_knowledge_path: str | Path,
        min_confidence: float | None = None,
    ):
        self.path = Path(graph_knowledge_path)
        self.min_confidence = configured_confidence_threshold(min_confidence)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = self.data.get("nodes", [])
        self.edges = self.data.get("edges", [])
        self.knowledge_items = self.data.get("knowledge_items", [])
        self.canonical_knowledge_items = self.data.get("canonical_knowledge_items", [])
        self.propagated_knowledge_items = self.data.get("propagated_knowledge_items", [])

        self.nodes_by_key = {node["node_key"]: node for node in self.nodes if node.get("node_key")}
        self.knowledge_by_id = {
            item["id"]: item for item in self.knowledge_items if item.get("id")
        }
        self.knowledge_by_id.update({
            item["id"]: item
            for item in self.propagated_knowledge_items
            if item.get("id")
        })
        self.canonical_by_id = {
            item["id"]: item
            for item in self.canonical_knowledge_items
            if item.get("id")
        }
        self.node_keys_by_symbol: dict[str, list[str]] = defaultdict(list)
        self.nodes_by_file_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.callers_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.callees_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

        self._build_symbol_index()
        self._build_file_range_index()
        self._build_call_indexes()

    def _build_symbol_index(self) -> None:
        for node in self.nodes:
            node_key = node.get("node_key")
            if not node_key:
                continue
            candidates = {
                str(node.get("id", "")),
                str(node.get("name", "")),
                str(node.get("qualified_name", "")),
                str(node_key),
            }
            parent_class = node.get("parent_class")
            name = node.get("name")
            if parent_class and name:
                candidates.add(f"{parent_class}.{name}")
            qualified_name = str(node.get("qualified_name", ""))
            if qualified_name:
                parts = qualified_name.split(".")
                if len(parts) >= 2:
                    candidates.add(".".join(parts[-2:]))
                if len(parts) >= 1:
                    candidates.add(parts[-1])

            for symbol in candidates:
                if symbol:
                    self.node_keys_by_symbol[symbol].append(node_key)

    @staticmethod
    def _normalize_file_path(path: str | Path | None) -> str:
        if path is None:
            return ""
        value = str(path).replace("\\", "/").strip()
        while value.startswith("./"):
            value = value[2:]
        for marker in ("/code/", "/knowledge_corpus/"):
            if marker in value:
                value = value.split(marker, 1)[1]
        for prefix in ("code/", "knowledge_corpus/"):
            if value.startswith(prefix):
                value = value[len(prefix):]
        return value.lstrip("/")

    def _build_file_range_index(self) -> None:
        for node in self.nodes:
            file_path = self._normalize_file_path(node.get("file_path"))
            if not file_path or node.get("lineno") is None or node.get("end_lineno") is None:
                continue
            self.nodes_by_file_path[file_path].append(node)
        for nodes in self.nodes_by_file_path.values():
            nodes.sort(key=lambda item: (int(item.get("lineno") or 0), int(item.get("end_lineno") or 0)))

    def _build_call_indexes(self) -> None:
        for edge in self.edges:
            if edge.get("kind") != "call":
                continue
            source_keys = self._edge_endpoint_keys(edge, "source")
            target_keys = self._edge_endpoint_keys(edge, "target")
            for target_key in target_keys:
                for source_key in source_keys:
                    self.callers_by_target[target_key].append({
                        "node_key": source_key,
                        "edge": edge,
                    })
            for source_key in source_keys:
                for target_key in target_keys:
                    self.callees_by_source[source_key].append({
                        "node_key": target_key,
                        "edge": edge,
                    })

    @staticmethod
    def _edge_endpoint_keys(edge: dict[str, Any], endpoint: str) -> list[str]:
        exact = edge.get(f"{endpoint}_node_key")
        if exact:
            return [exact]
        ambiguous = edge.get(f"{endpoint}_ambiguous_node_keys") or []
        return [key for key in ambiguous if key]

    def resolve_symbols(self, symbols: list[str] | str) -> list[dict[str, Any]]:
        """Resolve exact function/method symbols to graph nodes."""

        if isinstance(symbols, str):
            symbols = [symbols]

        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for symbol in symbols:
            for node_key in self._resolve_one_symbol(symbol):
                if node_key in seen:
                    continue
                node = self.nodes_by_key.get(node_key)
                if node:
                    resolved.append(node)
                    seen.add(node_key)
        return resolved

    def _resolve_one_symbol(self, symbol: str) -> list[str]:
        symbol = symbol.strip()
        if not symbol:
            return []
        if symbol in self.nodes_by_key:
            return [symbol]
        if symbol in self.node_keys_by_symbol:
            return self.node_keys_by_symbol[symbol]

        # Exact suffix fallback for records that provide Class.method or bare function names.
        suffix = f".{symbol}"
        matches = [
            key for key in self.nodes_by_key
            if key.endswith(suffix) or key.split(".")[-1] == symbol
        ]
        return matches

    def resolve_by_code_location(
        self,
        path: str | Path,
        start_line: int,
        end_line: int | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve code location to graph nodes whose function ranges overlap it."""

        normalized_path = self._normalize_file_path(path)
        if not normalized_path or start_line <= 0:
            return []
        if end_line is None or end_line < start_line:
            end_line = start_line

        candidates = self.nodes_by_file_path.get(normalized_path, [])
        if not candidates:
            candidates = [
                node
                for file_path, nodes in self.nodes_by_file_path.items()
                if file_path.endswith(f"/{normalized_path}") or normalized_path.endswith(f"/{file_path}")
                for node in nodes
            ]

        matches = []
        for node in candidates:
            node_start = int(node.get("lineno") or 0)
            node_end = int(node.get("end_lineno") or node_start)
            overlap_start = max(start_line, node_start)
            overlap_end = min(end_line, node_end)
            if overlap_start <= overlap_end:
                matches.append((overlap_end - overlap_start + 1, node_start, node))

        matches.sort(key=lambda item: (-item[0], item[1]))
        return [node for _overlap, _node_start, node in matches]

    def search_code_locations(
        self,
        locations: list[dict[str, Any]],
        max_nodes: int = 3,
        max_items_per_node: int = 5,
    ) -> dict[str, Any]:
        """Return node knowledge for code locations returned by other tools."""

        matched_nodes: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        for location in locations:
            if len(seen_nodes) >= max_nodes:
                break
            path = location.get("path")
            start_line = int(location.get("start_line") or 0)
            end_line = int(location.get("end_line") or start_line)
            for node in self.resolve_by_code_location(path, start_line, end_line):
                node_key = node.get("node_key")
                if not node_key or node_key in seen_nodes:
                    continue
                # Skip nodes without any practice knowledge so they don't
                # consume max_nodes budget. This matters when a coarse-grained
                # node (e.g. an outer class) overlaps the location more than a
                # fine-grained method that actually carries knowledge.
                node_items = self.get_node_knowledge(node_key)
                if not node_items:
                    continue
                seen_nodes.add(node_key)
                matched_nodes.append({
                    "node_key": node_key,
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "qualified_name": node.get("qualified_name"),
                    "file_path": node.get("file_path"),
                    "lineno": node.get("lineno"),
                    "end_lineno": node.get("end_lineno"),
                    "matched_location": {
                        "path": path,
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                })
                for item in node_items[:max_items_per_node]:
                    results.append({
                        "query_node_key": node_key,
                        "query_symbol": node.get("id") or node.get("name"),
                        "relation": "matched_code",
                        "node_key": node_key,
                        "node_name": node.get("id") or node.get("name"),
                        "qualified_name": node.get("qualified_name"),
                        "edge": None,
                        "knowledge": item,
                    })
                if len(seen_nodes) >= max_nodes:
                    break

        results.sort(
            key=lambda result: (
                -knowledge_confidence_score(result["knowledge"]),
                str(result["knowledge"].get("id") or ""),
            )
        )
        return {
            "matched_nodes": matched_nodes,
            "results": results,
        }

    def get_node_knowledge(self, node_key: str) -> list[dict[str, Any]]:
        """Return confidence-filtered knowledge sorted from highest to lowest."""

        node = self.nodes_by_key.get(node_key)
        if not node:
            return []

        knowledge = node.get("knowledge", {})
        canonical_ids = knowledge.get("canonical", [])
        if canonical_ids:
            canonical_items = [
                item
                for item in canonical_ids
                if isinstance(item, dict) and item.get("status", "active") == "active"
            ]
            removed_items = [
                item
                for item in knowledge.get("removed", [])
                if isinstance(item, dict)
            ]
            covered_source_ids = {
                source_id
                for item in canonical_items + removed_items
                for source_id in item.get("source_ids", [])
                if source_id
            }
            direct_additions = [
                item
                for item in knowledge.get("direct", [])
                if isinstance(item, dict) and item.get("id") not in covered_source_ids
            ]
            items = canonical_items + direct_additions
            return self._filter_and_rank_knowledge(items)

        direct_items = knowledge.get("direct", [])
        return self._filter_and_rank_knowledge([
            item for item in direct_items if isinstance(item, dict)
        ])

    def _filter_and_rank_knowledge(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        filtered = [
            item
            for item in items
            if knowledge_confidence_score(item) >= self.min_confidence
        ]
        return sorted(
            filtered,
            key=lambda item: (
                -knowledge_confidence_score(item),
                str(item.get("id") or ""),
            ),
        )

    def search_exact(
        self,
        symbols: list[str] | str,
        mode: str = "node",
        max_items: int | None = None,
    ) -> dict[str, Any]:
        """Search exact node/caller/callee knowledge for one or more symbols."""

        nodes = self.resolve_symbols(symbols)
        results: list[dict[str, Any]] = []
        visited: set[tuple[str, str, str]] = set()

        for node in nodes:
            node_key = node["node_key"]
            related = self._related_nodes(node_key, mode)
            for relation, related_node, edge in related:
                for item in self.get_node_knowledge(related_node["node_key"]):
                    dedup_key = (node_key, related_node["node_key"], item["id"])
                    if dedup_key in visited:
                        continue
                    visited.add(dedup_key)
                    results.append({
                        "query_node_key": node_key,
                        "query_symbol": node.get("id") or node.get("name"),
                        "relation": relation,
                        "node_key": related_node["node_key"],
                        "node_name": related_node.get("id") or related_node.get("name"),
                        "qualified_name": related_node.get("qualified_name"),
                        "edge": edge,
                        "knowledge": item,
                    })

        results.sort(
            key=lambda result: (
                -knowledge_confidence_score(result["knowledge"]),
                str(result["knowledge"].get("id") or ""),
            )
        )
        if max_items is not None:
            results = results[:max_items]

        return {
            "matched_nodes": [
                {
                    "node_key": node["node_key"],
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "qualified_name": node.get("qualified_name"),
                    "file_path": node.get("file_path"),
                    "lineno": node.get("lineno"),
                }
                for node in nodes
            ],
            "results": results,
        }

    def _related_nodes(self, node_key: str, mode: str) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
        node = self.nodes_by_key.get(node_key)
        if not node:
            return []

        if mode == "node":
            return [("self", node, None)]
        if mode == "callers":
            return self._neighbor_nodes(self.callers_by_target.get(node_key, []), "caller")
        if mode == "callees":
            return self._neighbor_nodes(self.callees_by_source.get(node_key, []), "callee")
        if mode == "subgraph":
            return (
                [("self", node, None)]
                + self._neighbor_nodes(self.callers_by_target.get(node_key, []), "caller")
                + self._neighbor_nodes(self.callees_by_source.get(node_key, []), "callee")
            )
        raise ValueError(f"Unsupported graph knowledge retrieval mode: {mode}")

    def _neighbor_nodes(
        self,
        neighbors: list[dict[str, Any]],
        relation: str,
    ) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
        results = []
        seen = set()
        for entry in neighbors:
            node_key = entry.get("node_key")
            if not node_key or node_key in seen:
                continue
            node = self.nodes_by_key.get(node_key)
            if node:
                results.append((relation, node, entry.get("edge")))
                seen.add(node_key)
        return results


def format_knowledge_results(
    results: list[dict[str, Any]],
    title: str,
    max_content_chars: int = 900,
    knowledge_format: str = "trigger_content",
) -> str:
    """Format graph knowledge retrieval results for an LLM prompt/tool output."""

    if not results:
        return ""

    lines = [title]
    for idx, result in enumerate(results, 1):
        item = result["knowledge"]
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        content = (item.get("content") or "").strip()
        if len(content) > max_content_chars:
            content = content[:max_content_chars].rstrip() + "..."
        if knowledge_format == "content_only":
            lines.append(
                f"{idx}. [Knowledge-ID: {item.get('id', '')}] "
                f"[confidence={knowledge_confidence_score(item):.4f}] {content}"
            )
            continue
        lines.append(f"{idx}. [{result['relation']}] {result['node_key']}")
        lines.append(f"   Knowledge-ID: {item.get('id', '')}")
        lines.append(
            f"   Confidence: {knowledge_confidence_score(item):.4f}"
        )
        if evidence.get("eval_passed") is False:
            pass_ratio = evidence.get("pass_ratio")
            pass_ratio_text = f", pass_ratio={pass_ratio}" if pass_ratio is not None else ""
            lines.append(
                f"   Source: FAILED_ATTEMPT{pass_ratio_text}. Use as caution/debug guidance, not as a proven implementation."
            )
        elif evidence.get("eval_passed") is True:
            pass_ratio = evidence.get("pass_ratio")
            pass_ratio_text = f", pass_ratio={pass_ratio}" if pass_ratio is not None else ""
            lines.append(f"   Source: PASSED_ATTEMPT{pass_ratio_text}.")
        lines.append(f"   Trigger: {item.get('trigger', '')}")
        lines.append(f"   Content: {content}")
    return "\n".join(lines)


def build_initial_caller_knowledge_context(
    graph_knowledge_path: str | Path,
    function_name: str,
    knowledge_format: str = "trigger_content",
    min_confidence: float | None = None,
) -> str:
    """Build initial prompt context using only one-hop caller knowledge."""

    path = Path(graph_knowledge_path)
    if not path.exists():
        return ""

    retriever = GraphKnowledgeRetriever(path, min_confidence=min_confidence)
    search = retriever.search_exact(function_name, mode="callers", max_items=None)
    results = search["results"]
    if not results:
        return ""

    header = (
        "GRAPH-RETRIEVED CALLER KNOWLEDGE:\n"
        "The following one-hop caller knowledge was retrieved from the dependency graph. "
        "It may imply constraints on the target function's outputs, exceptions, tensor shapes, or side effects."
    )
    return format_knowledge_results(
        results,
        header,
        knowledge_format=knowledge_format,
    )
