"""Optimize graph-mounted practice knowledge.

Pipeline:

1. use an LLM to decide whether each practice-knowledge item should propagate
   across bidirectional call-graph edges, continuing for multiple hops when the
   model says the knowledge remains relevant;
2. classify same-node knowledge pairs as duplicate, conflict, or independent,
   then merge active duplicate clusters with an LLM.

The input mounted graph is left untouched. This script writes
``dep_graph.with_knowledge.optimized.json`` next to the mounted graph by
default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.config import dep_graph_path


SCHEMA = "DEP_GRAPH_KNOWLEDGE_OPTIMIZED_V1"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROPAGATION_MAX_HOPS = 4
DEFAULT_PROPAGATION_MAX_TARGETS_PER_KNOWLEDGE = 12
DEFAULT_MAX_PROPAGATION_DECISIONS = 1000
KNOWLEDGE_RELATION_DUPLICATE = "duplicate"
KNOWLEDGE_RELATION_CONFLICT = "conflict"
KNOWLEDGE_RELATION_INDEPENDENT = "independent"
KNOWLEDGE_RELATIONSHIPS = {
    KNOWLEDGE_RELATION_DUPLICATE,
    KNOWLEDGE_RELATION_CONFLICT,
    KNOWLEDGE_RELATION_INDEPENDENT,
}
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _stable_id(prefix: str, *parts: str, length: int = 12) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _request_model_name(model: str, base_url: str) -> str:
    return model


def _http_retry_delay(status_code: int, headers: Any, attempt: int) -> float:
    retry_after = None
    if headers is not None:
        retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(1.0, min(float(retry_after), 300.0))
        except ValueError:
            pass
    if status_code == 429:
        return 60.0
    return min(5.0 * (2 ** max(0, attempt - 1)), 60.0)


def _target_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM text."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start:end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError(f"LLM response is not a JSON object: {text[:300]}")


@dataclass
class LLMClient:
    model: str
    api_key: str
    base_url: str
    timeout: float = 300.0
    cache_path: Path | None = None
    max_http_retries: int = 8

    def __post_init__(self) -> None:
        self.cache: dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def complete_json(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        cache_key = _stable_id("LLM", task, json.dumps(payload, sort_keys=True, ensure_ascii=False))
        if cache_key in self.cache:
            return self.cache[cache_key]

        request_model = _request_model_name(self.model, self.base_url)
        request_payload = {
            "model": request_model,
            "stream": False,
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict knowledge-graph optimizer. "
                        "Return only valid JSON. Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": _json_dumps({"task": task, **payload}),
                },
            ],
        }
        if not self.api_key:
            raise RuntimeError("OpenRouter API key is required. Pass --api-key or set OPENROUTER_API_KEY.")
        url = _target_url(self.base_url)
        data = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")

        last_error: tuple[int, str] | None = None
        for attempt in range(1, self.max_http_retries + 1):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                    break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = (exc.code, body)
                if exc.code not in RETRYABLE_HTTP_STATUS_CODES or attempt >= self.max_http_retries:
                    raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {body}") from exc
                delay = _http_retry_delay(exc.code, exc.headers, attempt)
                print(
                    f"LLM request failed with HTTP {exc.code}; "
                    f"retrying in {delay:.0f}s ({attempt}/{self.max_http_retries})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        else:
            code, body = last_error or (0, "unknown error")
            raise RuntimeError(f"LLM request failed: HTTP {code}: {body}")

        content = raw["choices"][0]["message"]["content"]
        result = _extract_json_object(content)
        self.cache[cache_key] = result
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(_json_dumps(self.cache), encoding="utf-8")
        return result


def edge_endpoint_keys(edge: dict[str, Any], endpoint: str) -> list[str]:
    exact = edge.get(f"{endpoint}_node_key")
    if exact:
        return [exact]
    return [key for key in edge.get(f"{endpoint}_ambiguous_node_keys", []) if key]


def build_call_indexes(edges: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    callers_by_target: dict[str, set[str]] = {}
    callees_by_source: dict[str, set[str]] = {}
    for edge in edges:
        if edge.get("kind") != "call":
            continue
        source_keys = edge_endpoint_keys(edge, "source")
        target_keys = edge_endpoint_keys(edge, "target")
        for target_key in target_keys:
            callers_by_target.setdefault(target_key, set()).update(source_keys)
        for source_key in source_keys:
            callees_by_source.setdefault(source_key, set()).update(target_keys)
    return callers_by_target, callees_by_source


KNOWLEDGE_PROMPT_FIELDS = ("id", "trigger", "content", "evidence", "confidence")


def knowledge_text(item: dict[str, Any]) -> str:
    return f"{item.get('trigger', '')}\n{item.get('content', '')}".strip()


def knowledge_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def knowledge_prompt_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {field: item.get(field) for field in KNOWLEDGE_PROMPT_FIELDS}


def unique_knowledge_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = knowledge_id(item)
        if not item_id or item_id in seen:
            continue
        unique.append(item)
        seen.add(item_id)
    return unique


def source_value_records(cluster_items: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [
        {
            "id": knowledge_id(item),
            field: item.get(field),
        }
        for item in cluster_items
        if knowledge_id(item)
    ]


def forget_llm_cache_entry(llm: LLMClient, task: str, payload: dict[str, Any]) -> None:
    cache_key = _stable_id("LLM", task, json.dumps(payload, sort_keys=True, ensure_ascii=False))
    cache = getattr(llm, "cache", None)
    if not isinstance(cache, dict) or cache_key not in cache:
        return
    del cache[cache_key]
    cache_path = getattr(llm, "cache_path", None)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json_dumps(cache), encoding="utf-8")


def complete_json_with_regeneration_retry(
    llm: LLMClient,
    task: str,
    payload: dict[str, Any],
    is_valid,
    max_attempts: int = 2,
) -> tuple[dict[str, Any], bool]:
    """Retry the original LLM request when the response is not valid JSON shape."""

    last_result: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        try:
            result = llm.complete_json(task, payload)
        except ValueError:
            forget_llm_cache_entry(llm, task, payload)
            if attempt + 1 >= max_attempts:
                raise
            continue
        if is_valid(result):
            return result, attempt > 0
        last_result = result
        forget_llm_cache_entry(llm, task, payload)
    if last_result is None:
        raise ValueError("LLM response is not a valid JSON object after retry")
    return last_result, max_attempts > 1


def node_payload(node: dict[str, Any] | None) -> dict[str, Any]:
    if not node:
        return {}
    payload = {
        "node_key": node.get("node_key"),
        "id": node.get("id"),
        "kind": node.get("kind"),
        "name": node.get("name"),
        "qualified_name": node.get("qualified_name"),
        "parent_class": node.get("parent_class"),
        "file_path": node.get("file_path"),
        "lineno": node.get("lineno"),
        "end_lineno": node.get("end_lineno"),
        "docstring": node.get("docstring"),
        "source_code": node.get("source_code", ""),
    }
    if node.get("source_code_error"):
        payload["source_code_error"] = node.get("source_code_error")
    return payload


def build_bidirectional_call_neighbors(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    neighbors: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        if edge.get("kind") != "call":
            continue
        source_keys = edge_endpoint_keys(edge, "source")
        target_keys = edge_endpoint_keys(edge, "target")
        for source_key in source_keys:
            for target_key in target_keys:
                entries = (
                    (source_key, target_key, "callee"),
                    (target_key, source_key, "caller"),
                )
                for current_key, neighbor_key, direction in entries:
                    dedup_key = (current_key, neighbor_key, direction)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    neighbors.setdefault(current_key, []).append({
                        "node_key": neighbor_key,
                        "direction": direction,
                        "edge": edge,
                    })
    for node_neighbors in neighbors.values():
        node_neighbors.sort(key=lambda item: (item["node_key"], item["direction"]))
    return neighbors


def structured_task_prompt(
    *,
    task_background: str,
    goal: str,
    input_fields: list[str],
    judgment_pipeline: list[str],
    output: str,
) -> str:
    """Build the structured task text sent in the LLM user payload."""

    sections = [
        ("Task Background", [task_background]),
        ("Goal", [goal]),
        ("Input Fields", input_fields),
        ("Judgment Pipeline", judgment_pipeline),
        ("Output", [output]),
    ]
    rendered: list[str] = []
    for title, lines in sections:
        body = "\n".join(f"- {line}" for line in lines)
        rendered.append(f"## {title}\n{body}")
    return "\n\n".join(rendered)


def propagation_prompt() -> str:
    common_background = (
        "The input graph is a dependency graph of functions and methods. Nodes "
        "represent code entities, call edges represent caller/callee relationships, "
        "and practice-knowledge items are implementation rules mounted on graph nodes."
    )
    common_inputs = [
        "`knowledge`: the practice-knowledge item being evaluated, including id, trigger, content, evidence, and confidence.",
        "`is_propagated_to_current_node`: true when this knowledge was propagated onto current_node from another node before this decision; false when this knowledge is current_node's direct knowledge.",
        "`current_node`: the node currently expanding the propagation frontier, including source_code.",
        "`target_node`: the candidate node reached by this one call edge, including source_code.",
        "`edge`: direction and meaning for the candidate caller/callee relationship.",
    ]
    return structured_task_prompt(
        task_background=common_background,
        goal=(
            "Decide whether the practice-knowledge item should propagate across "
            "this one call-graph edge to target_node."
        ),
        input_fields=common_inputs,
        judgment_pipeline=[
            "Read the knowledge trigger/content/evidence and identify the concrete implementation rule it states.",
            "Use current_node and target_node source code plus edge direction to decide whether target_node likely needs this rule.",
            "Use is_propagated_to_current_node only as provenance context; still judge the current edge on its own.",
            "Propagate only when target_node's implementation likely needs the rule because it calls, is called by, wraps, delegates to, validates for, or consumes behavior from current_node.",
            "Reject local implementation details that do not constrain target_node.",
            "Provide a concise reason.",
        ],
        output=(
            "Return only JSON: {\"propagate\": boolean, \"reason\": string}."
        ),
    )


def knowledge_relation_prompt() -> str:
    input_fields = [
        "`node`: the graph node whose attached knowledge items are being canonicalized, including source_code.",
        "`knowledge_a`: first candidate knowledge item, including id, trigger, content, evidence, and confidence.",
        "`knowledge_b`: second candidate knowledge item, including id, trigger, content, evidence, and confidence.",
    ]
    pipeline = [
        "Identify the concrete implementation rule stated by each knowledge item.",
        "Use node source code to ground whether the rules apply to the same implementation condition.",
        "Choose `duplicate` only for surface paraphrases, restatements, or one item being a strict subset of the other when merging would not lose any concrete constraint.",
        "Choose `conflict` only when both rules cannot be true under the same relevant condition; if so, choose exactly one rule to keep: `a` or `b`.",
        "Choose `independent` when the items cover complementary rules, different scenarios, different inputs, or different implementation aspects.",
        "For duplicate or independent, set keep to null.",
        "Provide a concise reason.",
    ]
    return structured_task_prompt(
        task_background=(
            "The optimizer is canonicalizing and resolving practice knowledge "
            "attached to one function or method node in the dependency graph. "
            "Direct and propagated knowledge may appear together on the same node."
        ),
        goal=(
            "Classify how two same-node practice-knowledge items relate for implementation."
        ),
        input_fields=input_fields,
        judgment_pipeline=pipeline,
        output=(
            "Return only JSON: {\"relationship\": \"duplicate\"|\"conflict\"|"
            "\"independent\", \"keep\": \"a\"|\"b\"|null, \"reason\": string}."
        ),
    )


def merge_prompt() -> str:
    input_fields = [
        "`node`: the graph node whose duplicate cluster is being merged, including source_code.",
        "`knowledge_items`: semantically duplicate practice-knowledge items to merge, each with id, trigger, content, evidence, and confidence.",
    ]
    pipeline = [
        "Read every item in the cluster and identify all concrete constraints that must be preserved.",
        "Use node source code to keep the merged rule grounded in the function implementation.",
        "Write one concise rule for this node that covers the duplicate cluster.",
        "Preserve all concrete constraints from the inputs.",
        "Do not invent new constraints, examples, APIs, or behavior not present in the inputs.",
    ]
    return structured_task_prompt(
        task_background=(
            "The optimizer has grouped duplicate practice knowledge attached to "
            "one dependency-graph node and is creating one canonical knowledge item."
        ),
        goal="Merge the duplicate cluster into one concise canonical implementation rule.",
        input_fields=input_fields,
        judgment_pipeline=pipeline,
        output=(
            "Return only JSON: {\"trigger\": string, \"content\": string}."
        ),
    )


def judge_edge_propagation(
    llm: LLMClient,
    item: dict[str, Any],
    current_node: dict[str, Any],
    target_node: dict[str, Any],
    direction: str,
    is_propagated_to_current_node: bool = False,
) -> tuple[dict[str, Any], bool]:
    return complete_json_with_regeneration_retry(
        llm=llm,
        task=propagation_prompt(),
        is_valid=validate_propagation_decision,
        payload={
            "knowledge": knowledge_prompt_payload(item),
            "is_propagated_to_current_node": is_propagated_to_current_node,
            "current_node": node_payload(current_node),
            "target_node": node_payload(target_node),
            "edge": {
                "direction": direction,
                "meaning": (
                    "target is a callee/helper used by current"
                    if direction == "callee"
                    else "target is a caller/wrapper of current"
                ),
            },
        },
    )


def validate_propagation_decision(decision: dict[str, Any]) -> bool:
    return isinstance(decision.get("propagate"), bool)


def propagation_stats(decisions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "num_propagation_decisions": len(decisions),
        "num_accepted_propagation_decisions": sum(
            1 for item in decisions if item.get("status") == "accepted"
        ),
        "num_rejected_propagation_decisions": sum(
            1 for item in decisions if str(item.get("status", "")).startswith("rejected")
        ),
        "num_capped_propagation_decisions": sum(
            1
            for item in decisions
            if str(item.get("status", "")).startswith("capped")
        ),
    }


def propagate_with_llm_edge_gating(
    graph: dict[str, Any],
    llm: LLMClient,
    max_hops: int = DEFAULT_PROPAGATION_MAX_HOPS,
    max_targets_per_knowledge: int = DEFAULT_PROPAGATION_MAX_TARGETS_PER_KNOWLEDGE,
    max_propagation_decisions: int | None = DEFAULT_MAX_PROPAGATION_DECISIONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_key = {node["node_key"]: node for node in graph.get("nodes", []) if node.get("node_key")}
    neighbors_by_node = build_bidirectional_call_neighbors(graph.get("edges", []))

    propagated: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    llm_decision_count = 0
    graph_decision_cap = max_propagation_decisions if max_propagation_decisions and max_propagation_decisions > 0 else None

    for origin_key in sorted(nodes_by_key):
        origin_node = nodes_by_key[origin_key]
        direct_items = unique_knowledge_items(origin_node.get("knowledge", {}).get("direct", []))
        if not direct_items or not neighbors_by_node.get(origin_key):
            continue
        for item in direct_items:
            source_id = knowledge_id(item)
            if not source_id:
                continue
            queue: list[tuple[str, list[str], int, bool]] = [(origin_key, [origin_key], 0, False)]
            queue_index = 0
            accepted_targets: set[str] = set()
            seen_candidates: set[tuple[str, str, str]] = set()
            reported_target_cap = False

            while queue_index < len(queue):
                current_key, path, hop_count, is_propagated_to_current_node = queue[queue_index]
                queue_index += 1
                current_node = nodes_by_key.get(current_key)
                if not current_node:
                    continue
                if hop_count >= max_hops:
                    if neighbors_by_node.get(current_key):
                        decisions.append({
                            "source_id": source_id,
                            "origin_node": origin_key,
                            "from_node": current_key,
                            "path": path,
                            "hop_count": hop_count,
                            "status": "capped_max_hops",
                            "max_hops": max_hops,
                        })
                    continue

                for neighbor in neighbors_by_node.get(current_key, []):
                    if len(accepted_targets) >= max_targets_per_knowledge:
                        if not reported_target_cap:
                            decisions.append({
                                "source_id": source_id,
                                "origin_node": origin_key,
                                "status": "capped_max_targets",
                                "max_targets_per_knowledge": max_targets_per_knowledge,
                            })
                            reported_target_cap = True
                        break

                    target_key = neighbor["node_key"]
                    if target_key not in nodes_by_key or target_key in path:
                        continue
                    if target_key in accepted_targets:
                        continue
                    target_node = nodes_by_key[target_key]
                    target_direct_ids = {
                        knowledge_id(direct_item)
                        for direct_item in target_node.get("knowledge", {}).get("direct", [])
                        if isinstance(direct_item, dict)
                    }
                    if source_id in target_direct_ids:
                        decisions.append({
                            "source_id": source_id,
                            "origin_node": origin_key,
                            "from_node": current_key,
                            "to_node": target_key,
                            "direction": neighbor["direction"],
                            "path": path + [target_key],
                            "hop_count": hop_count + 1,
                            "status": "skipped_existing_direct",
                        })
                        continue

                    candidate_key = (current_key, target_key, neighbor["direction"])
                    if candidate_key in seen_candidates:
                        continue
                    seen_candidates.add(candidate_key)
                    candidate_path = path + [target_key]
                    candidate_hop = hop_count + 1
                    base_decision = {
                        "source_id": source_id,
                        "origin_node": origin_key,
                        "from_node": current_key,
                        "to_node": target_key,
                        "direction": neighbor["direction"],
                        "path": candidate_path,
                        "hop_count": candidate_hop,
                    }

                    if graph_decision_cap is not None and llm_decision_count >= graph_decision_cap:
                        decisions.append({
                            **base_decision,
                            "propagate": False,
                            "is_propagated_to_current_node": is_propagated_to_current_node,
                            "status": "capped_max_graph_propagation_decisions",
                            "max_propagation_decisions": graph_decision_cap,
                            "num_llm_propagation_decisions": llm_decision_count,
                        })
                        return propagated, decisions

                    llm_decision_count += 1
                    try:
                        raw_decision, retried = judge_edge_propagation(
                            llm=llm,
                            item=item,
                            current_node=current_node,
                            target_node=target_node,
                            direction=neighbor["direction"],
                            is_propagated_to_current_node=is_propagated_to_current_node,
                        )
                    except Exception as exc:
                        decisions.append({
                            **base_decision,
                            "propagate": False,
                            "is_propagated_to_current_node": is_propagated_to_current_node,
                            "status": "rejected_error",
                            "error": str(exc),
                        })
                        continue

                    should_propagate = raw_decision.get("propagate")
                    decision = {
                        **base_decision,
                        "propagate": should_propagate,
                        "is_propagated_to_current_node": is_propagated_to_current_node,
                        "reason": raw_decision.get("reason", ""),
                        "retried": retried,
                    }
                    if not isinstance(should_propagate, bool):
                        decisions.append({**decision, "status": "rejected_invalid_schema"})
                        continue
                    if not should_propagate:
                        decisions.append({**decision, "status": "rejected_by_llm"})
                        continue

                    propagated_id = _stable_id("PK", source_id, origin_key, target_key)
                    propagated_item = {
                        "id": propagated_id,
                        "trigger": item.get("trigger", ""),
                        "content": item.get("content", ""),
                        "evidence": item.get("evidence"),
                        "confidence": item.get("confidence"),
                        "format": "propagated_llm",
                        "source_id": source_id,
                        "source_function": item.get("source_function"),
                        "origin_node": origin_key,
                        "from_node": current_key,
                        "to_node": target_key,
                        "propagation_type": "llm_edge_gated_call",
                        "direction": neighbor["direction"],
                        "hop_count": candidate_hop,
                        "path": candidate_path,
                        "is_propagated_to_current_node": is_propagated_to_current_node,
                        "reason": raw_decision.get("reason", ""),
                    }
                    nodes_by_key[target_key].setdefault("knowledge", {}).setdefault("propagated", []).append(propagated_item)
                    propagated.append(propagated_item)
                    accepted_targets.add(target_key)

                    decisions.append({
                        **decision,
                        "status": "accepted",
                    })
                    if candidate_hop >= max_hops:
                        decisions.append({
                            "source_id": source_id,
                            "origin_node": origin_key,
                            "from_node": target_key,
                            "path": candidate_path,
                            "hop_count": candidate_hop,
                            "status": "capped_max_hops",
                            "max_hops": max_hops,
                        })
                    else:
                        queue.append((target_key, candidate_path, candidate_hop, True))

    return propagated, decisions


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

    def clusters(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = {}
        for item in self.parent:
            grouped.setdefault(self.find(item), []).append(item)
        return list(grouped.values())


def merge_cluster_with_llm(
    llm: LLMClient,
    node: dict[str, Any],
    cluster_items: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(cluster_items) == 1:
        item = cluster_items[0]
        return {
            "trigger": item.get("trigger", ""),
            "content": item.get("content", ""),
        }
    result = llm.complete_json(
        merge_prompt(),
        {
            "node": node_payload(node),
            "knowledge_items": [
                knowledge_prompt_payload(item)
                for item in cluster_items
            ],
        },
    )
    return {
        "trigger": str(result.get("trigger") or cluster_items[0].get("trigger", "")),
        "content": str(result.get("content") or cluster_items[0].get("content", "")),
    }


def normalize_relation_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "relationship": decision.get("relationship"),
        "keep": decision.get("keep"),
        "reason": decision.get("reason", ""),
    }


def validate_knowledge_relation_decision(decision: dict[str, Any]) -> bool:
    normalized = normalize_relation_decision(decision)
    relationship = normalized.get("relationship")
    if relationship not in KNOWLEDGE_RELATIONSHIPS:
        return False
    keep = normalized.get("keep")
    if relationship == KNOWLEDGE_RELATION_CONFLICT:
        return keep in {"a", "b"}
    return keep is None


def judge_knowledge_relation(
    llm: LLMClient,
    node: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    return complete_json_with_regeneration_retry(
        llm=llm,
        task=knowledge_relation_prompt(),
        is_valid=validate_knowledge_relation_decision,
        payload={
            "node": node_payload(node),
            "knowledge_a": knowledge_prompt_payload(left),
            "knowledge_b": knowledge_prompt_payload(right),
        },
    )


def deduplicate_by_node(
    graph: dict[str, Any],
    propagated_items: list[dict[str, Any]],
    llm: LLMClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_by_id = {
        item["id"]: item
        for item in graph.get("knowledge_items", [])
        if item.get("id")
    }
    propagated_by_id = {
        item["id"]: item
        for item in propagated_items
        if item.get("id")
    }
    all_items = {**raw_by_id, **propagated_by_id}

    canonical_items: list[dict[str, Any]] = []
    merge_report: list[dict[str, Any]] = []
    relation_decisions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for node in graph.get("nodes", []):
        node_key = node.get("node_key")
        if not node_key:
            continue
        knowledge = node.setdefault("knowledge", {})
        node_items = unique_knowledge_items(
            knowledge.get("direct", []) + knowledge.get("propagated", [])
        )
        for item in node_items:
            all_items.setdefault(knowledge_id(item), item)
        ids = [knowledge_id(item) for item in node_items if knowledge_id(item) in all_items]
        if not ids:
            knowledge["canonical"] = []
            continue

        union_find = UnionFind(ids)
        conflict_candidates: list[dict[str, Any]] = []
        for idx, left_id in enumerate(ids):
            for right_id in ids[idx + 1:]:
                left_item = all_items[left_id]
                right_item = all_items[right_id]
                try:
                    raw_decision, retried = judge_knowledge_relation(
                        llm,
                        node,
                        left_item,
                        right_item,
                    )
                except Exception as exc:
                    relation_decisions.append({
                        "node_key": node_key,
                        "ids": [left_id, right_id],
                        "relationship": KNOWLEDGE_RELATION_INDEPENDENT,
                        "error": str(exc),
                    })
                    continue
                decision = normalize_relation_decision(raw_decision)
                relationship = decision.get("relationship")
                relation_record = {
                    "node_key": node_key,
                    "ids": [left_id, right_id],
                    "relationship": relationship,
                    "keep": decision.get("keep"),
                    "reason": decision.get("reason", ""),
                    "retried": retried,
                }
                relation_decisions.append(relation_record)
                if relationship == KNOWLEDGE_RELATION_DUPLICATE:
                    union_find.union(left_id, right_id)
                elif relationship == KNOWLEDGE_RELATION_CONFLICT:
                    conflict_candidates.append({
                        **relation_record,
                        "left_id": left_id,
                        "right_id": right_id,
                    })

        root_to_ids: dict[str, list[str]] = {}
        for item_id in ids:
            root_to_ids.setdefault(union_find.find(item_id), []).append(item_id)
        removed_roots: set[str] = set()

        def canonical_id_for_root(root: str) -> str:
            return _stable_id("CK", node_key, *sorted(root_to_ids[root]))

        for candidate in conflict_candidates:
            left_id = candidate["left_id"]
            right_id = candidate["right_id"]
            left_root = union_find.find(left_id)
            right_root = union_find.find(right_id)
            if left_root == right_root or left_root in removed_roots or right_root in removed_roots:
                continue
            left_canonical_id = canonical_id_for_root(left_root)
            right_canonical_id = canonical_id_for_root(right_root)
            conflict_record = {
                "conflict_id": _stable_id("C", left_id, right_id, "same_node"),
                "source_ids": [left_id, right_id],
                "canonical_ids": [left_canonical_id, right_canonical_id],
                "relation": "same_node",
                "is_conflict": True,
                "reason": candidate.get("reason", ""),
                "retried": candidate.get("retried", False),
            }
            keep_choice = str(candidate.get("keep") or "").strip().lower()
            if keep_choice == "a":
                kept_root, removed_root = left_root, right_root
            elif keep_choice == "b":
                kept_root, removed_root = right_root, left_root
            else:
                conflict_record["resolution_status"] = "unresolved_invalid_keep"
                conflicts.append(conflict_record)
                continue
            conflict_record["recommended_keep_id"] = canonical_id_for_root(kept_root)
            removed_roots.add(removed_root)
            conflict_record["kept_id"] = canonical_id_for_root(kept_root)
            conflict_record["removed_id"] = canonical_id_for_root(removed_root)
            conflict_record["resolution_status"] = "removed_same_node_conflict"
            conflicts.append(conflict_record)

        canonical_objects: list[dict[str, Any]] = []
        removed_canonical_objects: list[dict[str, Any]] = []
        for root, cluster_ids in root_to_ids.items():
            cluster_items = [all_items[item_id] for item_id in cluster_ids]
            canonical_id = _stable_id("CK", node_key, *sorted(cluster_ids))
            if root in removed_roots:
                merged = {
                    "trigger": cluster_items[0].get("trigger", ""),
                    "content": cluster_items[0].get("content", ""),
                }
                status = "removed_conflict"
            else:
                merged = merge_cluster_with_llm(
                    llm,
                    node,
                    cluster_items,
                )
                status = "active"
            canonical = {
                "id": canonical_id,
                "trigger": merged["trigger"],
                "content": merged["content"],
                "evidence": source_value_records(cluster_items, "evidence"),
                "confidence": source_value_records(cluster_items, "confidence"),
                "node_key": node_key,
                "source_ids": sorted(cluster_ids),
                "status": status,
            }
            if status == "active":
                canonical_objects.append(canonical)
            else:
                removed_canonical_objects.append(canonical)
            canonical_items.append(canonical)
            merge_report.append({
                "node_key": node_key,
                "id": canonical_id,
                "source_ids": sorted(cluster_ids),
                "cluster_size": len(cluster_ids),
                "status": status,
            })
        knowledge["canonical"] = canonical_objects
        knowledge["removed"] = removed_canonical_objects

    return canonical_items, merge_report, relation_decisions, conflicts


def build_optimized_graph(
    graph_path: str | Path,
    llm: LLMClient,
    max_propagation_decisions: int | None = DEFAULT_MAX_PROPAGATION_DECISIONS,
) -> dict[str, Any]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))

    propagated_items, propagation_decisions = propagate_with_llm_edge_gating(
        graph,
        llm,
        max_propagation_decisions=max_propagation_decisions,
    )
    canonical_items, merge_report, relation_decisions, conflicts = deduplicate_by_node(
        graph=graph,
        propagated_items=propagated_items,
        llm=llm,
    )

    active_canonical = [item for item in canonical_items if item.get("status") == "active"]
    implemented_steps = [
        "llm_edge_gated_bidirectional_call_propagation",
        "llm_pairwise_same_node_relation_with_llm_merge",
    ]
    propagation_report_stats = propagation_stats(propagation_decisions)
    propagation_config = {
        "mode": "llm_edge_gated_bidirectional_call",
        "max_hops": DEFAULT_PROPAGATION_MAX_HOPS,
        "max_targets_per_knowledge": DEFAULT_PROPAGATION_MAX_TARGETS_PER_KNOWLEDGE,
        "max_propagation_decisions": max_propagation_decisions,
    }
    return {
        "$schema": SCHEMA,
        "meta": {
            "base_graph": str(graph_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "propagation_enabled": True,
            "propagation_config": propagation_config,
            "implemented_steps": implemented_steps,
        },
        "stats": {
            "num_nodes": len(graph.get("nodes", [])),
            "num_edges": len(graph.get("edges", [])),
            "num_raw_knowledge_items": len(graph.get("knowledge_items", [])),
            "num_propagated_knowledge_items": len(propagated_items),
            **propagation_report_stats,
            "num_canonical_knowledge_items": len(canonical_items),
            "num_active_canonical_knowledge_items": len(active_canonical),
            "num_conflicts": len([c for c in conflicts if c.get("is_conflict")]),
        },
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "knowledge_items": graph.get("knowledge_items", []),
        "propagated_knowledge_items": propagated_items,
        "canonical_knowledge_items": canonical_items,
        "conflicts": conflicts,
        "reports": {
            "propagation_decisions": propagation_decisions,
            "knowledge_relation_decisions": relation_decisions,
            "merge_report": merge_report,
        },
    }


def default_output_path(graph_path: Path) -> Path:
    return graph_path.with_name("dep_graph.with_knowledge.optimized.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, help="Path to dep_graph.with_knowledge.json")
    parser.add_argument("--framework", help="Framework name for default mounted graph path lookup")
    parser.add_argument("--example", help="Example name for default mounted graph path lookup")
    parser.add_argument("--output", type=Path, help="Output optimized graph JSON path")
    parser.add_argument("--model", default=os.environ.get("OPTIMIZE_KNOWLEDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("OPTIMIZE_KNOWLEDGE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--cache-path", type=Path, help="LLM decision cache path")
    parser.add_argument(
        "--max-propagation-decisions",
        type=int,
        default=int(os.environ.get("OPTIMIZE_KNOWLEDGE_MAX_PROPAGATION_DECISIONS", DEFAULT_MAX_PROPAGATION_DECISIONS)),
        help=(
            "Graph-level cap on LLM propagation decisions. "
            "Use 0 to disable the cap. "
            f"Defaults to {DEFAULT_MAX_PROPAGATION_DECISIONS}."
        ),
    )
    args = parser.parse_args()

    if args.graph_path:
        graph_file = args.graph_path
    elif args.framework and args.example:
        graph_file = dep_graph_path(args.framework, args.example).with_name("dep_graph.with_knowledge.json")
    else:
        parser.error("Either --graph-path or both --framework and --example are required")

    if args.output:
        output = args.output
    else:
        output = default_output_path(graph_file)
    cache_path = args.cache_path or output.with_suffix(".llm_cache.json")
    llm = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        cache_path=cache_path,
    )
    artifact = build_optimized_graph(
        graph_path=graph_file,
        llm=llm,
        max_propagation_decisions=args.max_propagation_decisions,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json_dumps(artifact), encoding="utf-8")
    print(f"Wrote {output}")
    print(_json_dumps(artifact["stats"]))


if __name__ == "__main__":
    main()
