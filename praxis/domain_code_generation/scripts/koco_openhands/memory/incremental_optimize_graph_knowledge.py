"""Incrementally organize online practice knowledge on an optimized graph.

Mode B reconciles newly mounted online knowledge with existing canonical
knowledge on the same node, without propagation. Mode C starts from a Mode B
artifact, propagates only newly active online canonical knowledge one hop to
callers, and reconciles the propagated candidates on each target node.

Existing canonical objects keep their IDs and confidence feedback. This is
deliberately different from the full Stage 3 optimizer, which rebuilds all
canonical knowledge from raw direct and propagated items.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.confidence import (
    DEFAULT_CONFLICT_CONFIDENCE_MARGIN,
    configured_confidence_threshold,
    knowledge_confidence_score,
)
from memory.optimize_graph_knowledge import (
    KNOWLEDGE_RELATION_CONFLICT,
    KNOWLEDGE_RELATION_DUPLICATE,
    KNOWLEDGE_RELATION_INDEPENDENT,
    KNOWLEDGE_RELATIONSHIPS,
    LLMClient,
    _json_dumps,
    _stable_id,
    complete_json_with_regeneration_retry,
    judge_edge_propagation,
    knowledge_prompt_payload,
    merge_cluster_with_llm,
    node_payload,
)


SCHEMA = "DEP_GRAPH_KNOWLEDGE_INCREMENTAL_OPTIMIZED_V1"
MODE_B = "b"
MODE_C = "c"
DEFAULT_MAX_CALLER_TARGETS = 3


def incremental_relation_prompt() -> str:
    return """\
Classify two knowledge rules attached to the same dependency-graph node.

Inputs:
- node: the code node and its source.
- knowledge_a and knowledge_b: the candidate rules.
- role_a and role_b: existing_canonical, online_local, or online_propagated.

Return only JSON:
{"relationship":"duplicate"|"conflict"|"independent","keep":"a"|"b"|null,"reason":"..."}

Rules:
- duplicate means the rules are paraphrases or one is a strict subset and
  retaining one rule loses no concrete implementation constraint.
- conflict means both rules cannot be true under the same relevant condition.
- independent means they cover complementary constraints or different cases.
- For a conflict, choose exactly one rule to keep.
- Prefer a well-grounded existing canonical rule over a single online trace
  when both remain usable, because existing canonical knowledge may contain
  ground-truth practice evidence and accumulated online feedback.
- Prefer exact local knowledge over propagated knowledge when evidence is
  otherwise comparable.
- Do not classify merely related rules as duplicates.
"""


def validate_relation(decision: dict[str, Any]) -> bool:
    relationship = decision.get("relationship")
    if relationship not in KNOWLEDGE_RELATIONSHIPS:
        return False
    keep = decision.get("keep")
    if relationship == KNOWLEDGE_RELATION_CONFLICT:
        return keep in {"a", "b"}
    return keep is None


def judge_incremental_relation(
    llm: LLMClient,
    node: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_role: str,
    right_role: str,
) -> tuple[dict[str, Any], bool]:
    return complete_json_with_regeneration_retry(
        llm=llm,
        task=incremental_relation_prompt(),
        is_valid=validate_relation,
        payload={
            "node": node_payload(node),
            "knowledge_a": knowledge_prompt_payload(left),
            "knowledge_b": knowledge_prompt_payload(right),
            "role_a": left_role,
            "role_b": right_role,
        },
    )


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def _unique_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = _item_id(item)
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        result.append(item)
    return result


def _active_canonical(knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in knowledge.get("canonical", [])
        if isinstance(item, dict) and item.get("status", "active") == "active"
    ]


def _confidence_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_scores = [
        {"id": _item_id(item), "score": knowledge_confidence_score(item)}
        for item in items
        if _item_id(item)
    ]
    if len(items) == 1 and isinstance(items[0].get("confidence"), dict):
        return copy.deepcopy(items[0]["confidence"])
    return {
        "score": round(max((entry["score"] for entry in source_scores), default=0.0), 4),
        "aggregation": "max_correlated_online",
        "source_scores": source_scores,
    }


def _source_records(items: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [
        {"id": _item_id(item), field: copy.deepcopy(item.get(field))}
        for item in items
        if _item_id(item)
    ]


def _merge_candidate_cluster(
    llm: LLMClient,
    node: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, str]:
    if len(items) == 1:
        return {
            "trigger": str(items[0].get("trigger") or ""),
            "content": str(items[0].get("content") or ""),
        }
    try:
        return merge_cluster_with_llm(llm, node, items)
    except Exception:
        best = max(items, key=knowledge_confidence_score)
        return {
            "trigger": str(best.get("trigger") or ""),
            "content": str(best.get("content") or ""),
        }


def _candidate_canonical(
    llm: LLMClient,
    node: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    origin: str,
    status: str = "active",
) -> dict[str, Any]:
    source_ids = sorted(_item_id(item) for item in items if _item_id(item))
    merged = _merge_candidate_cluster(llm, node, items)
    return {
        "id": _stable_id("ICK", str(node.get("node_key") or ""), *source_ids),
        "trigger": merged["trigger"],
        "content": merged["content"],
        "evidence": _source_records(items, "evidence"),
        "confidence": _confidence_payload(items),
        "confidence_sources": _source_records(items, "confidence"),
        "node_key": node.get("node_key"),
        "source_ids": source_ids,
        "status": status,
        "incremental_origin": origin,
    }


def _append_removed(
    knowledge: dict[str, Any],
    item: dict[str, Any],
    *,
    status: str,
    resolution: dict[str, Any],
) -> None:
    removed = copy.deepcopy(item)
    removed["status"] = status
    removed["incremental_resolution"] = resolution
    existing = {
        _item_id(candidate)
        for candidate in knowledge.setdefault("removed", [])
        if isinstance(candidate, dict)
    }
    if _item_id(removed) not in existing:
        knowledge["removed"].append(removed)


def _remove_active_canonical(
    knowledge: dict[str, Any],
    item: dict[str, Any],
    *,
    status: str,
    resolution: dict[str, Any],
) -> None:
    item_id = _item_id(item)
    knowledge["canonical"] = [
        candidate
        for candidate in knowledge.get("canonical", [])
        if _item_id(candidate) != item_id
    ]
    _append_removed(
        knowledge,
        item,
        status=status,
        resolution=resolution,
    )


def _absorb_candidate(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    relationship: str,
    reason: str,
) -> None:
    existing["source_ids"] = sorted({
        *[str(value) for value in existing.get("source_ids", []) if str(value)],
        *[str(value) for value in candidate.get("source_ids", []) if str(value)],
    })
    absorbed = existing.setdefault("incremental_absorbed", [])
    absorbed.append({
        "canonical_id": candidate.get("id"),
        "source_ids": copy.deepcopy(candidate.get("source_ids", [])),
        "evidence": copy.deepcopy(candidate.get("evidence")),
        "confidence": copy.deepcopy(candidate.get("confidence")),
        "relationship": relationship,
        "reason": reason,
    })


def _choose_candidate_conflict(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    llm_keep: str,
    min_confidence: float,
    confidence_margin: float,
) -> str:
    existing_score = knowledge_confidence_score(existing)
    candidate_score = knowledge_confidence_score(candidate)
    if existing_score >= min_confidence:
        return "existing"
    if candidate_score >= min_confidence > existing_score:
        return "candidate"
    if abs(existing_score - candidate_score) >= confidence_margin:
        return "existing" if existing_score > candidate_score else "candidate"
    return "existing" if llm_keep == "a" else "candidate"


def _candidate_clusters(
    node: dict[str, Any],
    candidates: list[dict[str, Any]],
    llm: LLMClient,
    *,
    origin: str,
    confidence_margin: float,
    reports: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {_item_id(item): item for item in candidates}
    ids = sorted(by_id)
    union_find = UnionFind(ids)
    conflicts = []

    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1:]:
            left = by_id[left_id]
            right = by_id[right_id]
            try:
                decision, retried = judge_incremental_relation(
                    llm,
                    node,
                    left,
                    right,
                    left_role=origin,
                    right_role=origin,
                )
            except Exception as exc:
                reports["relation_decisions"].append({
                    "node_key": node.get("node_key"),
                    "ids": [left_id, right_id],
                    "relationship": KNOWLEDGE_RELATION_INDEPENDENT,
                    "error": str(exc),
                })
                continue
            relationship = decision.get("relationship")
            record = {
                "node_key": node.get("node_key"),
                "ids": [left_id, right_id],
                "relationship": relationship,
                "keep": decision.get("keep"),
                "reason": decision.get("reason", ""),
                "retried": retried,
                "phase": "candidate_candidate",
            }
            reports["relation_decisions"].append(record)
            if relationship == KNOWLEDGE_RELATION_DUPLICATE:
                union_find.union(left_id, right_id)
            elif relationship == KNOWLEDGE_RELATION_CONFLICT:
                conflicts.append(record)

    roots: dict[str, list[str]] = {}
    for item_id in ids:
        roots.setdefault(union_find.find(item_id), []).append(item_id)
    removed_roots = set()
    for conflict in conflicts:
        left_root = union_find.find(conflict["ids"][0])
        right_root = union_find.find(conflict["ids"][1])
        if left_root == right_root or left_root in removed_roots or right_root in removed_roots:
            continue
        left_items = [by_id[item_id] for item_id in roots[left_root]]
        right_items = [by_id[item_id] for item_id in roots[right_root]]
        left_score = max(map(knowledge_confidence_score, left_items))
        right_score = max(map(knowledge_confidence_score, right_items))
        keep = str(conflict.get("keep") or "")
        if abs(left_score - right_score) >= confidence_margin:
            keep = "a" if left_score > right_score else "b"
        removed_root = right_root if keep == "a" else left_root
        removed_roots.add(removed_root)

    active = []
    removed = []
    for root, cluster_ids in roots.items():
        items = [by_id[item_id] for item_id in cluster_ids]
        if root in removed_roots:
            removed.append(_candidate_canonical(
                llm,
                node,
                items,
                origin=origin,
                status="removed_incremental_conflict",
            ))
        else:
            active.append(_candidate_canonical(
                llm,
                node,
                items,
                origin=origin,
            ))
    return active, removed


def reconcile_candidates(
    graph: dict[str, Any],
    candidate_ids: set[str],
    llm: LLMClient,
    *,
    origin: str,
    min_confidence: float,
    confidence_margin: float,
) -> dict[str, Any]:
    reports = {
        "relation_decisions": [],
        "resolutions": [],
    }
    stats = {
        "num_candidate_items": 0,
        "num_low_confidence_candidates": 0,
        "num_new_active_canonical": 0,
        "num_absorbed_candidates": 0,
        "num_removed_candidate_canonical": 0,
        "num_removed_existing_canonical": 0,
    }

    for node in graph.get("nodes", []):
        knowledge = node.setdefault("knowledge", {})
        candidates = _unique_items([
            item
            for key in ("direct", "propagated")
            for item in knowledge.get(key, [])
            if isinstance(item, dict) and _item_id(item) in candidate_ids
        ])
        if not candidates:
            continue
        stats["num_candidate_items"] += len(candidates)

        eligible = []
        for item in candidates:
            if knowledge_confidence_score(item) >= min_confidence:
                eligible.append(item)
                continue
            canonical = _candidate_canonical(
                llm,
                node,
                [item],
                origin=origin,
                status="removed_low_confidence",
            )
            _append_removed(
                knowledge,
                canonical,
                status="removed_low_confidence",
                resolution={"reason": "candidate confidence below threshold"},
            )
            stats["num_low_confidence_candidates"] += 1
            stats["num_removed_candidate_canonical"] += 1

        active_candidates, removed_candidates = _candidate_clusters(
            node,
            eligible,
            llm,
            origin=origin,
            confidence_margin=confidence_margin,
            reports=reports,
        ) if eligible else ([], [])

        for removed in removed_candidates:
            _append_removed(
                knowledge,
                removed,
                status="removed_incremental_conflict",
                resolution={"reason": "conflict within incremental candidate batch"},
            )
            stats["num_removed_candidate_canonical"] += 1

        for candidate in active_candidates:
            candidate_active = True
            for existing in list(_active_canonical(knowledge)):
                if not candidate_active or _item_id(existing) == _item_id(candidate):
                    continue
                try:
                    decision, retried = judge_incremental_relation(
                        llm,
                        node,
                        existing,
                        candidate,
                        left_role="existing_canonical",
                        right_role=origin,
                    )
                except Exception as exc:
                    reports["relation_decisions"].append({
                        "node_key": node.get("node_key"),
                        "ids": [_item_id(existing), _item_id(candidate)],
                        "relationship": KNOWLEDGE_RELATION_INDEPENDENT,
                        "phase": "candidate_existing",
                        "error": str(exc),
                    })
                    continue
                relationship = decision.get("relationship")
                reason = str(decision.get("reason") or "")
                reports["relation_decisions"].append({
                    "node_key": node.get("node_key"),
                    "ids": [_item_id(existing), _item_id(candidate)],
                    "relationship": relationship,
                    "keep": decision.get("keep"),
                    "reason": reason,
                    "retried": retried,
                    "phase": "candidate_existing",
                })
                if relationship == KNOWLEDGE_RELATION_INDEPENDENT:
                    continue

                existing_score = knowledge_confidence_score(existing)
                candidate_score = knowledge_confidence_score(candidate)
                if relationship == KNOWLEDGE_RELATION_DUPLICATE:
                    if existing_score >= min_confidence or candidate_score < min_confidence:
                        _absorb_candidate(
                            existing,
                            candidate,
                            relationship=relationship,
                            reason=reason,
                        )
                        candidate_active = False
                        stats["num_absorbed_candidates"] += 1
                        reports["resolutions"].append({
                            "node_key": node.get("node_key"),
                            "candidate_id": _item_id(candidate),
                            "existing_id": _item_id(existing),
                            "resolution": "absorbed_into_existing",
                        })
                        break
                    _remove_active_canonical(
                        knowledge,
                        existing,
                        status="removed_replaced_by_incremental_duplicate",
                        resolution={
                            "candidate_id": _item_id(candidate),
                            "reason": reason,
                        },
                    )
                    candidate["source_ids"] = sorted({
                        *candidate.get("source_ids", []),
                        *existing.get("source_ids", []),
                    })
                    candidate.setdefault("supersedes_canonical_ids", []).append(
                        _item_id(existing)
                    )
                    stats["num_removed_existing_canonical"] += 1
                    continue

                winner = _choose_candidate_conflict(
                    existing,
                    candidate,
                    llm_keep=str(decision.get("keep") or ""),
                    min_confidence=min_confidence,
                    confidence_margin=confidence_margin,
                )
                if winner == "existing":
                    _append_removed(
                        knowledge,
                        candidate,
                        status="removed_conflict_with_existing",
                        resolution={
                            "kept_id": _item_id(existing),
                            "reason": reason,
                        },
                    )
                    candidate_active = False
                    stats["num_removed_candidate_canonical"] += 1
                    reports["resolutions"].append({
                        "node_key": node.get("node_key"),
                        "candidate_id": _item_id(candidate),
                        "existing_id": _item_id(existing),
                        "resolution": "existing_won_conflict",
                    })
                    break
                _remove_active_canonical(
                    knowledge,
                    existing,
                    status="removed_conflict_with_incremental",
                    resolution={
                        "kept_id": _item_id(candidate),
                        "reason": reason,
                    },
                )
                stats["num_removed_existing_canonical"] += 1

            if candidate_active:
                knowledge.setdefault("canonical", []).append(candidate)
                stats["num_new_active_canonical"] += 1

    return {
        "stats": stats,
        "reports": reports,
    }


def _candidate_ids_from_mounted_graph(
    graph: dict[str, Any],
    candidate_source_layer: str | None = None,
) -> set[str]:
    return {
        _item_id(item)
        for item in graph.get("knowledge_items", [])
        if isinstance(item, dict)
        and _item_id(item)
        and (
            not candidate_source_layer
            or str(item.get("source_layer") or item.get("profile") or "")
            == candidate_source_layer
        )
    }


def _collect_node_items(graph: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return _unique_items([
        item
        for node in graph.get("nodes", [])
        for item in node.get("knowledge", {}).get(key, [])
        if isinstance(item, dict)
    ])


def _build_callers_by_target(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    callers = {}
    for edge in edges:
        if edge.get("kind") != "call":
            continue
        source_keys = (
            [edge.get("source_node_key")]
            if edge.get("source_node_key")
            else edge.get("source_ambiguous_node_keys", [])
        )
        target_keys = (
            [edge.get("target_node_key")]
            if edge.get("target_node_key")
            else edge.get("target_ambiguous_node_keys", [])
        )
        for target in target_keys:
            if not target:
                continue
            callers.setdefault(target, set()).update(
                source for source in source_keys if source
            )
    return {
        target: sorted(values)
        for target, values in callers.items()
    }


def propagate_incremental_to_callers(
    graph: dict[str, Any],
    llm: LLMClient,
    *,
    min_confidence: float,
    max_targets_per_knowledge: int,
    max_candidates_per_target: int,
) -> dict[str, Any]:
    nodes_by_key = {
        str(node.get("node_key")): node
        for node in graph.get("nodes", [])
        if node.get("node_key")
    }
    callers_by_target = _build_callers_by_target(graph.get("edges", []))
    sources = [
        (node, item)
        for node in graph.get("nodes", [])
        for item in _active_canonical(node.get("knowledge", {}))
        if item.get("incremental_origin") == "online_local"
        and knowledge_confidence_score(item) >= min_confidence
    ]
    accepted_per_target: dict[str, int] = {}
    propagated = []
    decisions = []

    for origin_node, item in sources:
        origin_key = str(origin_node.get("node_key") or "")
        accepted_for_item = 0
        for target_key in callers_by_target.get(origin_key, []):
            if accepted_for_item >= max_targets_per_knowledge:
                break
            if accepted_per_target.get(target_key, 0) >= max_candidates_per_target:
                decisions.append({
                    "source_id": _item_id(item),
                    "origin_node": origin_key,
                    "to_node": target_key,
                    "status": "capped_target_candidate_limit",
                })
                continue
            target_node = nodes_by_key.get(target_key)
            if target_node is None:
                continue
            try:
                decision, retried = judge_edge_propagation(
                    llm=llm,
                    item=item,
                    current_node=origin_node,
                    target_node=target_node,
                    direction="caller",
                    is_propagated_to_current_node=False,
                )
            except Exception as exc:
                decisions.append({
                    "source_id": _item_id(item),
                    "origin_node": origin_key,
                    "to_node": target_key,
                    "status": "rejected_error",
                    "error": str(exc),
                })
                continue
            if decision.get("propagate") is not True:
                decisions.append({
                    "source_id": _item_id(item),
                    "origin_node": origin_key,
                    "to_node": target_key,
                    "status": "rejected_by_llm",
                    "reason": decision.get("reason", ""),
                    "retried": retried,
                })
                continue
            propagated_item = {
                "id": _stable_id("IPK", _item_id(item), origin_key, target_key),
                "trigger": item.get("trigger", ""),
                "content": item.get("content", ""),
                "evidence": copy.deepcopy(item.get("evidence")),
                "confidence": copy.deepcopy(item.get("confidence")),
                "format": "incremental_propagated_llm",
                "source_id": _item_id(item),
                "source_function": item.get("source_function"),
                "origin_node": origin_key,
                "from_node": origin_key,
                "to_node": target_key,
                "propagation_type": "incremental_one_hop_caller",
                "edge_kind": "call",
                "direction": "caller",
                "hop_count": 1,
                "path": [origin_key, target_key],
                "reason": decision.get("reason", ""),
            }
            target_node.setdefault("knowledge", {}).setdefault(
                "propagated", []
            ).append(propagated_item)
            propagated.append(propagated_item)
            accepted_for_item += 1
            accepted_per_target[target_key] = accepted_per_target.get(target_key, 0) + 1
            decisions.append({
                "source_id": _item_id(item),
                "origin_node": origin_key,
                "to_node": target_key,
                "status": "accepted",
                "reason": decision.get("reason", ""),
                "retried": retried,
            })

    return {
        "propagated_items": propagated,
        "decisions": decisions,
    }


def _finalize_artifact(
    graph: dict[str, Any],
    *,
    mode: str,
    reports: dict[str, Any],
    stats: dict[str, Any],
    source_path: str | Path,
) -> dict[str, Any]:
    canonical = _collect_node_items(graph, "canonical")
    removed = _collect_node_items(graph, "removed")
    propagated = _collect_node_items(graph, "propagated")
    active = [
        item
        for item in canonical
        if item.get("status", "active") == "active"
    ]
    graph["$schema"] = SCHEMA
    graph["meta"] = {
        **copy.deepcopy(graph.get("meta", {})),
        "incremental_base_graph": str(source_path),
        "incremental_mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implemented_steps": (
            ["incremental_same_node_dedup_conflict"]
            if mode == MODE_B
            else [
                "incremental_same_node_dedup_conflict",
                "incremental_one_hop_caller_propagation",
                "incremental_target_node_dedup_conflict",
            ]
        ),
    }
    graph["stats"] = {
        **copy.deepcopy(graph.get("stats", {})),
        **stats,
        "num_canonical_knowledge_items": len(canonical) + len(removed),
        "num_active_canonical_knowledge_items": len(active),
        "num_removed_canonical_knowledge_items": len(removed),
        "num_propagated_knowledge_items": len(propagated),
    }
    graph["canonical_knowledge_items"] = canonical + removed
    graph["propagated_knowledge_items"] = propagated
    graph["incremental_reports"] = reports
    return graph


def build_mode_b(
    graph_path: str | Path,
    llm: LLMClient,
    *,
    min_confidence: float | None = None,
    confidence_margin: float = DEFAULT_CONFLICT_CONFIDENCE_MARGIN,
    candidate_source_layer: str | None = None,
) -> dict[str, Any]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    threshold = configured_confidence_threshold(min_confidence)
    result = reconcile_candidates(
        graph,
        _candidate_ids_from_mounted_graph(
            graph,
            candidate_source_layer=candidate_source_layer,
        ),
        llm,
        origin="online_local",
        min_confidence=threshold,
        confidence_margin=confidence_margin,
    )
    return _finalize_artifact(
        graph,
        mode=MODE_B,
        reports={"local_reconciliation": result["reports"]},
        stats={
            **{
                f"incremental_local_{key}": value
                for key, value in result["stats"].items()
            },
            "incremental_min_confidence": threshold,
        },
        source_path=graph_path,
    )


def build_mode_c(
    graph_path: str | Path,
    llm: LLMClient,
    *,
    min_confidence: float | None = None,
    confidence_margin: float = DEFAULT_CONFLICT_CONFIDENCE_MARGIN,
    max_targets_per_knowledge: int = DEFAULT_MAX_CALLER_TARGETS,
    max_candidates_per_target: int = DEFAULT_MAX_CALLER_TARGETS,
) -> dict[str, Any]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    threshold = configured_confidence_threshold(min_confidence)
    propagation = propagate_incremental_to_callers(
        graph,
        llm,
        min_confidence=threshold,
        max_targets_per_knowledge=max_targets_per_knowledge,
        max_candidates_per_target=max_candidates_per_target,
    )
    propagated_ids = {
        _item_id(item)
        for item in propagation["propagated_items"]
    }
    reconciliation = reconcile_candidates(
        graph,
        propagated_ids,
        llm,
        origin="online_propagated",
        min_confidence=threshold,
        confidence_margin=confidence_margin,
    )
    accepted = sum(
        decision.get("status") == "accepted"
        for decision in propagation["decisions"]
    )
    return _finalize_artifact(
        graph,
        mode=MODE_C,
        reports={
            "propagation_decisions": propagation["decisions"],
            "propagated_reconciliation": reconciliation["reports"],
        },
        stats={
            "incremental_propagation_candidates": len(propagation["decisions"]),
            "incremental_accepted_propagations": accepted,
            **{
                f"incremental_propagated_{key}": value
                for key, value in reconciliation["stats"].items()
            },
            "incremental_min_confidence": threshold,
            "incremental_max_targets_per_knowledge": max_targets_per_knowledge,
            "incremental_max_candidates_per_target": max_candidates_per_target,
        },
        source_path=graph_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=(MODE_B, MODE_C), required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OPTIMIZE_KNOWLEDGE_MODEL",
            "deepseek/deepseek-v3.2",
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "OPTIMIZE_KNOWLEDGE_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--candidate-source-layer")
    parser.add_argument(
        "--conflict-confidence-margin",
        type=float,
        default=DEFAULT_CONFLICT_CONFIDENCE_MARGIN,
    )
    parser.add_argument(
        "--max-targets-per-knowledge",
        type=int,
        default=DEFAULT_MAX_CALLER_TARGETS,
    )
    parser.add_argument(
        "--max-candidates-per-target",
        type=int,
        default=DEFAULT_MAX_CALLER_TARGETS,
    )
    args = parser.parse_args()

    cache_path = args.cache_path or args.output.with_suffix(".llm_cache.json")
    llm = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        cache_path=cache_path,
    )
    if args.mode == MODE_B:
        artifact = build_mode_b(
            args.graph_path,
            llm,
            min_confidence=args.min_confidence,
            confidence_margin=args.conflict_confidence_margin,
            candidate_source_layer=args.candidate_source_layer,
        )
    else:
        artifact = build_mode_c(
            args.graph_path,
            llm,
            min_confidence=args.min_confidence,
            confidence_margin=args.conflict_confidence_margin,
            max_targets_per_knowledge=args.max_targets_per_knowledge,
            max_candidates_per_target=args.max_candidates_per_target,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_json_dumps(artifact), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(_json_dumps(artifact["stats"]))


if __name__ == "__main__":
    main()
