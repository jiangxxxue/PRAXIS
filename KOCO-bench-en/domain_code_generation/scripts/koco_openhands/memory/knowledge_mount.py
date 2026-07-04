"""Mount practice knowledge onto dependency graph nodes.

This module implements the deterministic first stage of knowledge mounting:

1. parse a DEP_GRAPH_V1 graph and add stable node keys;
2. parse profile-scoped practice_knowledge.jsonl files into structured items;
3. mount function-scoped knowledge items to exact graph nodes.

It intentionally does not perform semantic deduplication, propagation, or LLM
conflict resolution. Those steps should consume this module's structured output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.config import GRAPH_KNOWLEDGE_DIR, code_dir, dep_graph_path


SCHEMA = "DEP_GRAPH_KNOWLEDGE_MOUNT_V1"
KNOWLEDGE_HEADING_RE = re.compile(r"^##\s+Knowledge\s+(\d+)\s+[—-]\s+(.+?)\s*$", re.MULTILINE)
LESSON_HEADING_RE = re.compile(r"^##\s+Lesson:\s*(.+?)\s*$", re.MULTILINE)
SECTION_RE = re.compile(r"^\*\*(Lesson|Evidence|Sources):\*\*\s*(.*)$", re.MULTILINE)


@dataclass
class GraphIndex:
    """Parsed dependency graph plus lookup indexes."""

    raw_graph: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    nodes_by_key: dict[str, dict[str, Any]]
    nodes_by_id: dict[str, list[dict[str, Any]]]
    nodes_by_name: dict[str, list[dict[str, Any]]]
    nodes_by_qualified_name: dict[str, dict[str, Any]]
    edges_by_node_key: list[dict[str, Any]] = field(default_factory=list)


def make_node_key(node: dict[str, Any]) -> str:
    """Return a stable node key for a graph node."""

    qualified_name = node.get("qualified_name")
    if qualified_name:
        return str(qualified_name)
    file_path = node.get("file_path", "<unknown>")
    lineno = node.get("lineno", "<unknown>")
    name = node.get("name") or node.get("id") or "<unknown>"
    return f"{file_path}:{lineno}:{name}"


def normalize_source_file_path(path: str | Path | None) -> str:
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


def resolve_source_root(
    graph_path: str | Path,
    framework: str | None = None,
    project: str | None = None,
    source_root: str | Path | None = None,
) -> Path | None:
    if source_root:
        return Path(source_root)
    if framework and project:
        return code_dir(framework, project)
    try:
        rel = Path(graph_path).resolve().relative_to(GRAPH_KNOWLEDGE_DIR.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    return code_dir(parts[0], parts[1])


def read_node_source(node: dict[str, Any], source_root: Path | None) -> tuple[str, str | None]:
    if source_root is None:
        return "", "source_root_not_configured"
    raw_file_path = str(node.get("file_path") or "")
    file_path = normalize_source_file_path(node.get("file_path"))
    if not file_path:
        return "", "missing_file_path"
    raw_source_path = Path(raw_file_path)
    candidates = [raw_source_path] if raw_source_path.is_absolute() else []
    candidates.extend([
        source_root / file_path,
        source_root / raw_file_path,
    ])
    resolved_path = next((path for path in candidates if path.exists() and path.is_file()), None)
    if resolved_path is None:
        return "", f"source_file_not_found:{file_path}"
    try:
        start = int(node.get("lineno") or 0)
        end = int(node.get("end_lineno") or start)
    except (TypeError, ValueError):
        return "", "invalid_line_range"
    if start <= 0 or end < start:
        return "", "invalid_line_range"
    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = resolved_path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return "", f"source_read_error:{exc}"
    if start > len(lines):
        return "", "line_range_out_of_bounds"
    end = min(end, len(lines))
    return "\n".join(lines[start - 1:end]), None


def attach_node_source_code(nodes: list[dict[str, Any]], source_root: Path | None) -> None:
    for node in nodes:
        source_code, source_error = read_node_source(node, source_root)
        node["source_code"] = source_code
        if source_error:
            node["source_code_error"] = source_error
        else:
            node.pop("source_code_error", None)


def parse_graph(graph_path: str | Path, source_root: Path | None = None) -> GraphIndex:
    """Parse a dependency graph and build stable node indexes."""

    graph_path = Path(graph_path)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = []
    nodes_by_key: dict[str, dict[str, Any]] = {}
    nodes_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nodes_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nodes_by_qualified_name: dict[str, dict[str, Any]] = {}

    for node in graph.get("nodes", []):
        normalized = dict(node)
        node_key = make_node_key(normalized)
        normalized["node_key"] = node_key
        normalized.setdefault("knowledge", {"direct": [], "propagated": [], "conflicts": []})
        nodes.append(normalized)
        nodes_by_key[node_key] = normalized
        nodes_by_id[str(normalized.get("id", ""))].append(normalized)
        nodes_by_name[str(normalized.get("name", ""))].append(normalized)
        if normalized.get("qualified_name"):
            nodes_by_qualified_name[str(normalized["qualified_name"])] = normalized
    attach_node_source_code(nodes, source_root)

    edges_by_node_key = [
        normalize_edge(edge, nodes_by_id, nodes_by_key)
        for edge in graph.get("edges", [])
    ]

    return GraphIndex(
        raw_graph=graph,
        nodes=nodes,
        edges=graph.get("edges", []),
        nodes_by_key=nodes_by_key,
        nodes_by_id=dict(nodes_by_id),
        nodes_by_name=dict(nodes_by_name),
        nodes_by_qualified_name=nodes_by_qualified_name,
        edges_by_node_key=edges_by_node_key,
    )


def normalize_edge(
    edge: dict[str, Any],
    nodes_by_id: dict[str, list[dict[str, Any]]],
    nodes_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Add best-effort source/target node keys to an edge."""

    normalized = dict(edge)
    for endpoint in ("source", "target"):
        value = edge.get(endpoint)
        key_field = f"{endpoint}_node_key"
        candidates = nodes_by_id.get(str(value), [])
        if len(candidates) == 1:
            normalized[key_field] = candidates[0]["node_key"]
        elif str(value) in nodes_by_key:
            normalized[key_field] = str(value)
        else:
            normalized[key_field] = None
            if len(candidates) > 1:
                normalized[f"{endpoint}_ambiguous_node_keys"] = [n["node_key"] for n in candidates]
    return normalized


def parse_knowledge_root(
    knowledge_root: str | Path,
    framework: str | None = None,
    project: str | None = None,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    """Parse project-level practice_knowledge.jsonl files under a root directory."""

    root = Path(knowledge_root)
    items: list[dict[str, Any]] = []
    if profile:
        search_root = root / profile
    else:
        combined_root = root / "combined"
        search_root = combined_root if combined_root.exists() else root
    if not search_root.exists():
        return items
    knowledge_paths = sorted(search_root.rglob("practice_knowledge.jsonl"))
    for knowledge_path in knowledge_paths:
        rel_path = knowledge_path.relative_to(root)
        parts = rel_path.parts
        if len(parts) < 2:
            continue
        source_layer = parts[0]
        if profile and source_layer != profile:
            continue
        if not profile and source_layer != "combined":
            continue
        if len(parts) >= 4:
            item_framework = parts[1]
            item_project = parts[2]
        else:
            item_framework = None
            item_project = parts[1]
        if framework and item_framework and item_framework != framework:
            continue
        if project and item_project != project:
            continue
        is_per_function = "per_function" in parts
        source_function = knowledge_path.stem if is_per_function else None
        if knowledge_path.suffix == ".jsonl":
            parsed = parse_jsonl_knowledge_file(
                path=knowledge_path,
                rel_path=str(rel_path),
                source_layer=source_layer,
                framework=item_framework,
                project=item_project,
                source_function=source_function,
                is_per_function=is_per_function,
            )
        else:
            text = knowledge_path.read_text(encoding="utf-8")
            parsed = parse_knowledge_file(
                text=text,
                rel_path=str(rel_path),
                source_layer=source_layer,
                framework=item_framework,
                project=item_project,
                source_function=source_function,
                is_per_function=is_per_function,
            )
        items.extend(parsed)
    return items


def parse_jsonl_knowledge_file(
    path: Path,
    rel_path: str,
    source_layer: str,
    framework: str | None,
    project: str,
    source_function: str | None,
    is_per_function: bool,
) -> list[dict[str, Any]]:
    """Parse newline-delimited JSON knowledge files."""

    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for idx, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            item_source_function = raw.get("source_function") or source_function
            evidence = raw.get("evidence")
            content = raw.get("content") or ""
            trigger = raw.get("trigger") or content[:120]
            item = {
                "id": raw.get("id") or make_knowledge_id(rel_path, idx, trigger),
                "trigger": trigger,
                "content": content,
                "evidence": evidence,
                "confidence": raw.get("confidence"),
                "item_index": idx,
                "format": "jsonl",
                "framework": raw.get("framework") or framework,
                "project": raw.get("example") or project,
                "source_layer": raw.get("profile") or source_layer,
                "source_file": rel_path,
                "source_function": item_source_function,
                "implementation_location": raw.get("implementation_location", ""),
                "function_signature": raw.get("function_signature", ""),
                "is_per_function": is_per_function,
                "sources": [item_source_function] if item_source_function else [],
                "raw_text": line,
            }
            items.append(item)
    return items


def parse_knowledge_file(
    text: str,
    rel_path: str,
    source_layer: str,
    framework: str | None,
    project: str,
    source_function: str | None,
    is_per_function: bool,
) -> list[dict[str, Any]]:
    """Parse one practice knowledge markdown file."""

    if is_per_function:
        return parse_lesson_file(
            text, rel_path, source_layer, framework, project, source_function
        )
    return parse_aggregate_file(text, rel_path, source_layer, framework, project)


def parse_lesson_file(
    text: str,
    rel_path: str,
    source_layer: str,
    framework: str | None,
    project: str,
    source_function: str | None,
) -> list[dict[str, Any]]:
    """Parse per_function files using the '## Lesson:' format."""

    items: list[dict[str, Any]] = []
    matches = list(LESSON_HEADING_RE.finditer(text))
    for idx, match in enumerate(matches, start=1):
        start = match.end()
        end = matches[idx].start() if idx < len(matches) else len(text)
        block = text[start:end].strip()
        fields = parse_lesson_sections(block)
        title = match.group(1).strip()
        content = fields.get("Lesson") or block
        evidence = fields.get("Evidence")
        sources = split_sources(fields.get("Sources"))
        item = {
            "id": make_knowledge_id(rel_path, idx, title),
            "trigger": title,
            "content": content,
            "evidence": evidence,
            "confidence": None,
            "item_index": idx,
            "format": "lesson",
            "framework": framework,
            "project": project,
            "source_layer": source_layer,
            "source_file": rel_path,
            "source_function": source_function,
            "is_per_function": True,
            "sources": sources,
            "raw_text": block,
        }
        items.append(item)
    return items


def parse_lesson_sections(block: str) -> dict[str, str]:
    """Extract **Lesson:**, **Evidence:**, and **Sources:** sections."""

    matches = list(SECTION_RE.finditer(block))
    if not matches:
        return {}

    fields: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1)
        first_line = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        rest = block[start:end].strip()
        value = "\n".join(part for part in (first_line, rest) if part).strip()
        fields[label] = value
    return fields


def parse_aggregate_file(
    text: str,
    rel_path: str,
    source_layer: str,
    framework: str | None,
    project: str,
) -> list[dict[str, Any]]:
    """Parse project aggregate files using the '## Knowledge N - title' format."""

    items: list[dict[str, Any]] = []
    matches = list(KNOWLEDGE_HEADING_RE.finditer(text))
    for idx, match in enumerate(matches, start=1):
        start = match.end()
        end = matches[idx].start() if idx < len(matches) else len(text)
        title = match.group(2).strip()
        content = text[start:end].strip()
        item = {
            "id": make_knowledge_id(rel_path, int(match.group(1)), title),
            "trigger": title,
            "content": content,
            "evidence": None,
            "confidence": None,
            "item_index": int(match.group(1)),
            "format": "knowledge",
            "framework": framework,
            "project": project,
            "source_layer": source_layer,
            "source_file": rel_path,
            "source_function": None,
            "is_per_function": False,
            "sources": [],
            "raw_text": content,
        }
        items.append(item)
    return items


def split_sources(value: str | None) -> list[str]:
    """Split a Sources field into symbol names."""

    if not value:
        return []
    return [part.strip(" `") for part in re.split(r"[,;\n]+", value) if part.strip(" `")]


def make_knowledge_id(rel_path: str, item_index: int, title: str) -> str:
    """Create a stable knowledge id from source path and item title."""

    stem = f"{rel_path}#{item_index}:{title}"
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    return f"K-{digest}"


def exact_mount_per_function(
    graph: GraphIndex,
    knowledge_items: list[dict[str, Any]],
    project: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mount function-scoped knowledge items to exact matching graph nodes."""

    mounts: list[dict[str, Any]] = []
    unmounted: list[dict[str, Any]] = []

    for item in knowledge_items:
        if not item.get("source_function"):
            continue
        if project and item.get("project") != project:
            continue

        symbol = item.get("source_function")
        candidates, match_detail = find_mount_candidates(graph, item)
        if candidates:
            confidence = 1.0 if len(candidates) == 1 else 0.8
            mount_prefix = "per_function" if item.get("is_per_function") else "source_function"
            if match_detail == "implementation_location_target":
                reason = "knowledge implementation_location matched the graph node"
                mount_type = f"{mount_prefix}_implementation_location"
            elif match_detail == "implementation_location_target_narrowed":
                reason = "knowledge implementation_location matched multiple graph nodes; narrowest span selected"
                mount_type = f"{mount_prefix}_implementation_location_narrowed"
            elif match_detail == "location_disambiguated":
                reason = "knowledge source_function matched multiple graph nodes; implementation_location selected one"
                mount_type = f"{mount_prefix}_location_disambiguated"
            elif match_detail == "symbol_hint_disambiguated":
                reason = "knowledge source_function matched multiple graph nodes; parenthetical hint selected one"
                mount_type = f"{mount_prefix}_symbol_hint_disambiguated"
            elif match_detail == "duplicate_implementation_disambiguated":
                reason = "knowledge source_function matched duplicate graph aliases for the same implementation"
                mount_type = f"{mount_prefix}_duplicate_implementation_disambiguated"
            elif len(candidates) == 1:
                reason = "knowledge source_function matched exactly one graph node"
                mount_type = f"{mount_prefix}_exact"
            else:
                reason = "knowledge source_function matched multiple exact graph nodes; mounted to all candidates"
                mount_type = f"{mount_prefix}_exact_ambiguous"
            for node in candidates:
                existing_ids = {
                    existing.get("id")
                    for existing in node.get("knowledge", {}).get("direct", [])
                    if isinstance(existing, dict)
                }
                if item.get("id") in existing_ids:
                    continue
                node["knowledge"]["direct"].append(item)
                mount = {
                    "id": item["id"],
                    "node_key": node["node_key"],
                    "target_symbol": symbol,
                    "mount_type": mount_type,
                    "confidence": confidence,
                    "reason": reason,
                    "source_file": item["source_file"],
                }
                if item.get("implementation_location"):
                    mount["implementation_location"] = item["implementation_location"]
                if len(candidates) > 1:
                    mount["candidate_node_keys"] = [n["node_key"] for n in candidates]
                mounts.append(mount)
        elif match_detail == "ambiguous_unresolved":
            unmounted.append({
                "id": item["id"],
                "target_symbol": symbol,
                "source_file": item["source_file"],
                "implementation_location": item.get("implementation_location", ""),
                "reason": "ambiguous graph node match not location-disambiguated",
            })
        elif not candidates:
            unmounted.append({
                "id": item["id"],
                "target_symbol": symbol,
                "source_file": item["source_file"],
                "implementation_location": item.get("implementation_location", ""),
                "reason": "no exact graph node match",
            })

    return mounts, unmounted


def find_mount_candidates(graph: GraphIndex, item: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Find target nodes, preferring explicit implementation location when present."""

    location_targets, location_detail = find_implementation_location_targets(
        graph,
        item.get("implementation_location"),
    )
    if location_targets:
        return location_targets, location_detail

    symbol, hint = split_symbol_location_hint(item.get("source_function"))
    candidates = find_exact_symbol_candidates(
        graph,
        symbol,
        item.get("project"),
    )
    if len(candidates) <= 1:
        return candidates, "exact"

    location_matches = filter_candidates_by_implementation_location(
        candidates,
        item.get("implementation_location"),
    )
    if len(location_matches) == 1:
        return location_matches, "location_disambiguated"

    hint_matches = filter_candidates_by_symbol_hint(candidates, hint)
    if len(hint_matches) == 1:
        return hint_matches, "symbol_hint_disambiguated"

    deduped = collapse_duplicate_implementation_candidates(candidates)
    if len(deduped) == 1:
        return deduped, "duplicate_implementation_disambiguated"
    return [], "ambiguous_unresolved"


def find_implementation_location_targets(
    graph: GraphIndex,
    implementation_location: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Find target function nodes directly from a source location."""

    location_matches = filter_candidates_by_implementation_location(
        graph.nodes,
        implementation_location,
    )
    if not location_matches:
        return [], "no_location_match"

    deduped = collapse_duplicate_implementation_candidates(location_matches)
    if len(deduped) == 1:
        return deduped, "implementation_location_target"

    narrowed = filter_narrowest_span_candidates(deduped)
    if len(narrowed) == 1:
        return narrowed, "implementation_location_target_narrowed"
    return [], "ambiguous_unresolved"


def split_symbol_location_hint(symbol: str | None) -> tuple[str | None, str | None]:
    """Split direct-memory symbols such as ``compute_score (r1v.py)``."""

    if not symbol:
        return symbol, None
    value = str(symbol).strip()
    match = re.match(r"^(.+?)\s+\(([^()]+)\)\s*$", value)
    if not match:
        return value, None
    return match.group(1).strip(), match.group(2).strip()


def filter_candidates_by_symbol_hint(
    candidates: list[dict[str, Any]],
    hint: str | None,
) -> list[dict[str, Any]]:
    """Use parenthetical file/class hints from direct knowledge to choose a node."""

    if not hint:
        return []
    normalized_hint = normalize_source_file_path(hint)
    hint_stem = Path(normalized_hint).stem
    hints = {normalized_hint, hint_stem}
    matched = []
    for node in candidates:
        file_path = normalize_source_file_path(node.get("file_path"))
        qualified_name = str(node.get("qualified_name") or "")
        node_id = str(node.get("id") or "")
        if any(
            value
            and (
                value in file_path
                or f".{value}." in f".{qualified_name}."
                or qualified_name.endswith(f".{value}")
                or node_id.startswith(f"{value}.")
            )
            for value in hints
        ):
            matched.append(node)
    return matched


def collapse_duplicate_implementation_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse duplicate graph nodes that point at the same implementation span."""

    grouped: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for node in candidates:
        try:
            start = int(node.get("lineno") or 0)
            end = int(node.get("end_lineno") or start)
        except (TypeError, ValueError):
            return candidates
        key = (
            normalize_source_file_path(node.get("file_path")),
            start,
            end,
            str(node.get("id") or ""),
        )
        grouped.setdefault(key, node)
    if len(grouped) == 1:
        return list(grouped.values())
    return candidates


def filter_candidates_by_implementation_location(
    candidates: list[dict[str, Any]],
    implementation_location: str | None,
) -> list[dict[str, Any]]:
    """Prefer candidates whose source file and line range overlaps the target location."""

    parsed = parse_implementation_location(implementation_location)
    if not parsed:
        return []
    target_file, target_start, target_end = parsed
    matched = []
    for node in candidates:
        node_file = normalize_source_file_path(node.get("file_path"))
        if node_file != target_file:
            continue
        try:
            node_start = int(node.get("lineno") or 0)
            node_end = int(node.get("end_lineno") or node_start)
        except (TypeError, ValueError):
            continue
        if node_start <= target_end and target_start <= node_end + 1:
            matched.append(node)
    return matched


def filter_narrowest_span_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer the smallest source span when location matches nested/parent nodes."""

    spans: list[tuple[int, dict[str, Any]]] = []
    for node in candidates:
        try:
            start = int(node.get("lineno") or 0)
            end = int(node.get("end_lineno") or start)
        except (TypeError, ValueError):
            return candidates
        spans.append((max(0, end - start), node))
    if not spans:
        return []
    min_span = min(span for span, _node in spans)
    return [node for span, node in spans if span == min_span]


def parse_implementation_location(value: str | None) -> tuple[str, int, int] | None:
    """Parse strings like 'path/to/file.py:line 10-20'."""

    if not value:
        return None
    match = re.match(r"^(.+?):line\s+(\d+)(?:-(\d+))?$", str(value).strip())
    if not match:
        return None
    file_path = normalize_source_file_path(match.group(1))
    start = int(match.group(2))
    end = int(match.group(3) or start)
    return file_path, start, end


def find_exact_symbol_candidates(
    graph: GraphIndex,
    symbol: str | None,
    project: str | None,
) -> list[dict[str, Any]]:
    """Find graph nodes that exactly match a per_function file stem."""

    if not symbol:
        return []

    candidates: dict[str, dict[str, Any]] = {}
    for node in graph.nodes:
        node_id = str(node.get("id", ""))
        qualified_name = str(node.get("qualified_name", ""))
        if node_id == symbol or qualified_name.endswith(f".{symbol}"):
            candidates[node["node_key"]] = node

        # File stems may use Class.method while node ids use the same string.
        if "." in symbol and (
            node_id == symbol
            or qualified_name.endswith(f".{symbol}")
        ):
            candidates[node["node_key"]] = node

    candidate_list = list(candidates.values())
    if len(candidate_list) <= 1:
        return candidate_list

    scoped = filter_project_scope(candidate_list, project)
    return scoped or candidate_list


def filter_project_scope(nodes: list[dict[str, Any]], project: str | None) -> list[dict[str, Any]]:
    """Prefer nodes whose file paths or qualified names belong to the example."""

    if not project:
        return nodes

    project_lower = project.lower()
    scoped = []
    for node in nodes:
        file_path = str(node.get("file_path", "")).lower()
        qualified_name = str(node.get("qualified_name", "")).lower()
        if (
            f"/{project_lower}/" in f"/{file_path}"
            or f".{project_lower}." in f".{qualified_name}."
            or f"recipe/{project_lower}/" in file_path
            or f"recipe.{project_lower}." in qualified_name
        ):
            scoped.append(node)
    return scoped


def build_knowledge_mount(
    graph_path: str | Path,
    knowledge_root: str | Path,
    framework: str | None = None,
    project: str | None = None,
    knowledge_profile: str | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the mounted knowledge artifact."""

    resolved_source_root = resolve_source_root(
        graph_path,
        framework=framework,
        project=project,
        source_root=source_root,
    )
    graph = parse_graph(graph_path, source_root=resolved_source_root)
    knowledge_items = parse_knowledge_root(
        knowledge_root,
        framework=framework,
        project=project,
        profile=knowledge_profile,
    )
    if framework:
        knowledge_items = [
            item for item in knowledge_items
            if not item.get("framework") or item.get("framework") == framework
        ]
    if project:
        knowledge_items = [
            item for item in knowledge_items
            if item.get("project") == project
        ]

    mounts, unmounted = exact_mount_per_function(graph, knowledge_items, project=project)
    implemented_steps = [
        "parse_graph",
        "attach_node_source_code",
        "parse_knowledge",
    ]
    implemented_steps.append("per_function_exact_mount")

    return {
        "$schema": SCHEMA,
        "meta": {
            "base_graph": str(graph_path),
            "knowledge_root": str(knowledge_root),
            "knowledge_profile": knowledge_profile,
            "framework_filter": framework,
            "project_filter": project,
            "source_root": str(resolved_source_root) if resolved_source_root else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "implemented_steps": implemented_steps,
        },
        "stats": {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
            "num_normalized_edges": len(graph.edges_by_node_key),
            "num_knowledge_items": len(knowledge_items),
            "num_per_function_items": sum(1 for item in knowledge_items if item.get("is_per_function")),
            "num_function_scoped_items": sum(1 for item in knowledge_items if item.get("source_function")),
            "num_mounts": len(mounts),
            "num_unmounted_function_scoped_items": len(unmounted),
            "num_nodes_with_source_code": sum(1 for node in graph.nodes if node.get("source_code")),
        },
        "nodes": graph.nodes,
        "edges": graph.edges_by_node_key,
        "knowledge_items": knowledge_items,
        "mounts": mounts,
        "unmounted": unmounted,
    }


def default_output_path(graph_path: Path) -> Path:
    """Return the default output path next to a dep_graph.json."""

    return graph_path.with_name("dep_graph.with_knowledge.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, help="Path to dep_graph.json")
    parser.add_argument("--knowledge-root", type=Path, required=True, help="Path to practice_knowledge root")
    parser.add_argument(
        "--knowledge-profile",
        help=(
            "Practice knowledge profile to mount. Pass this explicitly for "
            "profile-scoped procedural memory; omitted value preserves the "
            "legacy combined/default lookup."
        ),
    )
    parser.add_argument("--framework", help="Framework name for default graph path lookup")
    parser.add_argument("--example", help="Example/project name for default graph path lookup and mount filtering")
    parser.add_argument("--project", help="Project filter for practice_knowledge; defaults to --example")
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Root directory for function source files. Defaults to the framework/example "
            "code corpus when it can be inferred."
        ),
    )
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    if args.graph_path:
        graph_file = args.graph_path
    elif args.framework and args.example:
        graph_file = dep_graph_path(args.framework, args.example)
    else:
        parser.error("Either --graph-path or both --framework and --example are required")

    project = args.project or args.example
    output = args.output or default_output_path(graph_file)
    artifact = build_knowledge_mount(
        graph_file,
        args.knowledge_root,
        framework=args.framework,
        project=project,
        knowledge_profile=args.knowledge_profile,
        source_root=args.source_root,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(json.dumps(artifact["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
