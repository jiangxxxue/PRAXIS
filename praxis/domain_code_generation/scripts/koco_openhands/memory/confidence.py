"""Shared confidence semantics for practice knowledge."""

from __future__ import annotations

import math
import os
from typing import Any, Iterable


DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_ONLINE_FEEDBACK_BETA = 0.8
DEFAULT_CONFLICT_CONFIDENCE_MARGIN = 0.15


def configured_confidence_threshold(value: float | None = None) -> float:
    if value is not None:
        return _clip_unit(value)
    raw = os.environ.get("PRAXIS_CONFIDENCE_THRESHOLD", "")
    if not raw:
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        return _clip_unit(float(raw))
    except ValueError:
        return DEFAULT_CONFIDENCE_THRESHOLD


def confidence_score(value: Any, default: float = 0.0) -> float:
    """Read a score from raw, canonical, and legacy confidence shapes."""

    if _is_number(value):
        return _clip_unit(float(value))
    if isinstance(value, dict):
        score = value.get("score")
        if _is_number(score):
            return _clip_unit(float(score))
        nested = value.get("confidence")
        if nested is not None:
            return confidence_score(nested, default=default)
        source_scores = value.get("source_scores")
        if isinstance(source_scores, list):
            return noisy_or(
                confidence_score(item, default=0.0)
                for item in source_scores
            )
    if isinstance(value, list):
        scores = []
        for item in value:
            if isinstance(item, dict) and "confidence" in item:
                scores.append(confidence_score(item["confidence"], default=0.0))
            else:
                scores.append(confidence_score(item, default=0.0))
        if scores:
            return noisy_or(scores)
    return _clip_unit(default)


def knowledge_confidence_score(item: dict[str, Any], default: float = 0.0) -> float:
    return confidence_score(item.get("confidence"), default=default)


def noisy_or(scores: Iterable[float]) -> float:
    remaining = 1.0
    found = False
    for score in scores:
        found = True
        remaining *= 1.0 - _clip_unit(score)
    if not found:
        return 0.0
    return round(1.0 - remaining, 4)


def aggregate_knowledge_confidence(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent duplicate evidence while avoiding propagated copies."""

    source_scores: dict[str, float] = {}
    for item in items:
        source_id = str(item.get("source_id") or item.get("id") or "")
        if not source_id:
            continue
        score = knowledge_confidence_score(item)
        source_scores[source_id] = max(source_scores.get(source_id, 0.0), score)

    ordered_sources = [
        {"id": source_id, "score": score}
        for source_id, score in sorted(source_scores.items())
    ]
    score = noisy_or(source_scores.values())
    aggregation = "identity" if len(source_scores) <= 1 else "noisy_or"
    return {
        "score": score,
        "aggregation": aggregation,
        "source_scores": ordered_sources,
    }


def update_confidence_from_outcome(
    confidence: Any,
    *,
    succeeded: bool,
    beta: float = DEFAULT_ONLINE_FEEDBACK_BETA,
    event_id: str = "",
) -> dict[str, Any]:
    """Apply online success reinforcement or failure decay."""

    beta = _clip_unit(beta)
    current = confidence_score(confidence)
    result = dict(confidence) if isinstance(confidence, dict) else {}
    feedback = result.get("online_feedback")
    feedback = dict(feedback) if isinstance(feedback, dict) else {}
    applied_event_ids = [
        str(value)
        for value in feedback.get("applied_event_ids") or []
        if str(value)
    ]
    if event_id and event_id in applied_event_ids:
        return result

    updated = (
        1.0 - beta * (1.0 - current)
        if succeeded
        else beta * current
    )
    feedback["successes"] = int(feedback.get("successes") or 0)
    feedback["failures"] = int(feedback.get("failures") or 0)
    feedback["successes" if succeeded else "failures"] += 1
    feedback["beta"] = beta
    feedback["last_outcome"] = "success" if succeeded else "failure"
    if event_id:
        applied_event_ids.append(event_id)
        feedback["applied_event_ids"] = applied_event_ids
    result["score"] = round(_clip_unit(updated), 4)
    result["online_feedback"] = feedback
    return result


def feedback_event_applied(confidence: Any, event_id: str) -> bool:
    if not event_id or not isinstance(confidence, dict):
        return False
    feedback = confidence.get("online_feedback")
    if not isinstance(feedback, dict):
        return False
    return event_id in {
        str(value)
        for value in feedback.get("applied_event_ids") or []
        if str(value)
    }


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _clip_unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
