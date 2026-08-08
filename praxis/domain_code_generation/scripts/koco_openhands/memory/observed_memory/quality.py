"""Shared quality gates for Stage 1 coverage artifacts."""

from __future__ import annotations


MIN_NORMAL_CASES = 5
MIN_NORMAL_EDGE_SUCCESS_RATIO = 0.8


def stage2_case_gate(coverage: dict) -> dict | None:
    recorded = coverage.get("stage2_case_gate")
    if isinstance(recorded, dict) and isinstance(recorded.get("passed"), bool):
        return recorded

    results = coverage.get("results")
    if not isinstance(results, list):
        return None

    normal = [
        record
        for record in results
        if isinstance(record, dict) and record.get("category") == "normal"
    ]
    strong = [
        record
        for record in results
        if isinstance(record, dict)
        and record.get("category") in {"normal", "edge"}
    ]
    successes = sum(1 for record in strong if record.get("success"))
    ratio = successes / len(strong) if strong else 0.0
    return {
        "normal_cases": len(normal),
        "normal_edge_cases": len(strong),
        "normal_edge_successes": successes,
        "normal_edge_success_ratio": round(ratio, 4),
        "passed": (
            len(normal) >= MIN_NORMAL_CASES
            and ratio >= MIN_NORMAL_EDGE_SUCCESS_RATIO
        ),
    }


def coverage_has_execution_failure(coverage: dict) -> bool:
    if coverage.get("is_execution_failure"):
        return True
    gate = stage2_case_gate(coverage)
    return gate is not None and not gate["passed"]


def coverage_ready(coverage: dict, threshold: float) -> bool:
    if coverage_has_execution_failure(coverage):
        return False
    return float(coverage.get("line_coverage") or 0.0) >= threshold


def coverage_quality_score(coverage: dict, threshold: float) -> tuple:
    """Rank coverage artifacts by proximity to the complete Stage 2 gate."""
    gate = stage2_case_gate(coverage) or {}
    line_coverage = float(coverage.get("line_coverage") or 0.0)
    normal_cases = int(gate.get("normal_cases") or 0)
    success_ratio = float(gate.get("normal_edge_success_ratio") or 0.0)
    raw_execution_ok = not bool(coverage.get("is_execution_failure"))
    progress = (
        min(line_coverage / threshold, 1.0) if threshold > 0 else 1.0
    )
    progress += min(normal_cases / MIN_NORMAL_CASES, 1.0)
    progress += min(
        success_ratio / MIN_NORMAL_EDGE_SUCCESS_RATIO,
        1.0,
    )
    return (
        coverage_ready(coverage, threshold),
        raw_execution_ok,
        bool(gate.get("passed")),
        round(progress, 6),
        line_coverage,
        normal_cases,
        success_ratio,
    )
