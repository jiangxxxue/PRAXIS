#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE3_CONCURRENCY="$(praxis_concurrency_value PRAXIS_STAGE3_CONCURRENCY "$PRAXIS_STAGE3_CONCURRENCY")"
STAGE3_FAILURES_FILE="${PROFILE_GRAPH_ROOT}/stage3_failures.tsv"
rm -f "$STAGE3_FAILURES_FILE"

record_stage3_failure() {
  local framework="$1"
  local example="$2"
  local reason="$3"
  praxis_append_line_locked "$STAGE3_FAILURES_FILE" "$(printf '%s\t%s\t%s' "$framework" "$example" "$reason")"
}

run_stage3_example() {
  local FRAMEWORK="$1"
  local EXAMPLE="$2"
  local PRACTICE_FILE="memory/derived/procedural_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}/practice_knowledge.jsonl"
  local OPTIMIZED_GRAPH="${PROFILE_GRAPH_ROOT}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.optimized.json"

  if [[ ! -s "$PRACTICE_FILE" ]]; then
    echo
    echo "SKIP Stage 3 example ${FRAMEWORK}/${EXAMPLE}: no Stage 2 practice knowledge."
    return 0
  fi
  if [[ "${PRAXIS_STAGE3_FORCE:-0}" != "1" && -s "$OPTIMIZED_GRAPH" ]]; then
    echo
    echo "SKIP Stage 3 example ${FRAMEWORK}/${EXAMPLE}: optimized graph already exists."
    return 0
  fi

  if ! uv run python -m memory.knowledge_mount \
    --graph-path "${STAGE1_GRAPH_ROOT}/${FRAMEWORK}/${EXAMPLE}/dep_graph.json" \
    --knowledge-root memory/derived/procedural_knowledge \
    --knowledge-profile "$PROFILE" \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --require-knowledge \
    --output "${PROFILE_GRAPH_ROOT}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.json"; then
    record_stage3_failure "$FRAMEWORK" "$EXAMPLE" "knowledge_mount failed"
    return 1
  fi

  if ! uv run python -m memory.optimize_graph_knowledge \
    --graph-path "${PROFILE_GRAPH_ROOT}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.json" \
    --output "$OPTIMIZED_GRAPH" \
    --base-url "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}" \
    --api-key "${OPENROUTER_API_KEY:-}" \
    --model "$MODEL" \
    --min-confidence "${PRAXIS_CONFIDENCE_THRESHOLD:-0.6}" \
    --max-propagation-decisions "${PRAXIS_MAX_PROPAGATION_DECISIONS:-1000}"; then
    record_stage3_failure "$FRAMEWORK" "$EXAMPLE" "optimize_graph_knowledge failed"
    return 1
  fi
}

STAGE3_RC=0
for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    if ! praxis_wait_for_job_slot "$STAGE3_CONCURRENCY"; then
      STAGE3_RC=1
    fi
    run_stage3_example "$FRAMEWORK" "$EXAMPLE" &
  done
done
if ! praxis_wait_for_all_jobs; then
  STAGE3_RC=1
fi

if [[ -s "$STAGE3_FAILURES_FILE" ]]; then
  echo
  echo "ERROR: Stage 3 completed with failed example(s):" >&2
  cat "$STAGE3_FAILURES_FILE" >&2
  exit 1
fi

exit "$STAGE3_RC"
