#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${PRAXIS_INCREMENTAL_MODE:?Set PRAXIS_INCREMENTAL_MODE to b or c}"
: "${PRAXIS_INCREMENTAL_SOURCE_PROFILE:?Set PRAXIS_INCREMENTAL_SOURCE_PROFILE}"

case "$PRAXIS_INCREMENTAL_MODE" in
  b)
    SOURCE_ARTIFACT="${PRAXIS_INCREMENTAL_SOURCE_ARTIFACT:-mounted}"
    ;;
  c)
    SOURCE_ARTIFACT="${PRAXIS_INCREMENTAL_SOURCE_ARTIFACT:-optimized}"
    ;;
  *)
    echo "ERROR: PRAXIS_INCREMENTAL_MODE must be b or c." >&2
    exit 1
    ;;
esac

if [[ "$SOURCE_ARTIFACT" == "mounted" ]]; then
  SOURCE_FILENAME="dep_graph.with_knowledge.json"
elif [[ "$SOURCE_ARTIFACT" == "optimized" ]]; then
  SOURCE_FILENAME="dep_graph.with_knowledge.optimized.json"
else
  echo "ERROR: PRAXIS_INCREMENTAL_SOURCE_ARTIFACT must be mounted or optimized." >&2
  exit 1
fi

INCREMENTAL_CONCURRENCY="$(
  praxis_concurrency_value \
    PRAXIS_INCREMENTAL_CONCURRENCY \
    "${PRAXIS_INCREMENTAL_CONCURRENCY:-$PRAXIS_CONCURRENCY}"
)"
FAILURES_FILE="${PROFILE_GRAPH_ROOT}/incremental_${PRAXIS_INCREMENTAL_MODE}_failures.tsv"
rm -f "$FAILURES_FILE"

record_incremental_failure() {
  local framework="$1"
  local example="$2"
  local reason="$3"
  praxis_append_line_locked \
    "$FAILURES_FILE" \
    "$(printf '%s\t%s\t%s' "$framework" "$example" "$reason")"
}

run_incremental_example() {
  local framework="$1"
  local example="$2"
  local source_graph
  local output_dir
  local output_graph
  local cache_path

  source_graph="memory/derived/graph_knowledge/${PRAXIS_INCREMENTAL_SOURCE_PROFILE}/${framework}/${example}/${SOURCE_FILENAME}"
  output_dir="${PROFILE_GRAPH_ROOT}/${framework}/${example}"
  output_graph="${output_dir}/dep_graph.with_knowledge.optimized.json"
  cache_path="${output_dir}/dep_graph.with_knowledge.incremental_${PRAXIS_INCREMENTAL_MODE}.llm_cache.json"

  if [[ ! -s "$source_graph" ]]; then
    record_incremental_failure "$framework" "$example" "source graph missing: ${source_graph}"
    return 1
  fi
  if [[ "${PRAXIS_INCREMENTAL_FORCE:-0}" != "1" && -s "$output_graph" ]]; then
    echo "SKIP Incremental ${PRAXIS_INCREMENTAL_MODE} ${framework}/${example}: output exists."
    return 0
  fi

  mkdir -p "$output_dir"
  echo
  echo ">>> Incremental ${PRAXIS_INCREMENTAL_MODE}: ${framework}/${example}"
  set +e
  uv run python -m memory.incremental_optimize_graph_knowledge \
    --graph-path "$source_graph" \
    --output "$output_graph" \
    --mode "$PRAXIS_INCREMENTAL_MODE" \
    --model "$MODEL" \
    --base-url "${OPTIMIZE_KNOWLEDGE_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}" \
    --api-key "${OPENROUTER_API_KEY:-}" \
    --cache-path "$cache_path" \
    --min-confidence "${PRAXIS_CONFIDENCE_THRESHOLD:-0.7}" \
    --candidate-source-layer "${PRAXIS_INCREMENTAL_CANDIDATE_SOURCE_LAYER:-$PRAXIS_INCREMENTAL_SOURCE_PROFILE}" \
    --conflict-confidence-margin "${PRAXIS_INCREMENTAL_CONFLICT_MARGIN:-0.15}" \
    --max-targets-per-knowledge "${PRAXIS_INCREMENTAL_MAX_TARGETS:-3}" \
    --max-candidates-per-target "${PRAXIS_INCREMENTAL_MAX_CANDIDATES_PER_TARGET:-3}"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    record_incremental_failure "$framework" "$example" "optimizer failed rc=${rc}"
    return "$rc"
  fi
}

INCREMENTAL_RC=0
for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    if ! praxis_wait_for_job_slot "$INCREMENTAL_CONCURRENCY"; then
      INCREMENTAL_RC=1
    fi
    (
      set +e
      run_incremental_example "$FRAMEWORK" "$EXAMPLE"
      rc=$?
      set -e
      if (( rc != 0 )); then
        echo "WARN Incremental ${PRAXIS_INCREMENTAL_MODE} ${FRAMEWORK}/${EXAMPLE} failed." >&2
      fi
      exit "$rc"
    ) &
  done
done
if ! praxis_wait_for_all_jobs; then
  INCREMENTAL_RC=1
fi

if [[ -s "$FAILURES_FILE" ]]; then
  echo
  echo "Incremental ${PRAXIS_INCREMENTAL_MODE} completed with failed example(s):" >&2
  cat "$FAILURES_FILE" >&2
  exit 1
fi

exit "$INCREMENTAL_RC"
