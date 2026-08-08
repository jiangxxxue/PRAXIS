#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FORCE_ARGS=()
if [[ "${PRAXIS_STAGE4_FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi
STAGE4_CONCURRENCY="$(praxis_concurrency_value PRAXIS_STAGE4_CONCURRENCY "$PRAXIS_STAGE4_CONCURRENCY")"
STAGE4_RETRIES="$(praxis_concurrency_value PRAXIS_STAGE4_RETRIES "$PRAXIS_STAGE4_RETRIES")"
STAGE4_FAILURES_FILE="${PROFILE_GRAPH_ROOT}/stage4_failures.tsv"
rm -f "$STAGE4_FAILURES_FILE"

record_stage4_failure() {
  local framework="$1"
  local example="$2"
  local reason="$3"
  praxis_append_line_locked "$STAGE4_FAILURES_FILE" "$(printf '%s\t%s\t%s' "$framework" "$example" "$reason")"
}

run_stage4_example() {
  local FRAMEWORK="$1"
  local EXAMPLE="$2"
  local OPTIMIZED_GRAPH="${PROFILE_GRAPH_ROOT}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.optimized.json"
  local GRAPH_ARGS=(
    --graph-knowledge-profile "$PROFILE"
    --graph-knowledge-artifact optimized
    --graph-knowledge-min-confidence "${PRAXIS_CONFIDENCE_THRESHOLD:-0.6}"
  )

  if [[ ! -s "$OPTIMIZED_GRAPH" ]]; then
    echo
    echo "SKIP Stage 4 example ${FRAMEWORK}/${EXAMPLE}: optimized graph is missing."
    return 0
  fi

  local attempt=1
  local max_attempts=$((STAGE4_RETRIES + 1))
  local rc=0
  while (( attempt <= max_attempts )); do
    echo
    echo ">>> Stage 4 example ${FRAMEWORK}/${EXAMPLE} (attempt ${attempt}/${max_attempts})"
    set +e
    uv run python cli.py run \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      "${GRAPH_ARGS[@]}" \
      --run-name "$PROFILE" \
      "${FORCE_ARGS[@]}" \
      --concurrency "${PRAXIS_STAGE4_INFER_CONCURRENCY:-1}" \
      --max-iterations "${PRAXIS_INFER_MAX_ITERATIONS:-100}"
    rc=$?
    set -e
    if (( rc == 0 )); then
      return 0
    fi
    if (( attempt >= max_attempts )); then
      break
    fi
    echo
    echo "WARN Stage 4 example ${FRAMEWORK}/${EXAMPLE}: inference or evaluation failed; retrying once with resume." >&2
    attempt=$((attempt + 1))
  done

  echo
  echo "ERROR Stage 4 example ${FRAMEWORK}/${EXAMPLE}: inference or evaluation failed after ${max_attempts} attempt(s)." >&2
  record_stage4_failure "$FRAMEWORK" "$EXAMPLE" "inference or evaluation failed after ${max_attempts} attempt(s)"
  return "$rc"
}

STAGE4_RC=0
for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    if ! praxis_wait_for_job_slot "$STAGE4_CONCURRENCY"; then
      STAGE4_RC=1
    fi
    run_stage4_example "$FRAMEWORK" "$EXAMPLE" &
  done
done
if ! praxis_wait_for_all_jobs; then
  STAGE4_RC=1
fi

if [[ -s "$STAGE4_FAILURES_FILE" ]]; then
  echo
  echo "Stage 4 completed with failed example(s):" >&2
  cat "$STAGE4_FAILURES_FILE" >&2
  exit 1
fi

exit "$STAGE4_RC"
