#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE4_CONCURRENCY="$(
  praxis_concurrency_value \
    PRAXIS_STAGE4_CONCURRENCY \
    "${PRAXIS_STAGE4_CONCURRENCY:-$PRAXIS_CONCURRENCY}"
)"
STAGE4_RETRIES="$(
  praxis_concurrency_value \
    PRAXIS_STAGE4_RETRIES \
    "${PRAXIS_STAGE4_RETRIES:-1}"
)"
FAILURES_FILE="${PROFILE_GRAPH_ROOT}/incremental_stage4_failures.tsv"
rm -f "$FAILURES_FILE"

record_failure() {
  local framework="$1"
  local example="$2"
  local reason="$3"
  praxis_append_line_locked \
    "$FAILURES_FILE" \
    "$(printf '%s\t%s\t%s' "$framework" "$example" "$reason")"
}

run_example() {
  local framework="$1"
  local example="$2"
  local graph="${PROFILE_GRAPH_ROOT}/${framework}/${example}/dep_graph.with_knowledge.optimized.json"
  local attempt=1
  local max_attempts=$((STAGE4_RETRIES + 1))
  local rc=0

  if [[ ! -s "$graph" ]]; then
    record_failure "$framework" "$example" "optimized graph missing"
    return 1
  fi

  while (( attempt <= max_attempts )); do
    echo
    echo ">>> Incremental Stage 4 ${framework}/${example} (attempt ${attempt}/${max_attempts})"
    set +e
    uv run python cli.py run \
      --framework "$framework" \
      --test-example "$example" \
      --model "$MODEL" \
      --graph-knowledge-profile "$PROFILE" \
      --graph-knowledge-artifact optimized \
      --graph-knowledge-min-confidence "${PRAXIS_CONFIDENCE_THRESHOLD:-0.7}" \
      --run-name "$PROFILE" \
      --force \
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
    echo "WARN Incremental Stage 4 ${framework}/${example}: retrying once." >&2
    attempt=$((attempt + 1))
  done

  record_failure "$framework" "$example" "inference or evaluation failed after ${max_attempts} attempt(s)"
  return "$rc"
}

STAGE4_RC=0
for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    if ! praxis_wait_for_job_slot "$STAGE4_CONCURRENCY"; then
      STAGE4_RC=1
    fi
    (
      set +e
      run_example "$FRAMEWORK" "$EXAMPLE"
      rc=$?
      set -e
      if (( rc != 0 )); then
        echo "WARN Incremental Stage 4 ${FRAMEWORK}/${EXAMPLE} failed." >&2
      fi
      exit "$rc"
    ) &
  done
done
if ! praxis_wait_for_all_jobs; then
  STAGE4_RC=1
fi

if [[ -s "$FAILURES_FILE" ]]; then
  echo
  echo "Incremental Stage 4 completed with failed example(s):" >&2
  cat "$FAILURES_FILE" >&2
  exit 1
fi

exit "$STAGE4_RC"
