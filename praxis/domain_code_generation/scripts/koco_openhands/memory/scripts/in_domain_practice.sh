#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

rm -f "$STAGE1_FAILURES_FILE"
STAGE1_CONCURRENCY="$(praxis_concurrency_value PRAXIS_STAGE1_CONCURRENCY "$PRAXIS_STAGE1_CONCURRENCY")"

run_stage1_example() {
  local FRAMEWORK="$1"
  local EXAMPLE="$2"
  local rc

    if [[ "${PRAXIS_FORCE:-0}" == "1" ]]; then
      echo
      echo ">>> Stage 1 clean: ${FRAMEWORK}/${EXAMPLE}"
      stage1_clean_example "$FRAMEWORK" "$EXAMPLE"
    fi

    if run_praxis_step_if_needed \
      "init" \
      "observed-memory init" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "" \
      uv run python cli.py memory init \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_OBSERVED_MAX_ITERATIONS:-100}"; then
      :
    else
      rc=$?
      record_stage1_failure "$FRAMEWORK" "$EXAMPLE" "observed-memory init" "$rc"
      [[ "${PRAXIS_STRICT:-0}" == "1" ]] && return "$rc"
      return 0
    fi

    if run_praxis_step_if_needed \
      "select" \
      "observed-memory select" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "" \
      uv run python cli.py memory select \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --strategy business_logic,indegree,outdegree \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_SELECT_MAX_ITERATIONS:-100}"; then
      :
    else
      rc=$?
      record_stage1_failure "$FRAMEWORK" "$EXAMPLE" "observed-memory select" "$rc"
      [[ "${PRAXIS_STRICT:-0}" == "1" ]] && return "$rc"
      return 0
    fi

    if run_praxis_step_if_needed \
      "generate" \
      "observed-memory generate" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "" \
      uv run python cli.py memory generate \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_GENERATE_MAX_ITERATIONS:-100}" \
      --terminal-max-iterations "$(stage1_generate_terminal_iterations)" \
      --concurrency "${PRAXIS_STAGE1_FUNCTION_CONCURRENCY:-1}"; then
      :
    else
      rc=$?
      record_stage1_failure "$FRAMEWORK" "$EXAMPLE" "observed-memory generate" "$rc"
      [[ "${PRAXIS_STRICT:-0}" == "1" ]] && return "$rc"
      return 0
    fi

    if run_praxis_step_if_needed \
      "coverage" \
      "observed-memory coverage" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "" \
      uv run python cli.py memory coverage \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --timeout "${PRAXIS_COVERAGE_TIMEOUT:-120}" \
      --concurrency "${PRAXIS_STAGE1_COVERAGE_CONCURRENCY:-1}" \
      --resume \
      --allow-empty \
      --allow-execution-failures; then
      :
    else
      rc=$?
      record_stage1_failure "$FRAMEWORK" "$EXAMPLE" "observed-memory coverage" "$rc"
      [[ "${PRAXIS_STRICT:-0}" == "1" ]] && return "$rc"
      return 0
    fi

    if run_praxis_step_if_needed \
      "feedback" \
      "observed-memory feedback" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "" \
      uv run python cli.py memory feedback \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_FEEDBACK_MAX_ITERATIONS:-120}" \
      --terminal-max-iterations "$(stage1_feedback_terminal_iterations)" \
      --coverage-threshold "${PRAXIS_COVERAGE_THRESHOLD:-0.8}" \
      --max-retries "${PRAXIS_FEEDBACK_RETRIES:-1}" \
      --terminal-max-retries "$(stage1_feedback_terminal_retries)" \
      --concurrency "${PRAXIS_STAGE1_FUNCTION_CONCURRENCY:-1}"; then
      :
    else
      rc=$?
      record_stage1_failure "$FRAMEWORK" "$EXAMPLE" "observed-memory feedback" "$rc"
      [[ "${PRAXIS_STRICT:-0}" == "1" ]] && return "$rc"
      return 0
    fi

    if [[ "${PRAXIS_STAGE1_EXAMPLE_SUMMARY:-0}" == "1" ]]; then
      summarize_stage1_example \
        "$FRAMEWORK" \
        "$EXAMPLE" \
        "${PRAXIS_COVERAGE_THRESHOLD:-0.8}"
    fi

  return 0
}

STAGE1_RC=0
for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    if ! praxis_wait_for_job_slot "$STAGE1_CONCURRENCY"; then
      STAGE1_RC=1
    fi
    run_stage1_example "$FRAMEWORK" "$EXAMPLE" &
  done
done
if ! praxis_wait_for_all_jobs; then
  STAGE1_RC=1
fi

summarize_stage1_run "${PRAXIS_COVERAGE_THRESHOLD:-0.8}"

if [[ -s "$STAGE1_FAILURES_FILE" ]]; then
  echo
  echo "ERROR: Stage 1 completed with unresolved example failure(s)." >&2
  exit 1
fi

exit "$STAGE1_RC"
