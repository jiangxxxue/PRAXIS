#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${PRAXIS_STAGE_RETRIES:=0}"
STAGE2_CONCURRENCY="$(praxis_concurrency_value PRAXIS_STAGE2_CONCURRENCY "$PRAXIS_STAGE2_CONCURRENCY")"
STAGE2_WORK_DIR="memory/derived/procedural_knowledge/${PROFILE}/_stage2_parallel"
STAGE2_WORK_ITEMS="${STAGE2_WORK_DIR}/work_items.tsv"
STAGE2_DONE_ITEMS="${STAGE2_WORK_DIR}/done_items.tsv"
STAGE2_FAILED_ITEMS="${STAGE2_WORK_DIR}/failed_items.tsv"

rm -rf "$STAGE2_WORK_DIR"
mkdir -p "$STAGE2_WORK_DIR"

record_stage2_done() {
  local framework="$1"
  local example="$2"
  local function="$3"
  praxis_append_line_locked "$STAGE2_DONE_ITEMS" "$(printf '%s\t%s\t%s' "$framework" "$example" "$function")"
}

record_stage2_failed() {
  local framework="$1"
  local example="$2"
  local function="$3"
  local reason="$4"
  praxis_append_line_locked "$STAGE2_FAILED_ITEMS" "$(printf '%s\t%s\t%s\t%s' "$framework" "$example" "$function" "$reason")"
}

run_stage2_function() {
  local FRAMEWORK="$1"
  local EXAMPLE="$2"
  local FUNCTION="$3"
  local PER_FUNCTION_DIR="memory/derived/procedural_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}/per_function"
  local FUNCTION_FILE="${PER_FUNCTION_DIR}/${FUNCTION}.jsonl"

  if [[ -s "$FUNCTION_FILE" ]]; then
    echo "skip ${FUNCTION}: per-function knowledge already exists"
    record_stage2_done "$FRAMEWORK" "$EXAMPLE" "$FUNCTION"
    return 0
  fi

  if ! run_praxis_step \
    "procedural init" \
    "$FRAMEWORK" \
    "$EXAMPLE/$FUNCTION" \
    "This function is not suitable for procedural oracle generation; it will be skipped for Stage 2." \
    uv run python cli.py memory procedural init \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --function "$FUNCTION" \
    --profile "$PROFILE"; then
    echo "SKIP Stage 2 function ${FRAMEWORK}/${EXAMPLE}/${FUNCTION}: init failed"
    record_stage2_failed "$FRAMEWORK" "$EXAMPLE" "$FUNCTION" "init failed"
    return 0
  fi

  if ! run_praxis_step \
    "procedural grade-smoke" \
    "$FRAMEWORK" \
    "$EXAMPLE/$FUNCTION" \
    "The GT oracle is not stable enough for procedural practice; this function will be skipped." \
    uv run python cli.py memory procedural grade-smoke \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --function "$FUNCTION" \
    --profile "$PROFILE"; then
    echo "SKIP Stage 2 function ${FRAMEWORK}/${EXAMPLE}/${FUNCTION}: grade-smoke failed"
    record_stage2_failed "$FRAMEWORK" "$EXAMPLE" "$FUNCTION" "grade-smoke failed"
    return 0
  fi

  if ! run_praxis_step \
    "procedural practice" \
    "$FRAMEWORK" \
    "$EXAMPLE/$FUNCTION" \
    "The practice agent could not produce a usable attempt; this function will be skipped." \
    uv run python cli.py memory procedural practice \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --function "$FUNCTION" \
    --profile "$PROFILE" \
    --K "${PRAXIS_PRACTICE_K:-8}" \
    --max-iterations "${PRAXIS_PRACTICE_MAX_ITERATIONS:-100}" \
    --model "$MODEL"; then
    echo "SKIP Stage 2 function ${FRAMEWORK}/${EXAMPLE}/${FUNCTION}: practice failed"
    record_stage2_failed "$FRAMEWORK" "$EXAMPLE" "$FUNCTION" "practice failed"
    return 0
  fi

  if ! run_praxis_step \
    "procedural distill" \
    "$FRAMEWORK" \
    "$EXAMPLE/$FUNCTION" \
    "Trace distillation failed; this function will be skipped." \
    uv run python cli.py memory procedural distill-structured \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --function "$FUNCTION" \
    --profile "$PROFILE" \
    --model "$MODEL"; then
    echo "SKIP Stage 2 function ${FRAMEWORK}/${EXAMPLE}/${FUNCTION}: distill failed"
    record_stage2_failed "$FRAMEWORK" "$EXAMPLE" "$FUNCTION" "distill failed"
    return 0
  fi

  record_stage2_done "$FRAMEWORK" "$EXAMPLE" "$FUNCTION"
}

for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    EXAMPLE_DIR="memory/derived/procedural_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}"
    PER_FUNCTION_DIR="memory/derived/procedural_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}/per_function"
    PRACTICE_FILE="${EXAMPLE_DIR}/practice_knowledge.jsonl"

    if [[ "${PRAXIS_STAGE2_CLEAN:-0}" == "1" ]]; then
      rm -rf "$PER_FUNCTION_DIR"
      rm -f "$PRACTICE_FILE"
    elif [[ -s "$PRACTICE_FILE" ]]; then
      echo
      echo ">>> Stage 2 input filter: ${FRAMEWORK}/${EXAMPLE} (resume: practice_knowledge already exists)"
      continue
    fi

    echo
    echo ">>> Stage 2 input filter: ${FRAMEWORK}/${EXAMPLE}"
    mapfile -t FUNCTIONS < <(
      stage2_candidate_functions \
        "$FRAMEWORK" \
        "$EXAMPLE"
    )

    if ((${#FUNCTIONS[@]} == 0)); then
      echo "SKIP Stage 2 example ${FRAMEWORK}/${EXAMPLE}: no candidates have requirement and test input artifacts." >&2
      continue
    fi

    for FUNCTION in "${FUNCTIONS[@]}"; do
      printf '%s\t%s\t%s\n' "$FRAMEWORK" "$EXAMPLE" "$FUNCTION" >> "$STAGE2_WORK_ITEMS"
    done
  done
done

if [[ -s "$STAGE2_WORK_ITEMS" ]]; then
  STAGE2_RC=0
  while IFS=$'\t' read -r FRAMEWORK EXAMPLE FUNCTION; do
    if ! praxis_wait_for_job_slot "$STAGE2_CONCURRENCY"; then
      STAGE2_RC=1
    fi
    run_stage2_function "$FRAMEWORK" "$EXAMPLE" "$FUNCTION" &
  done < "$STAGE2_WORK_ITEMS"
  if ! praxis_wait_for_all_jobs; then
    STAGE2_RC=1
  fi
  if (( STAGE2_RC != 0 )); then
    echo "ERROR: Stage 2 had unexpected worker failure(s)." >&2
    exit "$STAGE2_RC"
  fi
fi

for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    EXAMPLE_DIR="memory/derived/procedural_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}"
    PRACTICE_FILE="${EXAMPLE_DIR}/practice_knowledge.jsonl"
    if [[ "${PRAXIS_STAGE2_CLEAN:-0}" != "1" && -s "$PRACTICE_FILE" ]]; then
      continue
    fi
    if ! awk -F '\t' -v fw="$FRAMEWORK" -v ex="$EXAMPLE" '$1 == fw && $2 == ex { found = 1 } END { exit(found ? 0 : 1) }' "$STAGE2_DONE_ITEMS" 2>/dev/null; then
      echo "SKIP Stage 2 example ${FRAMEWORK}/${EXAMPLE}: no functions distilled." >&2
      continue
    fi
    uv run python cli.py memory procedural consolidate-structured \
      --framework "$FRAMEWORK" \
      --example "$EXAMPLE" \
      --profile "$PROFILE"

    if [[ ! -s "$PRACTICE_FILE" ]]; then
      echo "ERROR: Stage 2 produced no knowledge entries for ${FRAMEWORK}/${EXAMPLE}: ${PRACTICE_FILE}" >&2
      exit 1
    fi
  done
done
