#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${PRAXIS_STAGE_RETRIES:=0}"

for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    echo
    echo ">>> Stage 2 input filter: ${FRAMEWORK}/${EXAMPLE}"
    mapfile -t FUNCTIONS < <(
      stage1_ready_functions \
        "$FRAMEWORK" \
        "$EXAMPLE" \
        "${PRAXIS_COVERAGE_THRESHOLD:-0.5}"
    )

    if ((${#FUNCTIONS[@]} == 0)); then
      echo "No Stage 1 functions met coverage threshold for ${FRAMEWORK}/${EXAMPLE}; skipping Stage 2."
      continue
    fi

    DISTILLED=0
    for FUNCTION in "${FUNCTIONS[@]}"; do
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
        continue
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
        continue
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
        continue
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
        continue
      fi
      DISTILLED=$((DISTILLED + 1))
    done

    if ((DISTILLED == 0)); then
      echo "No Stage 2 functions distilled for ${FRAMEWORK}/${EXAMPLE}; skipping consolidation."
      continue
    fi

    uv run python cli.py memory procedural consolidate-structured \
      --framework "$FRAMEWORK" \
      --example "$EXAMPLE" \
      --profile "$PROFILE"
  done
done
