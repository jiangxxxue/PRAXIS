#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    run_praxis_step \
      "observed-memory init" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "If the agent is STUCK or hits max iterations, increase: export PRAXIS_OBSERVED_MAX_ITERATIONS=150" \
      uv run python cli.py memory init \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_OBSERVED_MAX_ITERATIONS:-100}"

    run_praxis_step \
      "observed-memory select" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "If selection fails, increase: export PRAXIS_SELECT_MAX_ITERATIONS=150" \
      uv run python cli.py memory select \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --strategy llm \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_SELECT_MAX_ITERATIONS:-100}"

    run_praxis_step \
      "observed-memory generate" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "If test-input generation fails, increase: export PRAXIS_GENERATE_MAX_ITERATIONS=150" \
      uv run python cli.py memory generate \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_GENERATE_MAX_ITERATIONS:-100}" \
      --concurrency "${PRAXIS_CONCURRENCY:-1}"

    run_praxis_step \
      "observed-memory coverage" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "If coverage fails, inspect the generated *_test_input.py file and rerun after fixing the input." \
      uv run python cli.py memory coverage \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE"

    run_praxis_step \
      "observed-memory feedback" \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "If feedback fails because the repair agent stops early, increase PRAXIS_FEEDBACK_MAX_ITERATIONS or PRAXIS_FEEDBACK_RETRIES, then rerun." \
      uv run python cli.py memory feedback \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --max-iterations "${PRAXIS_FEEDBACK_MAX_ITERATIONS:-100}" \
      --coverage-threshold "${PRAXIS_COVERAGE_THRESHOLD:-0.5}" \
      --max-retries "${PRAXIS_FEEDBACK_RETRIES:-3}" \
      --concurrency "${PRAXIS_CONCURRENCY:-1}"

    summarize_stage1_example \
      "$FRAMEWORK" \
      "$EXAMPLE" \
      "${PRAXIS_COVERAGE_THRESHOLD:-0.5}"
  done
done
