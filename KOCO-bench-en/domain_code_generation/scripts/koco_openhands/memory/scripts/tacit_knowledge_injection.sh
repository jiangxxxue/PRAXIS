#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    uv run python cli.py run \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --graph-knowledge-profile "$PROFILE" \
      --graph-knowledge-artifact optimized \
      --concurrency "${PRAXIS_CONCURRENCY:-1}" \
      --max-iterations "${PRAXIS_INFER_MAX_ITERATIONS:-100}"
  done
done
