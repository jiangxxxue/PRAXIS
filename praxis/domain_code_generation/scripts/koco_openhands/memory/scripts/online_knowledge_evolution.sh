#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EVOLVED_PROFILE="${PRAXIS_EVOLVED_PROFILE:-${PROFILE}_online}"

for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    uv run python -m memory.procedural_memory.target_practice import \
      --profile "$EVOLVED_PROFILE" \
      --framework "$FRAMEWORK" \
      --source-model-dir "${MODEL##*/}" \
      --example "$EXAMPLE" \
      --require-result \
      --force

    uv run python -m memory.procedural_memory.target_practice distill \
      --profile "$EVOLVED_PROFILE" \
      --framework "$FRAMEWORK" \
      --example "$EXAMPLE" \
      --model "$MODEL" \
      --force

    PER_FUNCTION_DIR="memory/derived/procedural_knowledge/${EVOLVED_PROFILE}/${FRAMEWORK}/${EXAMPLE}/per_function"
    if ! find "$PER_FUNCTION_DIR" -maxdepth 1 -name '*.jsonl' -type f 2>/dev/null | grep -q .; then
      echo "online evolution: no passed target-practice memory for ${FRAMEWORK}/${EXAMPLE}; skipping evolved graph and re-inference."
      continue
    fi

    uv run python -m memory.procedural_memory.target_practice consolidate \
      --profile "$EVOLVED_PROFILE" \
      --framework "$FRAMEWORK" \
      --example "$EXAMPLE"

    uv run python -m memory.knowledge_mount \
      --graph-path "memory/derived/graph_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.optimized.json" \
      --knowledge-root memory/derived/procedural_knowledge \
      --knowledge-profile "$EVOLVED_PROFILE" \
      --framework "$FRAMEWORK" \
      --example "$EXAMPLE" \
      --output "memory/derived/graph_knowledge/${EVOLVED_PROFILE}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.json"

    uv run python cli.py run \
      --framework "$FRAMEWORK" \
      --test-example "$EXAMPLE" \
      --model "$MODEL" \
      --graph-knowledge-profile "$EVOLVED_PROFILE" \
      --graph-knowledge-artifact mounted \
      --concurrency "${PRAXIS_CONCURRENCY:-1}" \
      --max-iterations "${PRAXIS_INFER_MAX_ITERATIONS:-100}"
  done
done
