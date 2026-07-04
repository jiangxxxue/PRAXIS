#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    uv run python -m memory.knowledge_mount \
      --graph-path "memory/derived/graph_knowledge/${FRAMEWORK}/${EXAMPLE}/dep_graph.json" \
      --knowledge-root memory/derived/procedural_knowledge \
      --knowledge-profile "$PROFILE" \
      --framework "$FRAMEWORK" \
      --example "$EXAMPLE" \
      --output "memory/derived/graph_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.json"

    uv run python -m memory.optimize_graph_knowledge \
      --graph-path "memory/derived/graph_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.json" \
      --output "memory/derived/graph_knowledge/${PROFILE}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.optimized.json" \
      --base-url "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}" \
      --api-key "$OPENROUTER_API_KEY" \
      --model "$MODEL" \
      --max-propagation-decisions "${PRAXIS_MAX_PROPAGATION_DECISIONS:-1000}"
  done
done
