#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EVOLVED_PROFILE="${PRAXIS_EVOLVED_PROFILE:-${PROFILE}_online}"
EVOLVED_PROFILE="${EVOLVED_PROFILE//\//__}"
EVOLVED_PROFILE="${EVOLVED_PROFILE//[^A-Za-z0-9_.-]/_}"
SOURCE_RUN_NAME="${PRAXIS_SOURCE_RUN_NAME:-$PROFILE}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME//\//__}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME//[^A-Za-z0-9_.-]/_}"
SOURCE_MODEL_DIR="${MODEL_KEY}--${SOURCE_RUN_NAME}"
ONLINE_FAILURES_FILE="memory/derived/graph_knowledge/${EVOLVED_PROFILE}/online_failures.tsv"
ONLINE_CONCURRENCY="$(praxis_concurrency_value PRAXIS_ONLINE_CONCURRENCY "${PRAXIS_ONLINE_CONCURRENCY:-$PRAXIS_CONCURRENCY}")"
rm -f "$ONLINE_FAILURES_FILE"

record_online_failure() {
  local framework="$1"
  local example="$2"
  local reason="$3"
  praxis_append_line_locked "$ONLINE_FAILURES_FILE" "$(printf '%s\t%s\t%s' "$framework" "$example" "$reason")"
}

run_online_infer() {
  local framework="$1"
  local example="$2"
  local attempt=1
  local retries
  retries="$(praxis_concurrency_value PRAXIS_STAGE4_RETRIES "${PRAXIS_STAGE4_RETRIES:-1}")"
  local max_attempts=$((retries + 1))
  local rc=0

  while (( attempt <= max_attempts )); do
    echo
    echo ">>> Online Stage 4 example ${framework}/${example} (attempt ${attempt}/${max_attempts})"
    set +e
    uv run python cli.py run \
      --framework "$framework" \
      --test-example "$example" \
      --model "$MODEL" \
      --graph-knowledge-profile "$EVOLVED_PROFILE" \
      --graph-knowledge-artifact optimized \
      --graph-knowledge-min-confidence "${PRAXIS_CONFIDENCE_THRESHOLD:-0.6}" \
      --run-name "$EVOLVED_PROFILE" \
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
    echo
    echo "WARN Online Stage 4 example ${framework}/${example}: inference or evaluation failed; retrying once." >&2
    attempt=$((attempt + 1))
  done

  echo
  echo "ERROR Online Stage 4 example ${framework}/${example}: inference or evaluation failed after ${max_attempts} attempt(s)." >&2
  return "$rc"
}

run_online_example() {
  local FRAMEWORK="$1"
  local EXAMPLE="$2"
  local model_dir="${MODEL_KEY}--${EVOLVED_PROFILE}"
  local online_result_dir="../data/${FRAMEWORK}/openhands/${model_dir}"
  local online_output="${online_result_dir}/algorithm_methods_data_${EXAMPLE}_output.jsonl"
  local online_metrics="${online_result_dir}/algorithm_methods_data_${EXAMPLE}_result.metrics.json"
  local evolved_graph_dir="memory/derived/graph_knowledge/${EVOLVED_PROFILE}/${FRAMEWORK}/${EXAMPLE}"
  local evolved_base_graph="${evolved_graph_dir}/dep_graph.base.with_feedback.json"
  local evolved_mounted_graph="${evolved_graph_dir}/dep_graph.with_knowledge.json"
  local evolved_local_graph="${evolved_graph_dir}/dep_graph.with_knowledge.online_local.optimized.json"
  local evolved_optimized_graph="${evolved_graph_dir}/dep_graph.with_knowledge.optimized.json"

  if [[ "${PRAXIS_ONLINE_FORCE:-0}" != "1" && -s "$online_output" && -s "$online_metrics" ]]; then
    echo
    echo "SKIP Online evolution example ${FRAMEWORK}/${EXAMPLE}: evolved output and metrics already exist."
    return 0
  fi

  rm -rf "memory/derived/procedural_knowledge/_target_traces/${EVOLVED_PROFILE}/${FRAMEWORK}/${EXAMPLE}"

  uv run python -m memory.procedural_memory.target_practice import \
    --profile "$EVOLVED_PROFILE" \
    --framework "$FRAMEWORK" \
    --source-model-dir "$SOURCE_MODEL_DIR" \
    --example "$EXAMPLE" \
    --require-result \
    --force

  BASE_GRAPH="${PROFILE_GRAPH_ROOT}/${FRAMEWORK}/${EXAMPLE}/dep_graph.with_knowledge.optimized.json"
  mkdir -p "$evolved_graph_dir"
  cp "$BASE_GRAPH" "$evolved_base_graph"
  uv run python -m memory.procedural_memory.target_practice update-confidence \
    --profile "$EVOLVED_PROFILE" \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --graph-path "$evolved_base_graph" \
    --beta "${PRAXIS_ONLINE_CONFIDENCE_BETA:-0.8}"

  rm -rf "memory/derived/procedural_knowledge/${EVOLVED_PROFILE}/${FRAMEWORK}/${EXAMPLE}/per_function"
  rm -f "memory/derived/procedural_knowledge/${EVOLVED_PROFILE}/${FRAMEWORK}/${EXAMPLE}/practice_knowledge.jsonl"

  set +e
  uv run python -m memory.procedural_memory.target_practice distill \
    --profile "$EVOLVED_PROFILE" \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --model "$MODEL" \
    --force
  distill_rc=$?
  set -e

  PER_FUNCTION_DIR="memory/derived/procedural_knowledge/${EVOLVED_PROFILE}/${FRAMEWORK}/${EXAMPLE}/per_function"
  if ! find "$PER_FUNCTION_DIR" -maxdepth 1 -name '*.jsonl' -type f -size +0c 2>/dev/null | grep -q .; then
    if (( distill_rc != 0 )); then
      echo "WARN Online evolution example ${FRAMEWORK}/${EXAMPLE}: no passed target-practice memory; using feedback-only graph." >&2
    fi
    cp "$evolved_base_graph" "$evolved_mounted_graph"
    cp "$evolved_base_graph" "$evolved_optimized_graph"
    run_online_infer "$FRAMEWORK" "$EXAMPLE"
    return $?
  fi
  if (( distill_rc != 0 )); then
    echo "ERROR: target-practice distill failed for ${FRAMEWORK}/${EXAMPLE}." >&2
    return "$distill_rc"
  fi

  uv run python -m memory.procedural_memory.target_practice consolidate \
    --profile "$EVOLVED_PROFILE" \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE"

  uv run python -m memory.knowledge_mount \
    --graph-path "$evolved_base_graph" \
    --knowledge-root memory/derived/procedural_knowledge \
    --knowledge-profile "$EVOLVED_PROFILE" \
    --framework "$FRAMEWORK" \
    --example "$EXAMPLE" \
    --require-knowledge \
    --output "$evolved_mounted_graph"

  uv run python -m memory.incremental_optimize_graph_knowledge \
    --graph-path "$evolved_mounted_graph" \
    --output "$evolved_local_graph" \
    --mode b \
    --model "$MODEL" \
    --base-url "${OPTIMIZE_KNOWLEDGE_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}" \
    --api-key "${OPENROUTER_API_KEY:-}" \
    --cache-path "${evolved_graph_dir}/dep_graph.with_knowledge.online_local.llm_cache.json" \
    --min-confidence "${PRAXIS_CONFIDENCE_THRESHOLD:-0.6}" \
    --candidate-source-layer "$EVOLVED_PROFILE" \
    --conflict-confidence-margin "${PRAXIS_INCREMENTAL_CONFLICT_MARGIN:-0.15}"

  uv run python -m memory.incremental_optimize_graph_knowledge \
    --graph-path "$evolved_local_graph" \
    --output "$evolved_optimized_graph" \
    --mode c \
    --model "$MODEL" \
    --base-url "${OPTIMIZE_KNOWLEDGE_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}" \
    --api-key "${OPENROUTER_API_KEY:-}" \
    --cache-path "${evolved_graph_dir}/dep_graph.with_knowledge.online_propagated.llm_cache.json" \
    --min-confidence "${PRAXIS_CONFIDENCE_THRESHOLD:-0.6}" \
    --conflict-confidence-margin "${PRAXIS_INCREMENTAL_CONFLICT_MARGIN:-0.15}" \
    --max-targets-per-knowledge "${PRAXIS_INCREMENTAL_MAX_TARGETS:-3}" \
    --max-candidates-per-target "${PRAXIS_INCREMENTAL_MAX_CANDIDATES_PER_TARGET:-3}"

  run_online_infer "$FRAMEWORK" "$EXAMPLE"
}

ONLINE_RC=0
for FRAMEWORK in $FRAMEWORKS; do
  mapfile -t EXAMPLES < <(examples_for_framework "$FRAMEWORK")
  for EXAMPLE in "${EXAMPLES[@]}"; do
    if ! praxis_wait_for_job_slot "$ONLINE_CONCURRENCY"; then
      ONLINE_RC=1
    fi
    (
    set +e
    run_online_example "$FRAMEWORK" "$EXAMPLE"
    rc=$?
    set -e
    if (( rc != 0 )); then
      record_online_failure "$FRAMEWORK" "$EXAMPLE" "online evolution failed rc=${rc}"
      echo "WARN Online evolution example ${FRAMEWORK}/${EXAMPLE} failed; continuing." >&2
      exit "$rc"
    fi
    ) &
  done
done
if ! praxis_wait_for_all_jobs; then
  ONLINE_RC=1
fi

if [[ -s "$ONLINE_FAILURES_FILE" ]]; then
  echo
  echo "Online evolution completed with failed example(s):" >&2
  cat "$ONLINE_FAILURES_FILE" >&2
  exit 1
fi

exit "$ONLINE_RC"
