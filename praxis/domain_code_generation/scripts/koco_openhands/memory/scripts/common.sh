#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENHANDS_DIR:-}" ]]; then
  OPENHANDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$OPENHANDS_DIR"

: "${MODEL:=deepseek/deepseek-v3.2}"
: "${PROFILE:=deepseek_v3_2_full}"
: "${FRAMEWORKS:=verl open-r1 raganything smolagents tensorrt_model_optimizer}"
: "${PRAXIS_CONCURRENCY:=5}"
: "${PRAXIS_STAGE1_CONCURRENCY:=$PRAXIS_CONCURRENCY}"
: "${PRAXIS_STAGE2_CONCURRENCY:=$PRAXIS_CONCURRENCY}"
: "${PRAXIS_STAGE3_CONCURRENCY:=$PRAXIS_CONCURRENCY}"
: "${PRAXIS_STAGE4_CONCURRENCY:=$PRAXIS_CONCURRENCY}"
: "${PRAXIS_STAGE1_FUNCTION_CONCURRENCY:=1}"
: "${PRAXIS_COVERAGE_CONCURRENCY:=$PRAXIS_STAGE1_FUNCTION_CONCURRENCY}"
: "${PRAXIS_STAGE1_COVERAGE_CONCURRENCY:=$PRAXIS_COVERAGE_CONCURRENCY}"
: "${PRAXIS_STAGE4_INFER_CONCURRENCY:=1}"
: "${PRAXIS_STAGE4_RETRIES:=1}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running PRAXIS scripts}"

MODEL_KEY="${MODEL//\//__}"
MODEL_KEY="${MODEL_KEY//[^A-Za-z0-9_.-]/_}"
PROFILE_KEY="${PROFILE//\//__}"
PROFILE_KEY="${PROFILE_KEY//[^A-Za-z0-9_.-]/_}"
: "${PRAXIS_RUN_PROFILE:=${PROFILE_KEY}__${MODEL_KEY}}"
PRAXIS_RUN_PROFILE="${PRAXIS_RUN_PROFILE//\//__}"
PRAXIS_RUN_PROFILE="${PRAXIS_RUN_PROFILE//[^A-Za-z0-9_.-]/_}"
PROFILE="$PRAXIS_RUN_PROFILE"
: "${PRAXIS_MEMORY_RUN_ID:=$PROFILE}"
PRAXIS_MEMORY_RUN_ID="${PRAXIS_MEMORY_RUN_ID//\//__}"
PRAXIS_MEMORY_RUN_ID="${PRAXIS_MEMORY_RUN_ID//[^A-Za-z0-9_.-]/_}"
export PROFILE PRAXIS_RUN_PROFILE PRAXIS_MEMORY_RUN_ID
export PRAXIS_CONCURRENCY PRAXIS_STAGE1_CONCURRENCY PRAXIS_STAGE2_CONCURRENCY
export PRAXIS_STAGE3_CONCURRENCY PRAXIS_STAGE4_CONCURRENCY
export PRAXIS_STAGE1_FUNCTION_CONCURRENCY PRAXIS_COVERAGE_CONCURRENCY
export PRAXIS_STAGE1_COVERAGE_CONCURRENCY PRAXIS_STAGE4_INFER_CONCURRENCY
export PRAXIS_STAGE4_RETRIES

OBSERVED_RUN_ROOT="memory/derived/observed_knowledge/${PRAXIS_MEMORY_RUN_ID}"
STAGE1_GRAPH_ROOT="memory/derived/graph_knowledge/${PRAXIS_MEMORY_RUN_ID}"
PROFILE_GRAPH_ROOT="memory/derived/graph_knowledge/${PROFILE}"

STAGE1_FAILURES_FILE="${OBSERVED_RUN_ROOT}/stage1_failures.tsv"

praxis_concurrency_value() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${name} must be a positive integer; got '${value}'." >&2
    return 1
  fi
  printf '%s\n' "$value"
}

praxis_running_jobs_count() {
  jobs -pr | wc -l | tr -d ' '
}

praxis_wait_for_job_slot() {
  local limit="$1"
  local rc=0
  while (( "$(praxis_running_jobs_count)" >= limit )); do
    if ! wait -n; then
      rc=1
    fi
  done
  return "$rc"
}

praxis_wait_for_all_jobs() {
  local rc=0
  while (( "$(praxis_running_jobs_count)" > 0 )); do
    if ! wait -n; then
      rc=1
    fi
  done
  return "$rc"
}

praxis_append_line_locked() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  (
    flock -x 9
    printf '%s\n' "$*" >&9
  ) 9>>"$path"
}

stage1_clean_example() {
  local framework="$1"
  local example="$2"

  rm -rf "${OBSERVED_RUN_ROOT:?}/${framework}/${example}"
  rm -f "${OBSERVED_RUN_ROOT:?}/${framework}/${example}.md"
  rm -rf "${STAGE1_GRAPH_ROOT:?}/${framework}/${example}"
}

examples_for_framework() {
  local framework="$1"
  if [[ -n "${PRAXIS_EXAMPLES:-}" ]]; then
    printf '%s\n' $PRAXIS_EXAMPLES
    return
  fi
  find "$OPENHANDS_DIR/../../${framework}/test_examples" \
    -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

max_int() {
  local a="${1:-0}"
  local b="${2:-0}"
  if (( a >= b )); then
    printf '%s\n' "$a"
  else
    printf '%s\n' "$b"
  fi
}

tier_value() {
  local base="${1:-0}"
  local level="$2"
  shift 2

  local value="$base"
  local tier
  local i=1
  for tier in "$@"; do
    if (( level >= i )); then
      value="$tier"
    fi
    i=$((i + 1))
  done
  max_int "$base" "$value"
}

stage1_stage_retries() {
  local stage="$1"
  local env_name
  local default_retries

  case "$stage" in
    "observed-memory init")
      env_name="PRAXIS_INIT_STAGE_RETRIES"
      default_retries=1
      ;;
    "observed-memory select")
      env_name="PRAXIS_SELECT_STAGE_RETRIES"
      default_retries=1
      ;;
    "observed-memory generate")
      env_name="PRAXIS_GENERATE_STAGE_RETRIES"
      default_retries=0
      ;;
    "observed-memory coverage")
      env_name="PRAXIS_COVERAGE_STAGE_RETRIES"
      default_retries=0
      ;;
    "observed-memory feedback")
      env_name="PRAXIS_FEEDBACK_STAGE_RETRIES"
      default_retries=0
      ;;
    *)
      env_name=""
      default_retries=1
      ;;
  esac

  local retries=""
  if [[ -n "$env_name" ]]; then
    retries="${!env_name-}"
  fi
  if [[ -z "$retries" ]]; then
    retries="${PRAXIS_STAGE_RETRIES:-$default_retries}"
  fi
  if [[ ! "$retries" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${env_name:-PRAXIS_STAGE_RETRIES} must be a non-negative integer." >&2
    return 1
  fi
  printf '%s\n' "$retries"
}

stage1_generate_terminal_iterations() {
  if [[ -n "${PRAXIS_GENERATE_TERMINAL_MAX_ITERATIONS:-}" ]]; then
    max_int \
      "${PRAXIS_GENERATE_MAX_ITERATIONS:-100}" \
      "$PRAXIS_GENERATE_TERMINAL_MAX_ITERATIONS"
    return
  fi
  printf '%s\n' "${PRAXIS_GENERATE_MAX_ITERATIONS:-100}"
}

stage1_feedback_terminal_iterations() {
  if [[ -n "${PRAXIS_FEEDBACK_TERMINAL_MAX_ITERATIONS:-}" ]]; then
    max_int \
      "${PRAXIS_FEEDBACK_MAX_ITERATIONS:-120}" \
      "$PRAXIS_FEEDBACK_TERMINAL_MAX_ITERATIONS"
    return
  fi
  printf '%s\n' "${PRAXIS_FEEDBACK_MAX_ITERATIONS:-120}"
}

stage1_feedback_terminal_retries() {
  if [[ -n "${PRAXIS_FEEDBACK_TERMINAL_RETRIES:-}" ]]; then
    max_int \
      "${PRAXIS_FEEDBACK_RETRIES:-1}" \
      "$PRAXIS_FEEDBACK_TERMINAL_RETRIES"
    return
  fi
  max_int "${PRAXIS_FEEDBACK_RETRIES:-1}" 1
}

set_cli_arg() {
  local -n target_cmd="$1"
  local flag="$2"
  local value="$3"
  local i

  for ((i = 0; i < ${#target_cmd[@]}; i++)); do
    if [[ "${target_cmd[$i]}" == "$flag" ]]; then
      if (( i + 1 < ${#target_cmd[@]} )); then
        target_cmd[$((i + 1))]="$value"
      else
        target_cmd+=("$value")
      fi
      return 0
    fi
  done

  target_cmd+=("$flag" "$value")
}

stage1_escalate_command() {
  local stage="$1"
  local attempt="$2"
  local cmd_name="$3"

  if [[ "${PRAXIS_AUTO_ESCALATE:-1}" == "0" ]]; then
    return 0
  fi

  local level=$((attempt - 1))
  if (( level <= 0 )); then
    return 0
  fi

  local -n cmd_ref="$cmd_name"
  local max_iterations
  local max_retries
  local timeout

  case "$stage" in
    "observed-memory init")
      max_iterations="$(tier_value "${PRAXIS_OBSERVED_MAX_ITERATIONS:-100}" "$level" 150 220)"
      set_cli_arg cmd_ref "--max-iterations" "$max_iterations"
      echo "    auto-escalate: max_iterations=${max_iterations}"
      ;;
    "observed-memory select")
      max_iterations="$(tier_value "${PRAXIS_SELECT_MAX_ITERATIONS:-100}" "$level" 150 220)"
      set_cli_arg cmd_ref "--max-iterations" "$max_iterations"
      echo "    auto-escalate: max_iterations=${max_iterations}"
      ;;
    "observed-memory generate"|"observed-memory coverage"|"observed-memory feedback")
      echo "    retrying transient/process-level failure with the same function budgets"
      ;;
  esac
}

run_praxis_step() {
  local stage="$1"
  local framework="$2"
  local example="$3"
  local hint="$4"
  shift 4

  local retries
  retries="$(stage1_stage_retries "$stage")"
  local sleep_seconds="${PRAXIS_RETRY_SLEEP_SECONDS:-10}"
  local attempt=1
  local max_attempts=$((retries + 1))
  local rc=0
  local -a base_cmd=("$@")
  local -a attempt_cmd=()

  while (( attempt <= max_attempts )); do
    attempt_cmd=("${base_cmd[@]}")
    echo
    echo ">>> ${stage}: ${framework}/${example} (attempt ${attempt}/${max_attempts})"
    stage1_escalate_command "$stage" "$attempt" attempt_cmd
    set +e
    "${attempt_cmd[@]}"
    rc=$?
    set -e
    if (( rc == 0 )); then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      echo
      echo "ERROR: ${stage} failed for ${framework}/${example} after ${attempt} attempt(s)." >&2
      echo "Last exit code: ${rc}" >&2
      return "$rc"
    fi

    echo
    echo "WARN: ${stage} failed for ${framework}/${example}; retrying in ${sleep_seconds}s..." >&2
    sleep "$sleep_seconds"
    attempt=$((attempt + 1))
  done
}

stage1_step_done() {
  local step="$1"
  local framework="$2"
  local example="$3"
  local threshold="${4:-${PRAXIS_COVERAGE_THRESHOLD:-0.8}}"

  STAGE1_STEP="$step" \
  OBSERVED_RUN_ROOT="$OBSERVED_RUN_ROOT" \
  STAGE1_GRAPH_ROOT="$STAGE1_GRAPH_ROOT" \
  FRAMEWORK="$framework" \
  EXAMPLE="$example" \
  COVERAGE_THRESHOLD="$threshold" \
  uv run python - <<'PYCODE'
import json
import os
import sys
from pathlib import Path

from memory.observed_memory.quality import coverage_ready as stage2_coverage_ready

step = os.environ["STAGE1_STEP"]
framework = os.environ["FRAMEWORK"]
example = os.environ["EXAMPLE"]
threshold = float(os.environ["COVERAGE_THRESHOLD"])
observed_root = Path(os.environ["OBSERVED_RUN_ROOT"])
graph_root = Path(os.environ["STAGE1_GRAPH_ROOT"])

example_root = observed_root / framework / example
candidates_path = example_root / "candidates.json"


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def load_candidates():
    if not nonempty(candidates_path):
        return None
    try:
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(candidates, list) or not candidates:
        return None
    for item in candidates:
        if not isinstance(item, dict):
            return None
        if not item.get("function_name") or not item.get("implementation_location"):
            return None
    return candidates


def coverage_for(function_name: str):
    path = example_root / f"{function_name}_coverage.json"
    if not nonempty(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def coverage_ready(coverage: dict) -> bool:
    return stage2_coverage_ready(coverage, threshold)


def test_input_for(function_name: str) -> Path | None:
    path = example_root / f"{function_name}_test_input.py"
    if nonempty(path):
        return path
    leaf = function_name.rsplit(".", 1)[-1] if "." in function_name else ""
    if leaf and leaf != function_name:
        leaf_path = example_root / f"{leaf}_test_input.py"
        if nonempty(leaf_path):
            return leaf_path
    return None


def done() -> bool:
    if step == "init":
        return (
            nonempty(observed_root / framework / f"{example}.md")
            and nonempty(graph_root / framework / example / "dep_graph.json")
        )

    candidates = load_candidates()
    if candidates is None:
        return False

    if step == "select":
        return True

    generated_candidates = [
        item for item in candidates
        if test_input_for(item["function_name"]) is not None
    ]

    if step == "generate":
        return False

    if not generated_candidates:
        return False

    coverages = [coverage_for(item["function_name"]) for item in generated_candidates]
    if any(item is None for item in coverages):
        return False

    if step == "coverage":
        return False

    if step == "feedback":
        return all(coverage_ready(coverage) for coverage in coverages)

    raise SystemExit(f"unknown Stage 1 step: {step}")


sys.exit(0 if done() else 1)
PYCODE
}

run_praxis_step_if_needed() {
  local step_key="$1"
  local stage="$2"
  local framework="$3"
  local example="$4"
  local hint="$5"
  shift 5

  if stage1_step_done "$step_key" "$framework" "$example" "${PRAXIS_COVERAGE_THRESHOLD:-0.8}"; then
    echo
    echo ">>> ${stage}: ${framework}/${example} (resume: already complete)"
    return 0
  fi

  run_praxis_step "$stage" "$framework" "$example" "$hint" "$@"
}

record_stage1_failure() {
  local framework="$1"
  local example="$2"
  local stage="$3"
  local rc="$4"

  praxis_append_line_locked "$STAGE1_FAILURES_FILE" "$(printf '%s\t%s\t%s\t%s' "$framework" "$example" "$stage" "$rc")"
}

stage2_candidate_functions() {
  local framework="$1"
  local example="$2"
  local candidates="${OBSERVED_RUN_ROOT}/${framework}/${example}/candidates.json"

  CANDIDATES_PATH="$candidates" \
  FRAMEWORK="$framework" \
  EXAMPLE="$example" \
  uv run python - <<'PYCODE'
import json
import os
import sys
from pathlib import Path

path = Path(os.environ["CANDIDATES_PATH"])
framework = os.environ["FRAMEWORK"]
example = os.environ["EXAMPLE"]
root = path.parent

if not path.exists():
    print(f"skip {framework}/{example}: candidates.json missing", file=sys.stderr)
    raise SystemExit(0)

for item in json.loads(path.read_text()):
    fn = item["function_name"]
    requirement_path = root / f"{fn}_requirement.md"
    test_input_path = root / f"{fn}_test_input.py"
    if not requirement_path.is_file() or requirement_path.stat().st_size == 0:
        print(f"skip {fn}: requirement missing", file=sys.stderr)
        continue
    if not test_input_path.is_file() or test_input_path.stat().st_size == 0:
        print(f"skip {fn}: test input missing", file=sys.stderr)
        continue
    print(fn)
PYCODE
}

summarize_stage1_example() {
  local framework="$1"
  local example="$2"
  local threshold="${3:-${PRAXIS_COVERAGE_THRESHOLD:-0.8}}"
  local candidates="${OBSERVED_RUN_ROOT}/${framework}/${example}/candidates.json"

  echo
  echo ">>> Stage 1 summary: ${framework}/${example}"
  CANDIDATES_PATH="$candidates" \
  FRAMEWORK="$framework" \
  EXAMPLE="$example" \
  COVERAGE_THRESHOLD="$threshold" \
  uv run python - <<'PYCODE'
import json
import os
from pathlib import Path

from memory.observed_memory.quality import coverage_has_execution_failure

path = Path(os.environ["CANDIDATES_PATH"])
framework = os.environ["FRAMEWORK"]
example = os.environ["EXAMPLE"]
threshold = float(os.environ["COVERAGE_THRESHOLD"])
root = path.parent

if not path.exists():
    print("  candidates: missing")
    raise SystemExit(0)

candidates = json.loads(path.read_text())
ready = []
failed = []
low = []
missing = []

for item in candidates:
    fn = item["function_name"]
    cov_path = root / f"{fn}_coverage.json"
    if not cov_path.exists():
        missing.append(fn)
        continue
    coverage = json.loads(cov_path.read_text())
    line_coverage = float(coverage.get("line_coverage") or 0.0)
    if coverage_has_execution_failure(coverage):
        failed.append(f"{fn} ({line_coverage:.1%})")
    elif line_coverage < threshold:
        low.append(f"{fn} ({line_coverage:.1%})")
    else:
        ready.append(f"{fn} ({line_coverage:.1%})")

print(f"  candidates: {len(candidates)}")
print(f"  ready for Stage 2 (>= {threshold:.0%}): {len(ready)}")
for item in ready:
    print(f"    OK   {item}")
if failed:
    print(f"  execution failures: {len(failed)}")
    for item in failed:
        print(f"    FAIL {item}")
if low:
    print(f"  below threshold: {len(low)}")
    for item in low:
        print(f"    LOW  {item}")
if missing:
    print(f"  missing coverage: {len(missing)}")
    for fn in missing:
        print(f"    MISS {fn}")
PYCODE
}

summarize_stage1_run() {
  local threshold="${1:-${PRAXIS_COVERAGE_THRESHOLD:-0.8}}"

  echo
  echo ">>> Stage 1 final summary"
  FRAMEWORKS="$FRAMEWORKS" \
  OPENHANDS_DIR="$OPENHANDS_DIR" \
  OBSERVED_RUN_ROOT="$OBSERVED_RUN_ROOT" \
  STAGE1_FAILURES_FILE="$STAGE1_FAILURES_FILE" \
  COVERAGE_THRESHOLD="$threshold" \
  PRAXIS_EXAMPLES="${PRAXIS_EXAMPLES:-}" \
  uv run python - <<'PYCODE'
import json
import os
from pathlib import Path

from memory.observed_memory.quality import coverage_has_execution_failure

frameworks = [item for item in os.environ["FRAMEWORKS"].split() if item]
openhands_dir = Path(os.environ["OPENHANDS_DIR"])
observed_root = Path(os.environ["OBSERVED_RUN_ROOT"])
failures_path = Path(os.environ["STAGE1_FAILURES_FILE"])
threshold = float(os.environ["COVERAGE_THRESHOLD"])
explicit_examples = [item for item in os.environ.get("PRAXIS_EXAMPLES", "").split() if item]


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def examples_for(framework: str) -> list[str]:
    if explicit_examples:
        return sorted(explicit_examples)
    root = openhands_dir.parent.parent / framework / "test_examples"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ready_count(framework: str, example: str) -> tuple[int, int, int, int]:
    root = observed_root / framework / example
    candidates = load_json(root / "candidates.json")
    if not isinstance(candidates, list):
        return 0, 0, 0, 0
    ready = failed = low = missing = 0
    for item in candidates:
        if not isinstance(item, dict):
            continue
        fn = item.get("function_name")
        if not fn:
            continue
        coverage = load_json(root / f"{fn}_coverage.json")
        if not isinstance(coverage, dict):
            missing += 1
            continue
        line_coverage = float(coverage.get("line_coverage") or 0.0)
        if coverage_has_execution_failure(coverage):
            failed += 1
        elif line_coverage < threshold:
            low += 1
        else:
            ready += 1
    return ready, failed, low, missing


failures = []
if failures_path.exists():
    for line in failures_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            failures.append(tuple(parts))
failure_map = {(framework, example): (stage, rc) for framework, example, stage, rc in failures}

rows = []
for framework in frameworks:
    for example in examples_for(framework):
        ready, failed, low, missing = ready_count(framework, example)
        rows.append((framework, example, ready, failed, low, missing))

total = len(rows)
with_ready = sum(1 for row in rows if row[2] > 0)
without_ready = sum(1 for row in rows if row[2] == 0)
passed = [
    row for row in rows
    if (row[0], row[1]) not in failure_map and row[2] > 0
]
failed_or_skipped = [
    row for row in rows
    if (row[0], row[1]) in failure_map
]

print(f"  examples considered: {total}")
print(f"  examples passed Stage 1 with Stage 2-ready functions: {len(passed)}")
print(f"  examples without ready functions: {without_ready}")
print(f"  unresolved step failures: {len(failures)}")

if passed:
    print("  passed examples:")
    for framework, example, ready, failed, low, missing in passed:
        print(
            f"    PASS {framework}/{example}: "
            f"ready={ready}, execution_failures={failed}, "
            f"below_threshold={low}, missing_coverage={missing}"
        )

if failed_or_skipped:
    print("  failed/skipped examples:")
    for framework, example, ready, failed, low, missing in failed_or_skipped:
        stage, rc = failure_map[(framework, example)]
        print(
            f"    FAIL {framework}/{example}: {stage} (exit {rc}); "
            f"ready={ready}, execution_failures={failed}, "
            f"below_threshold={low}, missing_coverage={missing}"
        )

no_ready = [row for row in rows if row[2] == 0 and (row[3] or row[4] or row[5])]
if no_ready:
    print("  examples with coverage but no Stage 2-ready functions:")
    for framework, example, _ready, failed, low, missing in no_ready:
        print(
            f"    {framework}/{example}: "
            f"execution_failures={failed}, below_threshold={low}, missing_coverage={missing}"
        )

if failures:
    print(
        "  To fix failed/skipped examples, re-run: "
        "bash memory/scripts/in_domain_practice.sh"
    )
    print("  Completed steps will be reused; successful examples will be skipped.")
PYCODE
}
