#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENHANDS_DIR:-}" ]]; then
  OPENHANDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$OPENHANDS_DIR"

: "${MODEL:=deepseek/deepseek-v3.2}"
: "${PROFILE:=deepseek_v3_2_full}"
: "${FRAMEWORKS:=verl open-r1 raganything smolagents tensorrt_model_optimizer}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running PRAXIS scripts}"

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

run_praxis_step() {
  local stage="$1"
  local framework="$2"
  local example="$3"
  local hint="$4"
  shift 4

  local retries="${PRAXIS_STAGE_RETRIES:-1}"
  local sleep_seconds="${PRAXIS_RETRY_SLEEP_SECONDS:-5}"
  local attempt=1
  local max_attempts=$((retries + 1))
  local rc=0

  while (( attempt <= max_attempts )); do
    echo
    echo ">>> ${stage}: ${framework}/${example} (attempt ${attempt}/${max_attempts})"
    set +e
    "$@"
    rc=$?
    set -e
    if (( rc == 0 )); then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      echo
      echo "ERROR: ${stage} failed for ${framework}/${example} after ${attempt} attempt(s)." >&2
      echo "Last exit code: ${rc}" >&2
      echo >&2
      echo "Command to rerun this step:" >&2
      print_command "$@" >&2
      echo >&2
      echo "Recovery hints:" >&2
      echo "  - ${hint}" >&2
      echo "  - Increase automatic retries with: export PRAXIS_STAGE_RETRIES=2" >&2
      echo "  - Re-run the same script after changing the environment variable." >&2
      return "$rc"
    fi

    echo
    echo "WARN: ${stage} failed for ${framework}/${example}; retrying in ${sleep_seconds}s..." >&2
    sleep "$sleep_seconds"
    attempt=$((attempt + 1))
  done
}

stage1_ready_functions() {
  local framework="$1"
  local example="$2"
  local threshold="${3:-${PRAXIS_COVERAGE_THRESHOLD:-0.5}}"
  local candidates="memory/derived/observed_knowledge/${framework}/${example}/candidates.json"

  CANDIDATES_PATH="$candidates" \
  FRAMEWORK="$framework" \
  EXAMPLE="$example" \
  COVERAGE_THRESHOLD="$threshold" \
  uv run python - <<'PYCODE'
import json
import os
import sys
from pathlib import Path

path = Path(os.environ["CANDIDATES_PATH"])
framework = os.environ["FRAMEWORK"]
example = os.environ["EXAMPLE"]
threshold = float(os.environ["COVERAGE_THRESHOLD"])
root = Path("memory/derived/observed_knowledge") / framework / example

if not path.exists():
    print(f"skip {framework}/{example}: candidates.json missing", file=sys.stderr)
    raise SystemExit(0)

for item in json.loads(path.read_text()):
    fn = item["function_name"]
    cov_path = root / f"{fn}_coverage.json"
    if not cov_path.exists():
        print(f"skip {fn}: coverage missing", file=sys.stderr)
        continue
    coverage = json.loads(cov_path.read_text())
    if coverage.get("is_execution_failure"):
        print(f"skip {fn}: execution failure", file=sys.stderr)
        continue
    line_coverage = float(coverage.get("line_coverage") or 0.0)
    if line_coverage < threshold:
        print(f"skip {fn}: coverage {line_coverage:.1%} < {threshold:.0%}", file=sys.stderr)
        continue
    print(fn)
PYCODE
}

summarize_stage1_example() {
  local framework="$1"
  local example="$2"
  local threshold="${3:-${PRAXIS_COVERAGE_THRESHOLD:-0.5}}"
  local candidates="memory/derived/observed_knowledge/${framework}/${example}/candidates.json"

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

path = Path(os.environ["CANDIDATES_PATH"])
framework = os.environ["FRAMEWORK"]
example = os.environ["EXAMPLE"]
threshold = float(os.environ["COVERAGE_THRESHOLD"])
root = Path("memory/derived/observed_knowledge") / framework / example

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
    if coverage.get("is_execution_failure"):
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
