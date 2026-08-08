# Stage 1: In-Domain Development Practice

Stage 1 builds the observed-memory substrate used by procedural practice.

Run:

```bash
bash memory/scripts/in_domain_practice.sh
```

## Pipeline

Each configured framework/example runs:

```text
memory init
memory select --strategy business_logic,indegree,outdegree
memory generate
memory coverage
memory feedback
```

Benchmark target bodies are stubbed during exploration, selection, generation,
feedback, dependency-graph construction, and practice. Signatures and
docstrings remain available.

## Outputs

Artifacts are written under:

```text
memory/derived/observed_knowledge/{run_profile}/{framework}/
memory/derived/graph_knowledge/{run_profile}/{framework}/
```

Per-example outputs include:

```text
{example}.md
{example}/candidates.json
{example}/{function}_requirement.md
{example}/{function}_test_input.py
{example}/{function}_generate_status.json
{example}/{function}_coverage.json
{example}/{function}_feedback_log.json
{example}/{function}_feedback_status.json
```

## Resume Model

Re-run the same Stage 1 command after an interruption.

Generate and feedback record one state per function:

- `success`: usable output was produced.
- `exhausted`: deterministic attempts reached the configured terminal budget.
- `unrunnable`: required source or execution environment is unavailable.
- `retryable`: more budget remains or the attempt failed for a transient
  service/network reason.

Only the first three are terminal. Terminal states are input-sensitive rather
than permanent. They are invalidated automatically when relevant inputs change,
including:

- model or base URL
- candidate metadata or target source
- prompt or orchestration code
- generated test input or coverage
- coverage threshold
- terminal attempt budget

Transient errors such as rate limits, HTTP 5xx, connection failures, and
timeouts are never converted into an exhausted terminal state.

Coverage resume reuses only successful results whose test input, source,
resolved target location, runner, and execution context are unchanged.
Ordinary failures and stale results are measured again. A failed result is
reused only when a matching `exhausted` or `unrunnable` feedback state refers
to the same coverage fingerprint; changing any coverage input invalidates both.

## Budgets

Important defaults:

```text
PRAXIS_CONCURRENCY=5
PRAXIS_STAGE1_CONCURRENCY=5
PRAXIS_STAGE1_FUNCTION_CONCURRENCY=1
PRAXIS_STAGE1_COVERAGE_CONCURRENCY=1
PRAXIS_COVERAGE_TIMEOUT=120
PRAXIS_COVERAGE_THRESHOLD=0.8

PRAXIS_GENERATE_MAX_ITERATIONS=100

PRAXIS_FEEDBACK_MAX_ITERATIONS=120
PRAXIS_FEEDBACK_RETRIES=1
```

The Stage 1 wrapper runs up to `PRAXIS_STAGE1_CONCURRENCY` examples in
parallel. Inside each example, init, select, generate, coverage, and feedback
still run in order. Generate, coverage, and feedback also accept per-example
function-level concurrency, kept at `1` by default so the total agent/Docker
load stays close to the stage-level concurrency.

To increase per-example function fan-out as well:

```bash
export PRAXIS_STAGE1_FUNCTION_CONCURRENCY=2
export PRAXIS_STAGE1_COVERAGE_CONCURRENCY=2
```

Use coverage concurrency conservatively because each worker starts an
independent Docker container.

## Manual Commands

Run or resume coverage for one example:

```bash
uv run python cli.py memory coverage \
  --framework verl \
  --test-example prime \
  --resume \
  --concurrency 2 \
  --allow-execution-failures
```

Ignore reusable generate states:

```bash
uv run python cli.py memory generate \
  --framework verl \
  --test-example prime \
  --model "$MODEL" \
  --force
```

Ignore reusable feedback states:

```bash
uv run python cli.py memory feedback \
  --framework verl \
  --test-example prime \
  --model "$MODEL" \
  --coverage-threshold 0.8 \
  --force
```

## Stage 2 Boundary

Stage 1 completion does not mean every selected function reached the coverage
threshold. Stage 1 completes when every function either produced usable output
or reached a valid terminal state.

Stage 2 considers candidates that have both a requirement and a test input. It
does not use Stage 1 line coverage as an admission gate. Instead, Stage 2
reruns each candidate through its own oracle validation:

- at least five normal cases
- at least an 80% ground-truth success ratio across normal and edge cases
- ground truth passes `grade-smoke`
- an empty implementation fails `grade-smoke`

Candidates that fail these checks are skipped at function level. Stage 1
coverage remains a feedback and quality metric, but Stage 2 does not trust
legacy or stale coverage artifacts in place of rerunning the oracle.
