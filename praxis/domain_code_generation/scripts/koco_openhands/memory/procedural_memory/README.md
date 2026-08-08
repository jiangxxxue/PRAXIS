# Stage 2: Structured Knowledge Acquisition

Stage 2 practices Stage 1 functions and distills successful trajectories into
structured procedural knowledge.

Run:

```bash
bash memory/scripts/structured_knowledge_acquisition.sh
```

## Input Filter

For each candidate, Stage 2 requires:

- a non-empty requirement
- a non-empty test input

Stage 1 line coverage is a quality and feedback target, not a Stage 2
admission gate. Stage 2 validates each candidate independently:

- `procedural init` reruns the ground-truth implementation and requires at
  least five normal cases with an 80% success ratio across normal and edge
  cases.
- `procedural grade-smoke` requires the ground truth to pass and an empty
  implementation to fail.

Examples with no candidates that have both input artifacts are skipped.

## Pipeline

Stage 2 runs up to `PRAXIS_STAGE2_CONCURRENCY` eligible functions in parallel
and defaults to `PRAXIS_CONCURRENCY=5`. Each function still runs its internal
steps in order:

```text
procedural init
procedural grade-smoke
procedural practice
procedural distill-structured
```

After all per-function workers finish, Stage 2 consolidates each example into
`practice_knowledge.jsonl`. Failures are isolated at function level. One
unsuitable function does not discard other eligible functions in the same
example.

`distill-structured` assigns the final `confidence.score` for every extracted
knowledge entry. The score represents how reliable the knowledge is when its
trigger matches, based on the ground-truth implementation, practice outcomes,
and cited evidence. Downstream graph construction and retrieval use this score
to decide whether knowledge should be trusted or filtered out; there is no
separate confidence rescore step.

## Outputs

```text
memory/derived/procedural_knowledge/_traces/{profile}/{framework}/{example}/{function}/
memory/derived/procedural_knowledge/{profile}/{framework}/{example}/per_function/
memory/derived/procedural_knowledge/{profile}/{framework}/{example}/practice_knowledge.jsonl
```

The consolidated `practice_knowledge.jsonl` is the input to Stage 3.

## Main Parameters

```text
PRAXIS_PRACTICE_K=8
PRAXIS_PRACTICE_MAX_ITERATIONS=100
PRAXIS_CONCURRENCY=5
PRAXIS_STAGE2_CONCURRENCY=5
```

Keep practice budgets fixed across experiments intended for direct comparison.

## Resume

Re-run the same Stage 2 command after an interruption. Stage 2 reconstructs
per-example consolidated outputs from candidates that pass its oracle checks.
