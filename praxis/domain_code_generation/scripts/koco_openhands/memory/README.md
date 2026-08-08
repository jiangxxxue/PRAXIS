# Stages 3 and 4

## Stage 3: Graph-Grounded Knowledge Organization

Run after Stage 2:

```bash
bash memory/scripts/graph_grounded_organization.sh
```

Stage 3 mounts consolidated procedural knowledge onto dependency graphs and
optimizes the graph-mounted knowledge:

```text
memory.knowledge_mount
memory.optimize_graph_knowledge
```

Stage 3 runs up to `PRAXIS_STAGE3_CONCURRENCY` examples in parallel and
defaults to `PRAXIS_CONCURRENCY=5`. For each example, mounting still completes
before optimization starts.

Outputs:

```text
memory/derived/graph_knowledge/{profile}/{framework}/{example}/dep_graph.with_knowledge.json
memory/derived/graph_knowledge/{profile}/{framework}/{example}/dep_graph.with_knowledge.optimized.json
```

The stage requires non-empty procedural knowledge for an example. The default
propagation decision cap is `1000`; set
`PRAXIS_MAX_PROPAGATION_DECISIONS=0` to remove it.
Examples without a non-empty Stage 2 `practice_knowledge.jsonl` are reported
and skipped.
Examples with an existing non-empty optimized graph are also skipped so an
interrupted Stage 3 run can resume safely. Set `PRAXIS_STAGE3_FORCE=1` to
rebuild completed examples.

## Stage 4: Tacit Knowledge Injection

Run after Stage 3:

```bash
bash memory/scripts/tacit_knowledge_injection.sh
```

Stage 4 runs target inference with optimized graph knowledge and evaluates the
generated implementations. Target inference uses terminal/file-editor
exploration plus graph-knowledge injection; it does not enable the
`knowledge_search` tool.

Stage 4 runs up to `PRAXIS_STAGE4_CONCURRENCY` examples in parallel and
defaults to `PRAXIS_CONCURRENCY=5`. The inner `cli.py run --concurrency`
defaults to `PRAXIS_STAGE4_INFER_CONCURRENCY=1`, so the total default fan-out is
five examples rather than five examples times five functions.

Outputs are written under:

```text
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/
```

This includes inference JSONL, evaluation JSONL, metrics, task prompts, SDK
events, and tool traces. Re-running Stage 4 resumes incomplete function
attempts. Set `PRAXIS_STAGE4_FORCE=1` only when completed inference and
evaluation results should be discarded and rebuilt.

If an example has no optimized graph, Stage 4 skips it. A failed example is
reported and does not prevent later examples from running; the stage exits
nonzero after listing all failures.

## Optional Online Evolution

Run after Stage 4:

```bash
bash memory/scripts/online_knowledge_evolution.sh
```

Online evolution imports target-practice traces, distills target memories,
mounts them onto a copy of the baseline optimized graph, and runs the default
online optimization pipeline before inference and evaluation:

1. deduplicate new local knowledge and resolve same-node conflicts;
2. propagate active new local knowledge one hop to callers;
3. deduplicate propagated knowledge and resolve conflicts on each target node.

The final inference reads
`dep_graph.with_knowledge.optimized.json`. If an example produces no new
knowledge, online evolution still runs inference with the feedback-updated
baseline graph.

By default, evolved graph artifacts use:

```bash
export PRAXIS_EVOLVED_PROFILE="${PROFILE}_online"
```

Keep the evolved profile separate from the baseline profile when comparing
results.
