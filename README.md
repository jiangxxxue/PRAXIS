# PRAXIS

This repository provides the source code for Work "PRAXIS: Graph-Grounded Tacit Knowledge for Domain Code Generation".

PRAXIS has four stages:

1. **In-Domain Development Practice**: observe projects, select practice targets, generate and repair executable test inputs.
2. **Structured Knowledge Acquisition**: practice selected functions and distill traces into structured procedural knowledge.
3. **Graph-Grounded Knowledge Organization**: mount structured knowledge onto dependency graphs and optimize it.
4. **Tacit Knowledge Injection**: run OpenHands inference with observed memory and optimized graph knowledge, then evaluate.

## Setup

From a fresh clone:

```bash
cd PRAXIS
export REPO_ROOT="$PWD"
export OPENHANDS_DIR="$REPO_ROOT/praxis/domain_code_generation/scripts/koco_openhands"
cd "$OPENHANDS_DIR"

uv sync

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
```

PRAXIS executes benchmark code in the same Docker images used by KOCO-Bench evaluation. Pull and tag the pre-built images before running coverage, procedural grading, or evaluation:

```bash
# Used by verl and open-r1. Approximately 80.6 GB.
docker pull drunkpiano2005/koco-verl-openr1:1.0
docker tag drunkpiano2005/koco-verl-openr1:1.0 kocobench/verl-openr1:v0.4

# Used by tensorrt_model_optimizer. Approximately 42.4 GB.
docker pull drunkpiano2005/koco-tensorrt:1.0
docker tag drunkpiano2005/koco-tensorrt:1.0 tensorrt:latest

# Used by raganything and smolagents. Approximately 5 GB.
docker pull drunkpiano2005/koco-raganything-smolagents:1.0
docker tag drunkpiano2005/koco-raganything-smolagents:1.0 raganything-smolagents:test
```

## Quick Start

By default, PRAXIS runs all supported code-generation frameworks with DeepSeek V3.2:

```bash
export MODEL=deepseek/deepseek-v3.2
export PROFILE=deepseek_v3_2_full
export FRAMEWORKS="verl open-r1 raganything smolagents tensorrt_model_optimizer"
```

That is the standard full-framework PRAXIS workflow.

For a focused end-to-end run, restrict the workflow to one framework and one test example:

```bash
export FRAMEWORKS="verl"
export PRAXIS_EXAMPLES="prime"
export PROFILE=deepseek_v3_2_verl_prime
```

## Stage 1: In-Domain Development Practice

```bash
bash memory/scripts/in_domain_practice.sh
```

This stage creates the in-domain development substrate used by later stages. It explores each project, selects practice-worthy functions, generates executable test inputs, measures coverage, and uses feedback to repair weak tests.

Each step is retried once by default. If an OpenHands agent gets stuck or exits before producing the expected artifact, increase the relevant limit and rerun the same script:

```bash
export PRAXIS_STAGE_RETRIES=2
export PRAXIS_OBSERVED_MAX_ITERATIONS=150
export PRAXIS_SELECT_MAX_ITERATIONS=150
export PRAXIS_GENERATE_MAX_ITERATIONS=150
export PRAXIS_FEEDBACK_MAX_ITERATIONS=150
```

PRAXIS keeps OpenHands stuck detection enabled, but uses wider thresholds than the SDK defaults. For difficult generation cases, relax or disable the detector explicitly:

```bash
export KOCO_OPENHANDS_STUCK_ACTION_OBSERVATION=12
export KOCO_OPENHANDS_STUCK_ACTION_ERROR=6
export KOCO_OPENHANDS_STUCK_MONOLOGUE=8
export KOCO_OPENHANDS_STUCK_ALTERNATING_PATTERN=14
```

It runs, for every configured framework and example:

```text
memory init
memory select --strategy llm
memory generate
memory coverage
memory feedback
```

It produces:

```text
memory/derived/observed_knowledge/{framework}/{example}.md
memory/derived/observed_knowledge/{framework}/{example}/candidates.json
memory/derived/observed_knowledge/{framework}/{example}/{function}_requirement.md
memory/derived/observed_knowledge/{framework}/{example}/{function}_test_input.py
memory/derived/observed_knowledge/{framework}/{example}/{function}_coverage.json
memory/derived/observed_knowledge/{framework}/{example}/{function}_feedback_log.json
memory/derived/graph_knowledge/{framework}/{example}/dep_graph.json
```

## Stage 2: Structured Knowledge Acquisition

```bash
bash memory/scripts/structured_knowledge_acquisition.sh
```

This stage reads each `candidates.json`, keeps only functions whose Stage 1 coverage is at or above `PRAXIS_COVERAGE_THRESHOLD`, practices those functions, and consolidates structured procedural knowledge:

```text
procedural init
procedural grade-smoke
procedural practice
procedural distill-structured
procedural consolidate-structured
```

It produces:

```text
memory/derived/procedural_knowledge/_traces/{profile}/{framework}/{example}/{function}/pilot_spec.json
memory/derived/procedural_knowledge/_traces/{profile}/{framework}/{example}/{function}/oracle.json
memory/derived/procedural_knowledge/_traces/{profile}/{framework}/{example}/{function}/trace.json
memory/derived/procedural_knowledge/{profile}/{framework}/{example}/per_function/{function}.jsonl
memory/derived/procedural_knowledge/{profile}/{framework}/{example}/practice_knowledge.jsonl
```

## Stage 3: Graph-Grounded Knowledge Organization

```bash
bash memory/scripts/graph_grounded_organization.sh
```

This stage mounts procedural knowledge onto dependency graphs and optimizes graph-mounted knowledge:

```text
memory.knowledge_mount
memory.optimize_graph_knowledge
```

It produces:

```text
memory/derived/graph_knowledge/{profile}/{framework}/{example}/dep_graph.with_knowledge.json
memory/derived/graph_knowledge/{profile}/{framework}/{example}/dep_graph.with_knowledge.optimized.json
```

The default propagation decision cap is `1000`. To remove the cap:

```bash
export PRAXIS_MAX_PROPAGATION_DECISIONS=0
```

## Stage 4: Tacit Knowledge Injection

```bash
bash memory/scripts/tacit_knowledge_injection.sh
```

This stage runs OpenHands inference and evaluation using optimized graph knowledge:

```text
cli.py run --graph-knowledge-artifact optimized
```

It produces:

```text
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/algorithm_methods_data_{example}_output.jsonl
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/algorithm_methods_data_{example}_result.jsonl
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/algorithm_methods_data_{example}_result.metrics.json
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/agent_logs/{example}/{function}/task_prompt.txt
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/agent_logs/{example}/{function}/sdk_events.json
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/agent_logs/{example}/{function}/tool_trace.jsonl
```

## Optional: Online Knowledge Evolution

After Stage 4, PRAXIS can evolve its knowledge online from inference trajectories. This loop imports target-practice traces, distills target memories, consolidates them into procedural knowledge, mounts the resulting target-function practice knowledge onto the previously optimized graph, and runs inference/evaluation again with the mounted graph. It does not run another graph-knowledge propagation pass.

```bash
bash memory/scripts/online_knowledge_evolution.sh
```

By default, online evolution writes graph artifacts under a new profile:

```bash
export PRAXIS_EVOLVED_PROFILE="${PROFILE}_online"
```

This loop reads:

```text
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/algorithm_methods_data_{example}_output.jsonl
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/algorithm_methods_data_{example}_result.jsonl
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/agent_logs/{example}/{function}/
memory/derived/graph_knowledge/{profile}/{framework}/{example}/dep_graph.with_knowledge.optimized.json
```

It writes or updates:

```text
memory/derived/procedural_knowledge/_target_traces/{evolved_profile}/{framework}/{example}/{function}/trace.json
memory/derived/procedural_knowledge/{evolved_profile}/{framework}/{example}/practice_knowledge.jsonl
memory/derived/graph_knowledge/{evolved_profile}/{framework}/{example}/dep_graph.with_knowledge.json
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/algorithm_methods_data_{example}_output.jsonl
praxis/domain_code_generation/scripts/data/{framework}/openhands/{model_dir}/algorithm_methods_data_{example}_result.jsonl
```
