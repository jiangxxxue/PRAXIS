# PRAXIS

Source code for "PRAXIS: Graph-Grounded Tacit Knowledge for Domain Code
Generation".

PRAXIS runs four stages in order:

1. In-domain development practice
2. Structured knowledge acquisition
3. Graph-grounded knowledge organization
4. Tacit knowledge injection and evaluation

## Setup

```bash
cd PRAXIS
export REPO_ROOT="$PWD"
export OPENHANDS_DIR="$REPO_ROOT/praxis/domain_code_generation/scripts/koco_openhands"
cd "$OPENHANDS_DIR"

uv sync
export OPENROUTER_API_KEY="..."
```

The same OpenRouter key is used for chat models and semantic embeddings by
default. If your embedding service uses a different provider or credential,
override it explicitly:

```bash
export EMBEDDING_API_KEY="..."
export EMBEDDING_API_BASE="https://api.openai.com/v1"
export EMBEDDING_MODEL="openai/text-embedding-3-small"
```

Without an available embedding key, knowledge search uses BM25 only.

## Docker Images

```bash
docker pull anonymous-koco/koco-verl-openr1:1.0
docker tag anonymous-koco/koco-verl-openr1:1.0 kocobench/verl-openr1:v0.4

docker pull anonymous-koco/koco-tensorrt:1.0
docker tag anonymous-koco/koco-tensorrt:1.0 tensorrt:latest

docker pull anonymous-koco/koco-raganything-smolagents:1.0
docker tag anonymous-koco/koco-raganything-smolagents:1.0 raganything-smolagents:test
```

## Experiment

Configure one run before executing any stage:

```bash
export MODEL=deepseek/deepseek-v3.2
export PROFILE=deepseek_v3_2_full
export FRAMEWORKS="verl open-r1 raganything smolagents tensorrt_model_optimizer"
```

For a focused run:

```bash
export FRAMEWORKS="verl"
export PRAXIS_EXAMPLES="prime"
export PROFILE=deepseek_v3_2_verl_prime
```

Keep the same model, profile, frameworks, and examples when resuming an
experiment. Set `PRAXIS_RUN_PROFILE` explicitly only when resuming artifacts
created under an explicit run profile.

## Run

Run each stage after the previous stage finishes:

```bash
# Stage 1
bash memory/scripts/in_domain_practice.sh

# Stage 2
bash memory/scripts/structured_knowledge_acquisition.sh

# Stage 3
bash memory/scripts/graph_grounded_organization.sh

# Stage 4
bash memory/scripts/tacit_knowledge_injection.sh
```

Stage 1 and Stage 4 reuse completed work. Re-run an interrupted stage with the
same experiment configuration; see the stage-specific documentation for exact
resume and rebuild behavior.

Optional online evolution after Stage 4:

```bash
bash memory/scripts/online_knowledge_evolution.sh
```

## Details

- [Stage 1: observed memory](praxis/domain_code_generation/scripts/koco_openhands/memory/observed_memory/README.md)
- [Stage 2: procedural memory](praxis/domain_code_generation/scripts/koco_openhands/memory/procedural_memory/README.md)
- [Stages 3, 4, and online evolution](praxis/domain_code_generation/scripts/koco_openhands/memory/README.md)
