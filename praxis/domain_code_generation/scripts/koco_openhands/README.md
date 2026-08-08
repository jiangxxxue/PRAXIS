# koco_openhands

OpenHands SDK agent evaluation pipeline for KOCO-bench domain code generation. Runs an AI agent that explores a framework repository, reads existing code, and implements target functions — then evaluates results via Docker execution.

## Prerequisites

- Python ≥ 3.12, < 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for evaluation step)

## Setup

```bash
cd praxis/domain_code_generation/scripts/koco_openhands

# install dependencies (auto-creates venv)
uv sync
```

Set API keys in `scripts/.env` (see `scripts/.env.example`):

```
OPENROUTER_API_KEY=sk-or-v1-...     # chat and embeddings via OpenRouter
```

By default, `knowledge_search` uses the same OpenRouter key with
`openai/text-embedding-3-small`. To use a separate embedding provider or key:

```
EMBEDDING_API_KEY=sk-...
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_MODEL=openai/text-embedding-3-small
```

When no embedding key is available, `knowledge_search` falls back to BM25-only
search. This only affects stages that enable `knowledge_search`; target
inference keeps that tool disabled.

## Usage

All commands run from this directory (`koco_openhands/`):

```bash
# agent inference: explore repo and generate implementations
python cli.py infer --framework verl --model deepseek/deepseek-v3.2

# single test example only
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --test-example ARES

# specific functions only
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --instance-ids compute_score

# discard previous results and re-run from scratch
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --force

# preserve workspace artifacts (logs, code snapshot, agent events) for debugging
python cli.py infer --framework verl --model deepseek/deepseek-v3.2 --debug

# evaluation only (Docker execution + metrics aggregation)
python cli.py eval --framework verl --model deepseek/deepseek-v3.2

# full pipeline: infer + eval (resumes by default)
python cli.py run --framework verl --model deepseek/deepseek-v3.2
```

### Key CLI Options

| Option | Default | Description |
|---|---|---|
| `--framework` | *(required)* | Framework name (e.g., `verl`) |
| `--model` | `deepseek/deepseek-v3.2` | OpenRouter model identifier |
| `--test-example` | all | Single test example name (e.g., `ARES`) |
| `--instance-ids` | all | Comma-separated function names |
| `--concurrency` | `10` | Number of concurrent agents |
| `--max-iterations` | `50` | Max agent turns per function |
| `--force` | `false` | Discard previous results and re-run |
| `--debug` | `false` | Preserve workspace artifacts on completion |
| `--graph-knowledge-min-confidence` | `0.6` | Filter graph knowledge below this confidence before prompt/tool injection |

### Pipeline Flow

1. **Parse** — Extract function specs from algorithm methods markdown → JSONL
2. **Prompt** — Build system context + user task prompts
3. **Infer** — For each target function:
   - Copy workspace to isolated temp dir and stub ground-truth function bodies
   - Run SDK agent with `terminal` and `file_editor` tools; target inference keeps `knowledge_search` disabled
   - Extract implementation from modified source file (fallback: scan agent events)
4. **Eval** — Run Docker execution evaluation and aggregate metrics

### Confidence-Aware Knowledge

Structured practice knowledge receives its final `confidence.score` during
distillation. The score estimates how reliable the extracted knowledge is when
its trigger matches, using the ground-truth implementation, practice outcomes,
and cited evidence. Higher-confidence knowledge is trusted more during later
graph construction and retrieval; uncertain knowledge may be filtered out.

- Graph propagation skips knowledge below `PRAXIS_CONFIDENCE_THRESHOLD`
  (default `0.6`).
- Duplicate knowledge uses noisy-or confidence aggregation.
- Conflict resolution uses evidence plus confidence and deterministically keeps
  the higher-confidence cluster when scores differ by at least `0.15`.
- Initial prompt injection and terminal/file-editor/knowledge-search injection
  filter by the same threshold and sort by descending confidence.
- Online target outcomes can reinforce or decay actually injected knowledge
  with `memory.procedural_memory.target_practice update-confidence`.

The in-domain practice coverage threshold defaults to `0.8` and can be
overridden with `PRAXIS_COVERAGE_THRESHOLD`.

### Knowledge Search Tool

Hybrid BM25 + semantic search over framework source code and documentation. Allows the agent to efficiently query the knowledge base instead of manually exploring files.

- **BM25** (weight 0.3): SQLite FTS5 keyword search
- **Semantic** (weight 0.7): Cosine similarity via `openai/text-embedding-3-small`
- Falls back to BM25-only when no embedding API key is available

## Output

Results are written to `scripts/data/{framework}/openhands/{model}/`:

```
algorithm_methods_data_{example}_output.jsonl   # inference results
algorithm_methods_data_{example}_result.jsonl    # evaluation results
algorithm_methods_data_{example}_result.metrics.json  # aggregated metrics
```

Debug artifacts (on failure or `--debug`): `scripts/data/{framework}/openhands/debug/{example}/{function_name}/`

## Tests

```bash
cd praxis/domain_code_generation/scripts/koco_openhands
uv run pytest test_cli.py test_runner.py tools/knowledge_search/test_knowledge_search.py -v
```
