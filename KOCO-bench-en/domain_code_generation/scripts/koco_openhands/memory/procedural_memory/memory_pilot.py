#!/usr/bin/env python3
"""
Procedural memory pipeline — plumbing pilot.

Thin entry point that sets up sys.path automatically so no PYTHONPATH
export is needed.  All logic lives in koco_openhands/memory/procedural_memory/run_pilot.py.

Setup (first time only):
    cd KOCO-bench-en/domain_code_generation/scripts/koco_openhands
    uv sync
    # then fill in your API key (scripts/.env, shared with cli.py):
    cp ../.env.example ../.env   # if .env doesn't exist yet
    # edit ../.env → set OPENROUTER_API_KEY=sk-or-...
    # for token-plan Qwen, set BAILIAN_TOKEN_PLAN_API_KEY=sk-sp-...
    # scripts/.env.local is also loaded and overrides scripts/.env.

Usage:
    cd KOCO-bench-en/domain_code_generation/scripts/koco_openhands

    # init reads memory/derived/observed_knowledge and runs GT in Docker to produce
    # an oracle for each candidate. Omit --function to init every candidate;
    # pass --function to init only one.
    uv run python memory/procedural_memory/memory_pilot.py init \\
        --framework raganything --example BookWorm

    # Subsequent steps take --framework, --example, and --function to pick a spec.
    uv run python memory/procedural_memory/memory_pilot.py grade-smoke \\
        --framework raganything --example BookWorm --function _extract_text

    # --K is an upper bound; practice stops early on first PASS.
    uv run python memory/procedural_memory/memory_pilot.py practice \\
        --framework raganything --example BookWorm --function _extract_text \\
        --K 8 --model deepseek/deepseek-v3.2

    # structured per-function distill is independent; run many in parallel.
    uv run python memory/procedural_memory/memory_pilot.py distill-structured \\
        --framework raganything --example BookWorm --function _extract_text \\
        --model deepseek/deepseek-v3.2

    # after structured distill of all functions in this example, consolidate
    # this model profile's structured memory.
    uv run python memory/procedural_memory/memory_pilot.py consolidate-structured \\
        --framework raganything --example BookWorm --profile deepseek_v3_2

Prerequisites:
    - uv installed
    - scripts/.env with the provider key set, e.g. OPENROUTER_API_KEY or
      BAILIAN_TOKEN_PLAN_API_KEY
    - Docker with GPU support (docker run --gpus all)
    - Image: kocobench/verl-openr1:v0.4
    - memory/derived/observed_knowledge populated with candidates.json,
      {func}_requirement.md, and {func}_test_input.py

Output:
    koco_openhands/memory/derived/procedural_knowledge/_traces/{profile}/{framework}/{example}/{f}/pilot_spec.json
        — per-function spec
    koco_openhands/memory/derived/procedural_knowledge/_traces/{profile}/{framework}/{example}/{f}/oracle.json
        — GT-on-cases
    koco_openhands/memory/derived/procedural_knowledge/_traces/{profile}/{framework}/{example}/{f}/trace.json
        — practice trace
    koco_openhands/memory/derived/procedural_knowledge/{profile}/{framework}/{example}/per_function/{f}.jsonl
        — structured per-function distill output
    koco_openhands/memory/derived/procedural_knowledge/{profile}/{framework}/{example}/practice_knowledge.jsonl
        — consolidated structured knowledge file
"""
import sys
from pathlib import Path

# procedural_memory/ → memory/ → koco_openhands/ → scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from koco_openhands.memory.procedural_memory.run_pilot import main

main()
