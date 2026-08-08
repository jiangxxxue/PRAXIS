"""Path configuration for koco_openhands memory artifacts."""

import json
import os
from pathlib import Path

KOCO_OPENHANDS_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = KOCO_OPENHANDS_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

MEMORY_DIR = KOCO_OPENHANDS_DIR / "memory"
DERIVED_DIR = MEMORY_DIR / "derived"
OBSERVED_KNOWLEDGE_DIR = DERIVED_DIR / "observed_knowledge"
STATIC_MEMORY_DIR = OBSERVED_KNOWLEDGE_DIR
GRAPH_KNOWLEDGE_DIR = DERIVED_DIR / "graph_knowledge"
PROCEDURAL_KNOWLEDGE_DIR = DERIVED_DIR / "procedural_knowledge"
PRACTICE_KNOWLEDGE_DIR = PROCEDURAL_KNOWLEDGE_DIR
PROMPTS_DIR = MEMORY_DIR / "prompts"
MEMORY_RUN_ID = os.environ.get("PRAXIS_MEMORY_RUN_ID", "").strip()


def _run_scoped(root: Path) -> Path:
    return root / MEMORY_RUN_ID if MEMORY_RUN_ID else root


def observed_knowledge_path(framework: str, example: str) -> Path:
    return _run_scoped(OBSERVED_KNOWLEDGE_DIR) / framework / f"{example}.md"


def static_memory_path(framework: str, example: str) -> Path:
    return observed_knowledge_path(framework, example)


def observed_example_dir(framework: str, example: str) -> Path:
    return _run_scoped(OBSERVED_KNOWLEDGE_DIR) / framework / example


def graph_knowledge_example_dir(framework: str, example: str) -> Path:
    return _run_scoped(GRAPH_KNOWLEDGE_DIR) / framework / example


def candidates_path(framework: str, example: str) -> Path:
    return observed_example_dir(framework, example) / "candidates.json"


def requirement_path(framework: str, example: str, function_name: str) -> Path:
    return observed_example_dir(framework, example) / f"{function_name}_requirement.md"


def test_input_path(framework: str, example: str, function_name: str) -> Path:
    return observed_example_dir(framework, example) / f"{function_name}_test_input.py"


def coverage_result_path(framework: str, example: str, function_name: str) -> Path:
    return observed_example_dir(framework, example) / f"{function_name}_coverage.json"


def feedback_log_path(framework: str, example: str, function_name: str) -> Path:
    return observed_example_dir(framework, example) / f"{function_name}_feedback_log.json"


def generate_status_path(framework: str, example: str, function_name: str) -> Path:
    return observed_example_dir(framework, example) / f"{function_name}_generate_status.json"


def generate_log_path(framework: str, example: str, function_name: str) -> Path:
    return observed_example_dir(framework, example) / f"{function_name}_generate_log.json"


def feedback_status_path(framework: str, example: str, function_name: str) -> Path:
    return observed_example_dir(framework, example) / f"{function_name}_feedback_status.json"


def dep_graph_path(framework: str, example: str) -> Path:
    return graph_knowledge_example_dir(framework, example) / "dep_graph.json"


def experiences_path(framework: str, example: str) -> Path:
    return observed_example_dir(framework, example) / "experiences.json"


def practice_knowledge_root() -> Path:
    return PRACTICE_KNOWLEDGE_DIR


def code_dir(framework: str, example: str) -> Path:
    return PROJECT_ROOT / framework / "test_examples" / example / "code"


def prompt_template(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def get_metadata_path(framework: str) -> Path:
    return PROJECT_ROOT / framework / "knowledge_corpus" / "metadata.json"


def ensure_input_data(framework: str, example: str) -> bool:
    """Run parse + prompt construction for a single test example."""
    from parse_algorithm_methods import parse_markdown_file, process_functions
    from prompts_construction import add_prompts_to_data

    input_md = (
        PROJECT_ROOT / framework / "test_examples" / example
        / "requirements" / "03_algorithm_and_core_methods.md"
    )
    code_base = PROJECT_ROOT / framework / "test_examples" / example / "code"
    test_base = code_base / "tests"
    output_dir = SCRIPTS_DIR / "data" / framework
    output_file = output_dir / f"algorithm_methods_data_{example}.jsonl"

    if not input_md.exists():
        print(f"    Step 1: markdown not found ({input_md.name})")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        functions_data = parse_markdown_file(str(input_md))
        processed = process_functions(functions_data, str(code_base), str(test_base))
        with output_file.open("w", encoding="utf-8") as fh:
            for func in processed:
                fh.write(json.dumps(func, ensure_ascii=False) + "\n")
        print(f"    Step 1: parsed {len(processed)} functions")
    except Exception as exc:
        print(f"    Step 1 failed: {exc}")
        return False

    metadata_path = get_metadata_path(framework)
    try:
        add_prompts_to_data(str(output_file), str(output_file), str(metadata_path))
        print("    Step 2: prompts added")
    except Exception as exc:
        print(f"    Step 2 failed: {exc}")
        return False

    return output_file.exists()
