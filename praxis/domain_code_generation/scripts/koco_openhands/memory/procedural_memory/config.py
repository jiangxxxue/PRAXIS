from pathlib import Path

from config import get_docker_image

from ..config import (
    KOCO_OPENHANDS_DIR,
    MEMORY_RUN_ID,
    OBSERVED_KNOWLEDGE_DIR,
    PRACTICE_KNOWLEDGE_DIR,
    PROJECT_ROOT,
    SCRIPTS_DIR,
    candidates_path,
    requirement_path,
    test_input_path,
)

REPO_ROOT = PROJECT_ROOT.parents[1]
PROCEDURAL_MEMORY_DIR = KOCO_OPENHANDS_DIR / "memory" / "procedural_memory"
PROCEDURAL_DERIVED_DIR = PRACTICE_KNOWLEDGE_DIR
TRACES_DIR = PROCEDURAL_DERIVED_DIR / "_traces"
STRUCTURED_KNOWLEDGE_DIR = PRACTICE_KNOWLEDGE_DIR

DEFAULT_PROFILE = "default"
DEFAULT_DERIVED_SET = DEFAULT_PROFILE


def output_profile_for_set(profile: str | None = None) -> str:
    """Return the procedural-memory output profile name."""
    return profile or DEFAULT_PROFILE


def derived_dir_for_set(derived_set: str | None = None) -> Path:
    """Return the observed-memory root used as procedural-memory input."""
    return OBSERVED_KNOWLEDGE_DIR / MEMORY_RUN_ID if MEMORY_RUN_ID else OBSERVED_KNOWLEDGE_DIR


def trace_function_dir(framework: str, example: str, function_name: str,
                       derived_set: str | None = None) -> Path:
    """Return the trace directory for a function, namespaced by profile."""
    key = output_profile_for_set(derived_set)
    return TRACES_DIR / key / framework / example / function_name


def oracle_path_for_spec(spec: dict) -> Path:
    return trace_function_dir(
        spec["framework"],
        spec["example"],
        spec["function_name"],
        spec["derived_set"],
    ) / "oracle.json"


def trace_path_for_spec(spec: dict) -> Path:
    return trace_function_dir(
        spec["framework"],
        spec["example"],
        spec["function_name"],
        spec["derived_set"],
    ) / "trace.json"


def structured_profile_dir(example: str, profile: str | None = None,
                           framework: str | None = None) -> Path:
    key = output_profile_for_set(profile)
    if framework:
        return STRUCTURED_KNOWLEDGE_DIR / key / framework / example
    return STRUCTURED_KNOWLEDGE_DIR / key / example


def structured_per_function_dir(example: str, profile: str | None = None,
                                framework: str | None = None) -> Path:
    return structured_profile_dir(example, profile, framework) / "per_function"


def structured_per_function_path(example: str, function_name: str,
                                 profile: str | None = None,
                                 framework: str | None = None) -> Path:
    return structured_per_function_dir(example, profile, framework) / f"{function_name}.jsonl"


def structured_knowledge_path(example: str, profile: str | None = None,
                              framework: str | None = None) -> Path:
    return structured_profile_dir(example, profile, framework) / "practice_knowledge.jsonl"


DOCKER_IMAGE = get_docker_image("verl")
