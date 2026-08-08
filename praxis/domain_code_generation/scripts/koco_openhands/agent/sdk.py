"""OpenHands SDK agent wrapper.

Provides a thin interface over the OpenHands SDK to run an agent in a
given workspace directory.  All SDK-specific imports and configuration
are isolated here so that the rest of the codebase stays decoupled.
"""

import os

try:
    from pydantic import SecretStr
    from openhands.sdk import (
        LLM, Agent, Conversation, Tool, ConversationExecutionStatus,
    )
    from openhands.sdk.conversation.exceptions import ConversationRunError
    from openhands.tools import register_default_tools
    register_default_tools()  # registers terminal, file_editor, etc. into SDK registry
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    # Re-export a sentinel so callers can reference the name without importing
    # the real class.
    ConversationExecutionStatus = None  # type: ignore[assignment,misc]
    ConversationRunError = Exception    # type: ignore[assignment,misc]


def _resolve_llm_model(model: str, base_url: str) -> str:
    """Add ``openrouter/`` prefix when the base URL is OpenRouter.

    litellm uses the model-name prefix for provider routing.  Without the
    ``openrouter/`` prefix, ``deepseek/…`` gets routed directly to the
    DeepSeek API, ignoring the custom base URL.
    """
    if "openrouter.ai" in base_url and not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 1 else default


def _stuck_detection_config() -> tuple[bool, dict[str, int]]:
    """Return PRAXIS-friendly OpenHands stuck-detection settings.

    OpenHands defaults are intentionally conservative. PRAXIS generation tasks
    often read the same small source region several times before writing the
    required artifact, so we keep stuck detection enabled but relax thresholds.
    """

    return (
        _env_bool("KOCO_OPENHANDS_STUCK_DETECTION", True),
        {
            "action_observation": _env_int(
                "KOCO_OPENHANDS_STUCK_ACTION_OBSERVATION", 8
            ),
            "action_error": _env_int("KOCO_OPENHANDS_STUCK_ACTION_ERROR", 4),
            "monologue": _env_int("KOCO_OPENHANDS_STUCK_MONOLOGUE", 5),
            "alternating_pattern": _env_int(
                "KOCO_OPENHANDS_STUCK_ALTERNATING_PATTERN", 10
            ),
        },
    )


def run_sdk_agent(prompt, workspace, model, api_key, base_url,
                  max_iterations=50, corpus_dirs=None,
                  graph_knowledge_path=None,
                  graph_knowledge_format="trigger_content",
                  graph_knowledge_min_confidence=None):
    """Run the OpenHands SDK agent and return (events, status).

    Creates an ephemeral LLM → Agent → Conversation pipeline, sends
    ``prompt``, and blocks until the agent finishes.

    Parameters:
        corpus_dirs: List of directories to index for knowledge search.
            When non-empty, the ``knowledge_search`` hybrid-search tool is
            registered and added to the agent's tool list.
        graph_knowledge_path: Path to a mounted/optimized graph knowledge JSON.
            When set, terminal and file-editor observations can append matching
            graph knowledge for surfaced code locations.
        graph_knowledge_format: Formatting mode for graph knowledge shown to
            the agent.
        graph_knowledge_min_confidence: Minimum confidence score injected into
            prompts and tool observations.

    Returns:
        (events, status) where *events* is a list of SDK event objects and
        *status* is a :class:`ConversationExecutionStatus` enum value.

    Raises:
        RuntimeError: If the SDK is not installed.
        ConversationRunError: If the agent encounters a fatal error.
    """
    if not SDK_AVAILABLE:
        raise RuntimeError(
            "openhands-sdk not installed. "
            "Install with: uv pip install openhands-sdk openhands-tools --python 3.12"
        )

    llm_model = _resolve_llm_model(model, base_url)

    llm = LLM(
        model=llm_model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0.0,
        max_output_tokens=65536,
    )

    terminal_params = {}
    file_editor_params = {}
    if graph_knowledge_path:
        # Importing these modules re-registers the default tool names with
        # graph-aware wrappers before the Conversation resolves Tool objects.
        import tools.terminal_with_knowledge  # noqa: F401
        import tools.file_editor_with_knowledge  # noqa: F401

        terminal_params = {
            "terminal_type": "subprocess",
            "graph_knowledge_path": str(graph_knowledge_path),
            "graph_knowledge_format": graph_knowledge_format,
            "graph_knowledge_min_confidence": graph_knowledge_min_confidence,
        }
        file_editor_params = {
            "graph_knowledge_path": str(graph_knowledge_path),
            "graph_knowledge_format": graph_knowledge_format,
            "graph_knowledge_min_confidence": graph_knowledge_min_confidence,
        }

    tools_list = [
        Tool(name="terminal", params=terminal_params),
        Tool(name="file_editor", params=file_editor_params),
    ]
    if corpus_dirs:
        import tools.knowledge_search  # noqa: F401 — triggers register_tool()
        tools_list.append(Tool(
            name="knowledge_search",
            params={
                "corpus_dirs": corpus_dirs,
                "graph_knowledge_path": (
                    str(graph_knowledge_path)
                    if graph_knowledge_path
                    else None
                ),
                "graph_knowledge_format": graph_knowledge_format,
                "graph_knowledge_min_confidence": graph_knowledge_min_confidence,
            },
        ))

    agent = Agent(
        llm=llm,
        tools=tools_list,
        include_default_tools=["FinishTool", "ThinkTool"],
    )

    stuck_detection, stuck_detection_thresholds = _stuck_detection_config()
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
        stuck_detection=stuck_detection,
        stuck_detection_thresholds=stuck_detection_thresholds,
        visualizer=None,
    )

    try:
        conversation.send_message(prompt)
        conversation.run()
    finally:
        events = list(conversation.state.events)
        status = conversation.state.execution_status
        conversation.close()

    return events, status
