"""Terminal tool wrapper that injects graph node knowledge after grep -n / rg -n.

Subclasses ``TerminalTool`` and re-registers it under the same name so that
``Tool(name="terminal", params={...})`` resolves to this version.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from openhands.sdk.tool import (
    ToolAnnotations,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.schema import TextContent
from openhands.tools.terminal.definition import (
    TOOL_DESCRIPTION,
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)
from openhands.tools.terminal.impl import TerminalExecutor
from openhands.tools.terminal.terminal.factory import create_terminal_session

from tools._graph_knowledge_inject import inject_for_locations, make_graph_retriever

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState

GRAPH_KNOWLEDGE_TITLE = "GRAPH KNOWLEDGE FOR MATCHED LINES:"
GRAPH_KNOWLEDGE_MAX_NODES = 3
GRAPH_KNOWLEDGE_MAX_ITEMS_PER_NODE = 3
MAX_LOCATIONS = 5

# Detect that the user issued a line-numbered grep/rg search worth scanning.
_LINE_NUMBER_TRIGGER_RE = re.compile(r"\b(grep|rg)\b[^|;&]*?(?:-n\b|--line-number\b)")
# Match ``path:line:content`` (or ``path-line-content`` from rg context); restrict
# to *.py source so we don't match arbitrary numeric output.
_GREP_LINE_RE = re.compile(r"^([^:\s][^:\n]*?\.py)[:\-](\d+)[:\-]")
# Bare ``line:content`` rows produced when grep -n searches a single .py file
# (no path prefix is emitted in that case).
_BARE_LINE_RE = re.compile(r"^(\d+)[:\-]")
# Pull out a single .py target from the grep/rg command itself; used as a
# fallback when the output rows don't carry a path prefix.
_CMD_PY_PATH_RE = re.compile(r"(?<![\w/\-])(/?[\w./\-]+\.py)\b")


class _StableTerminalExecutor(TerminalExecutor):
    """TerminalExecutor that preserves the requested backend across reset."""

    def __init__(
        self,
        *args,
        terminal_type: Literal["tmux", "subprocess"] | None = None,
        **kwargs,
    ):
        self._stable_terminal_type = terminal_type
        super().__init__(*args, terminal_type=terminal_type, **kwargs)

    def reset(self) -> TerminalObservation:
        """Reset without letting the SDK auto-detect tmux after subprocess runs."""

        original_work_dir = self.session.work_dir
        original_username = self.session.username
        original_no_change_timeout = self.session.no_change_timeout_seconds

        self.session.close()
        self.session = create_terminal_session(
            work_dir=original_work_dir,
            username=original_username,
            no_change_timeout_seconds=original_no_change_timeout,
            terminal_type=self._stable_terminal_type,
            shell_path=self.shell_path,
        )
        self.session.initialize()

        return TerminalObservation.from_text(
            text=(
                "Terminal session has been reset. All previous environment "
                "variables and session state have been cleared."
            ),
            command="[RESET]",
            exit_code=0,
        )


def _is_line_number_search(command: str | None) -> bool:
    if not command:
        return False
    return bool(_LINE_NUMBER_TRIGGER_RE.search(command))


def _fallback_py_target(command: str) -> str | None:
    """Return the single .py path argument of a grep/rg command, if any."""
    matches = _CMD_PY_PATH_RE.findall(command or "")
    # Only return a fallback when there's exactly one .py target — multiple
    # candidates make the bare ``line:`` rows ambiguous.
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_locations(text: str, working_dir: str | None,
                       command: str | None = None) -> list[dict]:
    locations: list[dict] = []
    seen: set[tuple[str, int]] = set()
    bare_path: str | None = None  # resolved lazily

    def _absolutize(path: str) -> str:
        if working_dir and not os.path.isabs(path):
            return os.path.normpath(os.path.join(working_dir, path))
        return path

    for raw_line in text.splitlines():
        match = _GREP_LINE_RE.match(raw_line)
        path: str | None
        line_str: str
        if match:
            path, line_str = match.group(1), match.group(2)
        else:
            bare = _BARE_LINE_RE.match(raw_line)
            if not bare:
                continue
            if bare_path is None:
                fallback = _fallback_py_target(command or "")
                bare_path = fallback or ""
            if not bare_path:
                continue
            path, line_str = bare_path, bare.group(1)
        try:
            lineno = int(line_str)
        except ValueError:
            continue
        absolute = _absolutize(path)
        key = (absolute, lineno)
        if key in seen:
            continue
        seen.add(key)
        locations.append({
            "path": absolute,
            "start_line": lineno,
            "end_line": lineno,
        })
        if len(locations) >= MAX_LOCATIONS:
            break
    return locations


class _TerminalWithKnowledgeExecutor(ToolExecutor):
    """Wrap the default ``TerminalExecutor`` and append graph knowledge for grep -n / rg -n hits."""

    def __init__(self, inner, graph_knowledge_path: str | None = None,
                 working_dir: str | None = None,
                 graph_knowledge_format: str = "trigger_content"):
        self._inner = inner
        self._working_dir = working_dir
        self.graph_retriever = make_graph_retriever(graph_knowledge_path)
        self.graph_knowledge_format = graph_knowledge_format
        self._seen_node_keys: set[str] = set()

    def __call__(
        self,
        action: TerminalAction,
        conversation: "LocalConversation | None" = None,
    ) -> TerminalObservation:
        result: TerminalObservation = self._inner(action, conversation)

        if (
            self.graph_retriever is None
            or result.is_error
            or not _is_line_number_search(action.command)
        ):
            return result

        locations = _extract_locations(result.text, self._working_dir, action.command)
        if not locations:
            return result

        formatted, _ = inject_for_locations(
            self.graph_retriever,
            locations,
            title=GRAPH_KNOWLEDGE_TITLE,
            max_nodes=GRAPH_KNOWLEDGE_MAX_NODES,
            max_items_per_node=GRAPH_KNOWLEDGE_MAX_ITEMS_PER_NODE,
            knowledge_format=self.graph_knowledge_format,
            seen_node_keys=self._seen_node_keys,
        )
        if formatted:
            new_content = list(result.content) + [TextContent(text="\n\n" + formatted)]
            result = result.model_copy(update={"content": new_content})
        return result


class TerminalWithKnowledgeTool(TerminalTool):
    """``terminal`` that appends graph node knowledge after grep -n / rg -n searches."""

    name = "terminal"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        username: str | None = None,
        no_change_timeout_seconds: int | None = None,
        terminal_type: Literal["tmux", "subprocess"] | None = None,
        shell_path: str | None = None,
        executor: ToolExecutor | None = None,
        graph_knowledge_path: str | None = None,
        graph_knowledge_format: str = "trigger_content",
        **kwargs,
    ) -> Sequence["TerminalWithKnowledgeTool"]:
        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        stable_terminal_type = terminal_type or "subprocess"
        if executor is None:
            inner = _StableTerminalExecutor(
                working_dir=working_dir,
                username=username,
                no_change_timeout_seconds=no_change_timeout_seconds,
                terminal_type=stable_terminal_type,
                shell_path=shell_path,
                full_output_save_dir=conv_state.env_observation_persistence_dir,
            )
        else:
            inner = executor

        wrapped = _TerminalWithKnowledgeExecutor(
            inner,
            graph_knowledge_path=graph_knowledge_path,
            working_dir=working_dir,
            graph_knowledge_format=graph_knowledge_format,
        )

        return [
            cls(
                action_type=TerminalAction,
                observation_type=TerminalObservation,
                description=TOOL_DESCRIPTION,
                annotations=ToolAnnotations(
                    title="terminal",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=wrapped,
            )
        ]


# Override the upstream ``terminal`` registration with our wrapper.
register_tool("terminal", TerminalWithKnowledgeTool)
