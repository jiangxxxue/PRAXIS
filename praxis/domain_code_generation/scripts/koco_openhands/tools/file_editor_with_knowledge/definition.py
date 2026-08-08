"""File editor tool wrapper that injects graph node knowledge on `view`.

Subclasses ``FileEditorTool`` and re-registers it under the same name so that
``Tool(name="file_editor")`` resolves to this version.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from openhands.sdk.tool import (
    ToolAnnotations,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.schema import TextContent
from openhands.tools.file_editor.definition import (
    FileEditorAction,
    FileEditorObservation,
    FileEditorTool,
    TOOL_DESCRIPTION,
)

from tools._graph_knowledge_inject import inject_for_locations, make_graph_retriever

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState

GRAPH_KNOWLEDGE_TITLE = "GRAPH KNOWLEDGE FOR THIS FILE:"
GRAPH_KNOWLEDGE_MAX_NODES = 2
GRAPH_KNOWLEDGE_MAX_ITEMS_PER_NODE = 3
LARGE_LINE = 10**9


class _FileEditorWithKnowledgeExecutor(ToolExecutor):
    """Wrap the default ``FileEditorExecutor`` and append graph knowledge on view."""

    def __init__(
        self,
        inner,
        graph_knowledge_path: str | None = None,
        graph_knowledge_format: str = "trigger_content",
        graph_knowledge_min_confidence: float | None = None,
    ):
        self._inner = inner
        self.graph_retriever = make_graph_retriever(
            graph_knowledge_path,
            min_confidence=graph_knowledge_min_confidence,
        )
        self.graph_knowledge_format = graph_knowledge_format
        self._seen_node_keys: set[str] = set()

    def __call__(
        self,
        action: FileEditorAction,
        conversation: "LocalConversation | None" = None,
    ) -> FileEditorObservation:
        result: FileEditorObservation = self._inner(action, conversation)

        if (
            self.graph_retriever is None
            or result.is_error
            or action.command != "view"
            or not action.path
        ):
            return result

        view_range = action.view_range or []
        start_line = int(view_range[0]) if len(view_range) >= 1 and view_range[0] else 1
        end_line = (
            int(view_range[1])
            if len(view_range) >= 2 and view_range[1] and view_range[1] != -1
            else LARGE_LINE
        )

        locations = [{
            "path": action.path,
            "start_line": start_line,
            "end_line": end_line,
        }]

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


class FileEditorWithKnowledgeTool(FileEditorTool):
    """``file_editor`` that appends graph node knowledge to ``view`` outputs."""

    name = "file_editor"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        graph_knowledge_path: str | None = None,
        graph_knowledge_format: str = "trigger_content",
        graph_knowledge_min_confidence: float | None = None,
        **kwargs,
    ) -> Sequence["FileEditorWithKnowledgeTool"]:
        from openhands.tools.file_editor.impl import FileEditorExecutor

        inner = FileEditorExecutor(workspace_root=conv_state.workspace.working_dir)
        executor = _FileEditorWithKnowledgeExecutor(
            inner,
            graph_knowledge_path=graph_knowledge_path,
            graph_knowledge_format=graph_knowledge_format,
            graph_knowledge_min_confidence=graph_knowledge_min_confidence,
        )

        # Reuse the upstream description-construction logic.
        description_lines = TOOL_DESCRIPTION.split("\n")
        base_description = "\n".join(description_lines[:2])
        remaining_description = "\n".join(description_lines[2:])
        if conv_state.agent.llm.vision_is_active():
            tool_description = (
                f"{base_description}\n"
                "* If `path` is an image file (.png, .jpg, .jpeg, .gif, .webp, "
                ".bmp), `view` displays the image content\n"
                f"{remaining_description}"
            )
        else:
            tool_description = TOOL_DESCRIPTION
        working_dir = conv_state.workspace.working_dir
        enhanced_description = (
            f"{tool_description}\n\n"
            f"Your current working directory is: {working_dir}\n"
            f"When exploring project structure, start with this directory "
            f"instead of the root filesystem."
        )

        return [
            cls(
                action_type=FileEditorAction,
                observation_type=FileEditorObservation,
                description=enhanced_description,
                annotations=ToolAnnotations(
                    title="file_editor",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# Override the upstream ``file_editor`` registration with our wrapper.
register_tool("file_editor", FileEditorWithKnowledgeTool)
