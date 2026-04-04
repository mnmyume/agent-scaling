from __future__ import annotations

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall


def build_error_tool_message(
    tool_call: ToolCall,
    error: Exception,
    *,
    fallback_name: str = "",
) -> ToolMessage:
    """Build a tool response that preserves tool-call pairing on execution errors."""

    tool_name = tool_call.get("name") or fallback_name
    tool_call_id = tool_call.get("id")
    if not tool_call_id:
        raise ValueError(
            f"Tool call for {tool_name or 'unknown tool'} is missing an id; "
            "cannot create a matching ToolMessage."
        )

    return ToolMessage(
        content=(
            f"ERROR: Tool `{tool_name or 'unknown'}` failed with error: {error}. "
            "Please check the tool call arguments and try again."
        ),
        name=tool_name or None,
        tool_call_id=tool_call_id,
        status="error",
    )
