import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages


TRACE_SCHEMA_VERSION = "agent-scaling.trace.v1"


def _json_safe(value: Any) -> Any:
    """Convert common LangChain/LiteLLM/Pydantic objects into JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseMessage):
        return _json_safe(convert_to_openai_messages(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    return str(value)


def normalize_messages(messages: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized = []
    for message in messages:
        converted = _json_safe(message)
        if isinstance(converted, dict):
            normalized.append(converted)
        else:
            normalized.append({"role": "unknown", "content": converted})
    return normalized


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    tool_calls = getattr(response, "tool_calls", None) or []
    return _json_safe(tool_calls)


def extract_response_metadata(response: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    response_metadata = getattr(response, "response_metadata", {}) or {}
    litellm_response = response_metadata.get("litellm_response")

    if litellm_response is not None:
        metadata["model"] = getattr(litellm_response, "model", None)
        usage = getattr(litellm_response, "usage", None)
        if usage is not None:
            metadata["usage"] = _json_safe(usage)
        hidden = getattr(litellm_response, "_hidden_params", {}) or {}
        metadata["cache_hit"] = bool(hidden.get("cache_hit", False))
        if hidden.get("response_cost") is not None:
            metadata["cost"] = hidden.get("response_cost")

    if not metadata and response_metadata:
        metadata["response_metadata"] = _json_safe(response_metadata)

    return metadata


def make_trace_event(
    event_type: str,
    *,
    instance_idx: Optional[int] = None,
    agent_id: str = "main",
    step: Optional[int] = None,
    round_num: Optional[int] = None,
    iteration: Optional[int] = None,
    role: Optional[str] = None,
    content: Optional[Any] = None,
    messages: Optional[Iterable[Any]] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    observation: Optional[Any] = None,
    status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "agent_id": agent_id,
    }
    optional_fields = {
        "instance_idx": instance_idx,
        "step": step,
        "round": round_num,
        "iteration": iteration,
        "role": role,
        "content": content,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "observation": observation,
        "status": status,
        "metadata": metadata,
    }
    for key, value in optional_fields.items():
        if value is not None:
            event[key] = _json_safe(value)
    if messages is not None:
        event["messages"] = normalize_messages(messages)
        event["message_count"] = len(event["messages"])
    return event


def write_trace_events(events: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(_json_safe(event), ensure_ascii=False) + "\n")


def response_text(response: Any) -> str:
    if hasattr(response, "text"):
        try:
            return str(response.text())
        except Exception:
            pass
    return str(getattr(response, "content", ""))


def single_agent_response_events(
    *,
    instance_idx: Optional[int],
    step: int,
    messages: Iterable[Any],
    response: Any,
) -> List[Dict[str, Any]]:
    tool_calls = extract_tool_calls(response)
    return [
        make_trace_event(
            "llm_input",
            instance_idx=instance_idx,
            step=step,
            messages=messages,
        ),
        make_trace_event(
            "llm_response",
            instance_idx=instance_idx,
            step=step,
            role="assistant",
            content=response_text(response),
            metadata={
                **extract_response_metadata(response),
                "tool_calls": tool_calls,
            },
        ),
    ]


def conversation_history_events(
    conversation: Any,
    *,
    instance_idx: Optional[int],
    default_agent_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    agent_id = default_agent_id or getattr(conversation, "agent_id", "unknown")

    for response in getattr(conversation, "messages", []) or []:
        litellm_message = getattr(response, "litellm_message", None)
        tag = getattr(response, "tag", None)
        content = ""
        if litellm_message is not None:
            try:
                content = litellm_message.choices[0].message.content or ""
            except Exception:
                content = str(litellm_message)
        events.append(
            make_trace_event(
                "llm_response",
                instance_idx=instance_idx,
                agent_id=agent_id,
                role="assistant",
                content=content,
                metadata={
                    "tag": tag,
                    "litellm_response": _json_safe(litellm_message),
                },
            )
        )

    for external in getattr(conversation, "external_comms", []) or []:
        role = getattr(external, "role", None)
        events.append(
            make_trace_event(
                "coordination_message",
                instance_idx=instance_idx,
                agent_id=agent_id,
                round_num=getattr(external, "round_num", None),
                role=role,
                content=getattr(external, "message", ""),
            )
        )

    for round_messages in getattr(conversation, "internal_comms", []) or []:
        for turn in round_messages:
            message = getattr(turn, "message", {}) or {}
            role = message.get("role", getattr(turn, "role", None))
            content = message.get("content", "")
            event_type = "agent_message"
            tool_name = None
            tool_args = None
            observation = None
            metadata: Dict[str, Any] = {"message": message}

            if role == "assistant":
                event_type = "llm_response"
                metadata["tool_calls"] = message.get("tool_calls", [])
            elif role == "tool":
                event_type = "tool_observation"
                observation = content
                tool_name = message.get("name")
            elif role == "user" and str(content).startswith("ERROR:"):
                event_type = "error"

            events.append(
                make_trace_event(
                    event_type,
                    instance_idx=instance_idx,
                    agent_id=agent_id,
                    round_num=getattr(turn, "round_num", None),
                    iteration=getattr(turn, "iteration_num", None),
                    role=role,
                    content=content,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    metadata=metadata,
                )
            )
    return events


def subagent_trace_events(
    subagents: Dict[str, Any],
    *,
    instance_idx: Optional[int],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for agent_id, agent in subagents.items():
        conversation = getattr(agent, "conv_history", None)
        if conversation is not None:
            events.extend(
                conversation_history_events(
                    conversation,
                    instance_idx=instance_idx,
                    default_agent_id=agent_id,
                )
            )
    return events


def orchestration_trace_events(
    result: Any,
    *,
    instance_idx: Optional[int],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    lead = getattr(result, "lead_agent_conversation", None)
    if lead is not None:
        events.extend(
            conversation_history_events(
                lead,
                instance_idx=instance_idx,
                default_agent_id="lead_agent",
            )
        )

    for agent_id, conversation in (
        getattr(result, "subagent_conversations", {}) or {}
    ).items():
        events.extend(
            conversation_history_events(
                conversation,
                instance_idx=instance_idx,
                default_agent_id=agent_id,
            )
        )

    synthesized_answer = getattr(result, "synthesized_answer", None)
    if synthesized_answer is not None:
        events.append(
            make_trace_event(
                "final_answer",
                instance_idx=instance_idx,
                agent_id="lead_agent",
                content=synthesized_answer,
            )
        )
    return events
