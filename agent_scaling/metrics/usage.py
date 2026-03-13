from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AIMessage
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse

from .artifacts import LLMCallUsage


def _get_usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def extract_usage_from_model_response(response: ModelResponse) -> LLMCallUsage:
    usage = getattr(response, "usage", None)
    input_tokens = _get_usage_value(usage, "prompt_tokens")
    output_tokens = _get_usage_value(usage, "completion_tokens")
    total_tokens = _get_usage_value(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    try:
        cost_usd = float(completion_cost(response)) if response is not None else 0.0
    except Exception:
        cost_usd = 0.0
    model = getattr(response, "model", None)
    return LLMCallUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        model=model,
    )


def extract_usage_from_ai_message(message: AIMessage) -> Optional[LLMCallUsage]:
    response = message.response_metadata.get("litellm_response")
    if response is None:
        return None
    return extract_usage_from_model_response(response)
