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


def _get_obj_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _get_latency_ms(response: Any) -> float:
    direct = _get_obj_value(response, "_response_ms")
    if direct is not None:
        try:
            return float(direct)
        except Exception:
            pass
    hidden = _get_obj_value(response, "_hidden_params")
    if hidden is not None:
        hidden_ms = _get_obj_value(hidden, "_response_ms")
        if hidden_ms is None:
            hidden_ms = _get_obj_value(hidden, "response_ms")
        if hidden_ms is not None:
            try:
                return float(hidden_ms)
            except Exception:
                pass
    return 0.0


def _get_float_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _get_provider_cost_usd(usage: Any) -> float:
    direct = _get_float_value(_get_obj_value(usage, "cost"))
    if direct is not None:
        return direct

    details = _get_obj_value(usage, "cost_details")
    if details is not None:
        for key in ("upstream_inference_cost", "total_cost", "cost"):
            val = _get_float_value(_get_obj_value(details, key))
            if val is not None:
                return val
    return 0.0


def _get_llm_provider(response: Any) -> Optional[str]:
    hidden = _get_obj_value(response, "_hidden_params")
    provider = _get_obj_value(hidden, "custom_llm_provider")
    if provider is None:
        return None
    provider_str = str(provider).strip().lower()
    return provider_str if provider_str else None


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
    provider_cost_usd = _get_provider_cost_usd(usage)
    llm_provider = _get_llm_provider(response)
    openrouter_cost_usd = (
        provider_cost_usd
        if llm_provider is not None and "openrouter" in llm_provider
        else 0.0
    )
    model = getattr(response, "model", None)
    latency_ms = _get_latency_ms(response)
    return LLMCallUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        provider_cost_usd=provider_cost_usd,
        openrouter_cost_usd=openrouter_cost_usd,
        latency_ms=latency_ms,
        llm_provider=llm_provider,
        model=model,
    )


def extract_usage_from_ai_message(message: AIMessage) -> Optional[LLMCallUsage]:
    response = message.response_metadata.get("litellm_response")
    if response is None:
        return None
    return extract_usage_from_model_response(response)
