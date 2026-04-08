import os
from typing import Dict, Literal, Optional

import litellm
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from agent_scaling.llm.litellm_lc import ChatLiteLLMLC


class LLMParams(BaseModel):
    """
    see litellm.completion() for parameter details
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    # cache controls: https://docs.litellm.ai/docs/proxy/caching#dynamic-cache-controls
    cache: Optional[Dict[Literal["no-cache", "no-store"], bool]] = None

    @model_validator(mode="after")
    def check_cache(self) -> Self:
        if not self.cache:
            self.cache = {}
        return self


def _is_minimax_base_url(base_url: str) -> bool:
    """Check if ANTHROPIC_BASE_URL points to a non-Anthropic host (e.g. Minimax)."""
    return bool(base_url) and "anthropic.com" not in base_url


class LLMConfig(BaseModel):
    """
    use litellm.get_valid_models() to see model names

    Supported provider prefixes:
    - openrouter/ : Routes through OpenRouter (uses OPENROUTER_API_KEY)
    - minimax/    : Routes through Minimax Anthropic-compatible API
                    (uses ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY)
    """

    params: LLMParams
    model: str = "gemini/gemini-2.0-flash"

    def _resolve_provider_kwargs(self) -> Dict:
        """Resolve provider-specific api_key and api_base for LiteLLM."""
        extra: Dict = {}

        if self.model.startswith("openrouter/"):
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable is required for openrouter/ models"
                )
            extra["api_key"] = api_key
            extra["api_base"] = "https://openrouter.ai/api/v1"

        elif self.model.startswith("minimax/"):
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is required for minimax/ models"
                )
            if not _is_minimax_base_url(base_url):
                base_url = "https://api.minimax.io/anthropic"
            extra["api_key"] = api_key
            extra["api_base"] = base_url
            # Route through LiteLLM's anthropic/ prefix
            extra["_model_override"] = self.model.replace("minimax/", "anthropic/", 1)

        return extra

    def get_llm(self) -> ChatLiteLLMLC:
        provider_kwargs = self._resolve_provider_kwargs()
        model_name = provider_kwargs.pop("_model_override", self.model)
        return ChatLiteLLMLC(
            model=model_name,
            max_retries=3,
            **self.params.model_dump(exclude={"cache"}),
            **provider_kwargs,
        )
