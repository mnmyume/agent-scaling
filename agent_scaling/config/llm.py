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


class LLMConfig(BaseModel):
    """
    use litellm.get_valid_models() to see model names
    """

    params: LLMParams
    model: str = "gemini/gemini-2.0-flash"
    role_models: Dict[str, str] = {}

    def get_llm(self) -> ChatLiteLLMLC:
        return ChatLiteLLMLC(
            model=self.model, **self.params.model_dump(exclude={"cache"})
        )

    def get_llm_for_role(self, role: str) -> ChatLiteLLMLC:
        """Return a role-specific LLM, falling back to the global model."""
        model_name = self.role_models.get(role, self.model)
        return ChatLiteLLMLC(
            model=model_name, **self.params.model_dump(exclude={"cache"})
        )
