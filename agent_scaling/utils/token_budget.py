import threading
from typing import Tuple

from langchain_core.messages import AIMessage

from agent_scaling.logger import logger


class TokenBudgetManager:
    """
    Tracks a per-instance token budget reference shared across all LLM calls
    within a single agent system run.

    Runtime control is handled by architecture-specific iteration and round
    limits; this manager only records token usage against a nominal reference
    budget for monitoring and downstream metrics.

    Thread-safe for async multi-agent use.
    """

    DISABLED_BUDGET_SENTINEL = 10**18

    def __init__(self, total_budget: int = 4800, enabled: bool = True):
        self._total_budget = total_budget
        self._enabled = enabled
        self._input_tokens_used: int = 0
        self._output_tokens_used: int = 0
        self._lock = threading.Lock()

    def consume(self, input_tokens: int, output_tokens: int) -> None:
        """Call after every LLM invocation to record observed token usage."""
        with self._lock:
            self._input_tokens_used += input_tokens
            self._output_tokens_used += output_tokens

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self._total_budget - self.used)

    @property
    def used(self) -> int:
        return self._input_tokens_used + self._output_tokens_used

    @property
    def budget_exceeded(self) -> bool:
        if not self._enabled:
            return False
        return self.used > self._total_budget

    @property
    def total_budget(self) -> int:
        return self._total_budget

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_summary(self) -> dict:
        """Return summary dict for logging/metrics."""
        with self._lock:
            total_used = self._input_tokens_used + self._output_tokens_used
            remaining = max(0, self._total_budget - total_used)
            budget_exceeded = self._enabled and total_used > self._total_budget
            return {
                "total_budget": self._total_budget,
                "input_tokens_used": self._input_tokens_used,
                "output_tokens_used": self._output_tokens_used,
                "total_used": total_used,
                "remaining": remaining,
                "budget_exceeded": budget_exceeded,
            }

    def log_status(self) -> None:
        """Log current token budget status at INFO level."""
        summary = self.get_summary()
        logger.info(
            f"Token budget: {summary['total_used']}/{summary['total_budget']} used "
            f"(input={summary['input_tokens_used']}, output={summary['output_tokens_used']}), "
            f"{summary['remaining']} remaining"
            + (" [EXCEEDED]" if summary["budget_exceeded"] else "")
        )

    @classmethod
    def create(cls, enabled: bool, total_tokens: int = 4800) -> "TokenBudgetManager":
        """Factory: creates a manager with infinite budget when disabled."""
        if not enabled:
            return cls(total_budget=cls.DISABLED_BUDGET_SENTINEL, enabled=False)
        return cls(total_budget=total_tokens, enabled=True)


def extract_token_usage(response: AIMessage) -> Tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a LangChain AIMessage.

    Reads usage from the litellm_response stored in response_metadata
    by ChatLiteLLMLC._create_chat_result().
    Returns (0, 0) if usage info is not available.
    """
    metadata = getattr(response, "response_metadata", {}) or {}
    litellm_resp = metadata.get("litellm_response")
    if litellm_resp is None:
        return 0, 0
    # litellm_resp can be a ModelResponse object or a dict
    usage = getattr(litellm_resp, "usage", None)
    if usage is None and isinstance(litellm_resp, dict):
        usage = litellm_resp.get("usage")
    if usage is None:
        return 0, 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
    return prompt_tokens, completion_tokens
