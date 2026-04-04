from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel

from agent_scaling.logger import logger
from agent_scaling.utils.token_budget import (
    TokenBudgetExceeded,
    TokenBudgetManager,
    extract_token_usage,
)


class BudgetBucketState(BaseModel):
    allocated_tokens: Optional[int] = None
    expected_calls: int = 0
    calls_made: int = 0
    used_tokens: int = 0

    @property
    def remaining_tokens(self) -> Optional[int]:
        if self.allocated_tokens is None:
            return None
        return max(0, self.allocated_tokens - self.used_tokens)


class MASBudgetAllocator:
    """Topology-aware soft allocation layered over the shared token budget."""

    BUCKET_WEIGHTS: Dict[str, float] = {
        "planning": 0.7,
        "coordination": 0.45,
        "worker": 1.0,
        "peer": 0.8,
        "synthesis": 0.6,
        "consensus": 0.5,
    }
    RESERVED_FRACTION = 0.05
    SAFETY_MARGIN_TOKENS = 32
    MIN_COMPLETION_TOKENS = 48

    def __init__(
        self,
        architecture: str,
        budget_manager: Optional[TokenBudgetManager],
        states: Dict[str, BudgetBucketState],
        enabled: bool,
    ):
        self.architecture = architecture
        self.budget_manager = budget_manager
        self.states = states
        self.enabled = enabled and budget_manager is not None

    @classmethod
    def create(
        cls,
        architecture: str,
        budget_manager: Optional[TokenBudgetManager],
        n_agents: int,
        max_iterations_per_agent: int,
        max_rounds: int,
        peer_rounds: int = 0,
    ) -> "MASBudgetAllocator":
        states = cls._build_states(
            architecture=architecture,
            budget_manager=budget_manager,
            n_agents=n_agents,
            max_iterations_per_agent=max_iterations_per_agent,
            max_rounds=max_rounds,
            peer_rounds=peer_rounds,
        )
        enabled = bool(
            budget_manager is not None and budget_manager.enabled and budget_manager.total_budget
        )
        return cls(
            architecture=architecture,
            budget_manager=budget_manager,
            states=states,
            enabled=enabled,
        )

    @classmethod
    def _build_states(
        cls,
        architecture: str,
        budget_manager: Optional[TokenBudgetManager],
        n_agents: int,
        max_iterations_per_agent: int,
        max_rounds: int,
        peer_rounds: int,
    ) -> Dict[str, BudgetBucketState]:
        expected_calls = {
            "planning": 0,
            "coordination": 0,
            "worker": 0,
            "peer": 0,
            "synthesis": 0,
            "consensus": 0,
        }

        worker_round_calls = max(1, max_iterations_per_agent) + 1
        if architecture == "multi-agent-independent":
            expected_calls["worker"] = n_agents * worker_round_calls
            expected_calls["synthesis"] = 1
        elif architecture == "multi-agent-centralized":
            expected_calls["planning"] = 1
            expected_calls["coordination"] = n_agents * max(1, max_rounds) + max(
                1, max_rounds - 1
            )
            expected_calls["worker"] = n_agents * max(1, max_rounds) * worker_round_calls
            expected_calls["synthesis"] = 1
        elif architecture == "multi-agent-decentralized":
            expected_calls["worker"] = n_agents * worker_round_calls
            expected_calls["peer"] = (
                n_agents * max(0, max_rounds - 1) * worker_round_calls
            )
            expected_calls["consensus"] = n_agents
        elif architecture == "multi-agent-hybrid":
            expected_calls["planning"] = 1
            expected_calls["coordination"] = n_agents * max(1, max_rounds) + max(
                1, max_rounds - 1
            )
            expected_calls["worker"] = n_agents * max(1, max_rounds) * worker_round_calls
            expected_calls["peer"] = (
                n_agents * max(0, peer_rounds) * max(1, max_iterations_per_agent)
            )
            expected_calls["synthesis"] = 1

        states = {
            bucket: BudgetBucketState(expected_calls=calls)
            for bucket, calls in expected_calls.items()
        }
        if (
            budget_manager is None
            or not budget_manager.enabled
            or budget_manager.total_budget >= TokenBudgetManager.DISABLED_BUDGET_SENTINEL
        ):
            return states

        spendable_budget = int(budget_manager.total_budget * (1 - cls.RESERVED_FRACTION))
        weighted_calls = {
            bucket: calls * cls.BUCKET_WEIGHTS[bucket]
            for bucket, calls in expected_calls.items()
            if calls > 0
        }
        total_weight = sum(weighted_calls.values())
        if total_weight == 0:
            return states

        allocated = 0
        last_bucket = "worker"
        for bucket, weight in weighted_calls.items():
            tokens = int(spendable_budget * (weight / total_weight))
            states[bucket].allocated_tokens = tokens
            allocated += tokens
            last_bucket = bucket
        states[last_bucket].allocated_tokens = (
            states[last_bucket].allocated_tokens or 0
        ) + (spendable_budget - allocated)
        return states

    def prepare_call(
        self, bucket: str, messages: Any, llm_kwargs: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        kwargs = dict(llm_kwargs or {})
        if not self.enabled:
            return kwargs

        state = self.states[bucket]
        state_remaining = state.remaining_tokens
        if state_remaining is None:
            return kwargs
        global_remaining = (
            int(self.budget_manager.remaining) if self.budget_manager is not None else 0
        )
        remaining_tokens = max(state_remaining, global_remaining)

        estimated_prompt_tokens = estimate_message_tokens(messages)
        remaining_calls = max(1, state.expected_calls - state.calls_made)
        per_call_budget = max(0, remaining_tokens // remaining_calls)
        completion_budget = (
            per_call_budget - estimated_prompt_tokens - self.SAFETY_MARGIN_TOKENS
        )
        if completion_budget < self.MIN_COMPLETION_TOKENS:
            if global_remaining <= estimated_prompt_tokens + self.MIN_COMPLETION_TOKENS:
                raise TokenBudgetExceeded(
                    f"{bucket} budget exhausted for {self.architecture}: "
                    f"{global_remaining} tokens remaining."
                )
            completion_budget = self.MIN_COMPLETION_TOKENS

        if kwargs.get("max_tokens") is None:
            kwargs["max_tokens"] = completion_budget
        else:
            kwargs["max_tokens"] = min(int(kwargs["max_tokens"]), completion_budget)
        return kwargs

    def consume_response(self, bucket: str, response: Any) -> None:
        total_tokens = 0
        if response is not None:
            input_tokens, output_tokens = extract_token_usage(response)
            total_tokens = input_tokens + output_tokens
            if self.budget_manager is not None:
                self.budget_manager.consume(input_tokens, output_tokens)

        state = self.states[bucket]
        state.calls_made += 1
        state.used_tokens += total_tokens
        if (
            self.enabled
            and state.allocated_tokens is not None
            and state.used_tokens > state.allocated_tokens
        ):
            logger.info(
                f"{bucket} budget overspent for {self.architecture}: "
                f"{state.used_tokens}/{state.allocated_tokens}"
            )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "enabled": self.enabled,
            "buckets": {
                bucket: state.model_dump()
                | {"remaining_tokens": state.remaining_tokens}
                for bucket, state in self.states.items()
                if state.expected_calls > 0 or state.used_tokens > 0
            },
        }


def estimate_message_tokens(messages: Any) -> int:
    if messages is None:
        return 0
    if isinstance(messages, str):
        return max(1, len(messages) // 4)
    if isinstance(messages, list):
        total = 0
        for message in messages:
            total += estimate_message_tokens(message)
        return max(1, total)
    if isinstance(messages, dict):
        content = str(messages.get("content", ""))
        tool_calls = str(messages.get("tool_calls", ""))
        return max(1, (len(content) + len(tool_calls)) // 4 + 8)
    if hasattr(messages, "content"):
        return max(1, len(str(messages.content)) // 4 + 8)
    return max(1, len(str(messages)) // 4)
