from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

from agent_scaling.utils.token_budget import TokenBudgetManager, extract_token_usage


class BudgetBucketState(BaseModel):
    expected_calls: int = 0
    calls_made: int = 0
    used_tokens: int = 0


class MASBudgetAllocator:
    """Static paper-style token telemetry grouped by coordination phase."""

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
        peer_max_iterations: int = 1,
    ) -> "MASBudgetAllocator":
        states = cls._build_states(
            architecture=architecture,
            n_agents=n_agents,
            max_iterations_per_agent=max_iterations_per_agent,
            max_rounds=max_rounds,
            peer_rounds=peer_rounds,
            peer_max_iterations=peer_max_iterations,
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
        n_agents: int,
        max_iterations_per_agent: int,
        max_rounds: int,
        peer_rounds: int,
        peer_max_iterations: int,
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
                n_agents * max(0, peer_rounds) * max(1, peer_max_iterations)
            )
            expected_calls["synthesis"] = 1

        states = {
            bucket: BudgetBucketState(expected_calls=calls)
            for bucket, calls in expected_calls.items()
        }
        return states

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

    def snapshot(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "enabled": self.enabled,
            "buckets": {
                bucket: state.model_dump()
                for bucket, state in self.states.items()
                if state.expected_calls > 0 or state.used_tokens > 0
            },
        }
