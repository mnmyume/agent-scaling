from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LLMCallUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    provider_cost_usd: float = 0.0
    openrouter_cost_usd: float = 0.0
    latency_ms: float = 0.0
    llm_provider: Optional[str] = None
    model: Optional[str] = None


class MetricsArtifacts(BaseModel):
    llm_calls: List[LLMCallUsage] = Field(default_factory=list)
    llm_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_provider_cost_usd: float = 0.0
    total_openrouter_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    avg_latency_ms: Optional[float] = None
    tool_calls: int = 0
    reasoning_turns: int = 0
    inter_agent_messages: int = 0
    subagent_outputs: List[str] = Field(default_factory=list)
    subagent_outputs_by_id: Dict[str, str] = Field(default_factory=dict)
    final_output: Optional[str] = None
    pre_state_text: Optional[str] = None
    post_state_text: Optional[str] = None

    def finalize(self) -> None:
        self.llm_call_count = len(self.llm_calls)
        self.total_input_tokens = sum(c.input_tokens for c in self.llm_calls)
        self.total_output_tokens = sum(c.output_tokens for c in self.llm_calls)
        self.total_tokens = sum(c.total_tokens for c in self.llm_calls)
        self.total_cost_usd = sum(c.cost_usd for c in self.llm_calls)
        self.total_provider_cost_usd = sum(c.provider_cost_usd for c in self.llm_calls)
        self.total_openrouter_cost_usd = sum(c.openrouter_cost_usd for c in self.llm_calls)
        self.total_latency_ms = sum(float(c.latency_ms or 0.0) for c in self.llm_calls)
        self.avg_latency_ms = (
            self.total_latency_ms / self.llm_call_count
            if self.llm_call_count > 0
            else None
        )
