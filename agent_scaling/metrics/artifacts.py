from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LLMCallUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: Optional[str] = None


class MetricsArtifacts(BaseModel):
    llm_calls: List[LLMCallUsage] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    tool_calls: int = 0
    reasoning_turns: int = 0
    inter_agent_messages: int = 0
    subagent_outputs: List[str] = Field(default_factory=list)
    subagent_outputs_by_id: Dict[str, str] = Field(default_factory=dict)
    final_output: Optional[str] = None
    pre_state_text: Optional[str] = None
    post_state_text: Optional[str] = None

    def finalize(self) -> None:
        self.total_input_tokens = sum(c.input_tokens for c in self.llm_calls)
        self.total_output_tokens = sum(c.output_tokens for c in self.llm_calls)
        self.total_tokens = sum(c.total_tokens for c in self.llm_calls)
        self.total_cost_usd = sum(c.cost_usd for c in self.llm_calls)
