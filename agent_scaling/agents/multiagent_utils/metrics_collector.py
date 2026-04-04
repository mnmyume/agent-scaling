"""
Runtime metrics collection for single-agent and multi-agent executions.

The collector keeps the existing dataclass-driven style while extending it to:
- log actual LLM/tool/communication traces
- store per-agent activity and outputs
- export raw per-instance artifacts plus lightweight derived metrics

Paper-style paired metrics such as communication overhead O and coordination
efficiency E_c are computed post-hoc in the experiment aggregator, not here.
"""

from __future__ import annotations

import json
import os.path as osp
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from litellm.cost_calculator import completion_cost

from agent_scaling.utils import write_json


@dataclass
class RunMetadata:
    architecture: str
    total_agents: int
    dataset_id: Optional[str] = None
    instance_idx: Optional[int] = None
    instance_id: Optional[str] = None
    model_name: Optional[str] = None
    token_budget: Optional[int] = None


@dataclass
class CommunicationMetrics:
    """Track all explicit inter-agent or aggregator communication."""

    message_id: str
    timestamp: float
    sender: str
    recipients: List[str]
    message_type: str
    content_length: int
    round: Optional[int] = None
    iteration: Optional[int] = None
    latency_ms: Optional[float] = None
    channel: Optional[str] = None


@dataclass
class LLMMetrics:
    """Track LLM usage using actual runtime response metadata when available."""

    agent_id: str
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    call_type: str = "reasoning"
    round: Optional[int] = None
    iteration: Optional[int] = None
    rate_limited: bool = False
    retry_count: int = 0
    cache_hit: bool = False
    cost_source: str = "unknown"
    error_message: Optional[str] = None


@dataclass
class ToolMetrics:
    """Track tool usage patterns."""

    agent_id: str
    tool_name: str
    timestamp: float
    arguments: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    execution_time_ms: float
    round: Optional[int] = None
    iteration: Optional[int] = None
    result_length: int = 0


@dataclass
class AgentOutputMetrics:
    """Track textual outputs emitted by agents or aggregators."""

    agent_id: str
    timestamp: float
    output_type: str
    content: str
    content_length: int
    round: Optional[int] = None
    iteration: Optional[int] = None


@dataclass
class AgentMetrics:
    """Per-agent activity summary."""

    agent_id: str
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    total_findings: int = 0
    unique_findings: int = 0
    total_outputs: int = 0
    execution_time_s: float = 0.0
    communication_sent: int = 0
    communication_received: int = 0
    active_rounds: int = 0
    first_activity_ts: Optional[float] = None
    last_activity_ts: Optional[float] = None


@dataclass
class SystemMetrics:
    """Overall runtime summary for a single instance run."""

    architecture: str
    total_agents: int
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    total_execution_time_s: float = 0.0
    total_turns: int = 0

    total_messages: int = 0
    avg_message_latency_ms: float = 0.0
    communication_overhead_percent: float = 0.0
    message_density: float = 0.0

    total_llm_calls: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    avg_llm_latency_ms: float = 0.0

    total_tool_calls: int = 0
    successful_tool_calls: int = 0

    env_success: bool = False
    task_success: Optional[bool] = None
    task_success_source: Optional[str] = None
    success_per_1k_tokens: Optional[float] = None

    duplicate_work_ratio: float = 0.0
    error_recovery_success_rate: float = 0.0
    completion_reason: Optional[str] = None
    total_errors: int = 0


class MetricsCollector:
    """Centralized metrics collection for a single runtime execution."""

    # Used only as a last fallback when LiteLLM metadata does not provide cost.
    MODEL_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "gemini-pro": {"input": 0.00025, "output": 0.00125},
        "gemini-2.0-flash": {"input": 0.00025, "output": 0.00125},
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    }

    def __init__(
        self,
        architecture: str,
        num_agents: int,
        dataset_id: Optional[str] = None,
        instance_idx: Optional[int] = None,
        instance_id: Optional[str] = None,
        model_name: Optional[str] = None,
        token_budget: Optional[int] = None,
    ):
        self.run_metadata = RunMetadata(
            architecture=architecture,
            total_agents=num_agents,
            dataset_id=dataset_id,
            instance_idx=instance_idx,
            instance_id=instance_id,
            model_name=model_name,
            token_budget=token_budget,
        )
        self.system_metrics = SystemMetrics(
            architecture=architecture,
            total_agents=num_agents,
        )
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.communication_log: List[CommunicationMetrics] = []
        self.llm_log: List[LLMMetrics] = []
        self.tool_log: List[ToolMetrics] = []
        self.agent_output_log: List[AgentOutputMetrics] = []

        self._agent_rounds_seen: Dict[str, set[int]] = {}
        self._agent_finding_texts: Dict[str, set[str]] = {}

    def _ensure_agent(self, agent_id: str) -> AgentMetrics:
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        return self.agent_metrics[agent_id]

    def _mark_agent_activity(
        self,
        agent_id: str,
        timestamp: Optional[float] = None,
        round_num: Optional[int] = None,
    ) -> AgentMetrics:
        ts = timestamp if timestamp is not None else time.time()
        agent = self._ensure_agent(agent_id)
        if agent.first_activity_ts is None:
            agent.first_activity_ts = ts
        agent.last_activity_ts = ts
        if round_num is not None:
            self._agent_rounds_seen.setdefault(agent_id, set()).add(round_num)
        return agent

    @staticmethod
    def _coerce_timestamp(timestamp: Optional[float | str]) -> float:
        if timestamp is None:
            return time.time()
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        try:
            return datetime.fromisoformat(timestamp).timestamp()
        except ValueError:
            return time.time()

    @staticmethod
    def _safe_content_length(content: Any) -> int:
        if content is None:
            return 0
        return len(str(content))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_key = model.split("/")[-1]
        pricing = self.MODEL_PRICING.get(model_key)
        if pricing is None:
            return 0.0
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

    @staticmethod
    def _extract_usage_dict(litellm_response: Any) -> Dict[str, Any]:
        usage = getattr(litellm_response, "usage", None)
        if usage is None and isinstance(litellm_response, dict):
            usage = litellm_response.get("usage")
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return dict(usage)
        return {}

    @staticmethod
    def _extract_hidden_params(litellm_response: Any) -> Dict[str, Any]:
        hidden = getattr(litellm_response, "_hidden_params", None)
        if hidden is None and isinstance(litellm_response, dict):
            hidden = litellm_response.get("_hidden_params") or litellm_response.get(
                "hidden_params"
            )
        return dict(hidden or {})

    def extract_llm_response_metrics(self, response: AIMessage) -> Dict[str, Any]:
        metadata = getattr(response, "response_metadata", {}) or {}
        litellm_response = metadata.get("litellm_response")
        usage = self._extract_usage_dict(litellm_response)
        hidden = self._extract_hidden_params(litellm_response)

        model = (
            metadata.get("model_name")
            or getattr(litellm_response, "model", None)
            or (litellm_response.get("model") if isinstance(litellm_response, dict) else None)
            or self.run_metadata.model_name
            or "unknown"
        )

        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

        cost_usd = usage.get("cost")
        cost_source = "usage.cost"
        if cost_usd is None and hidden.get("response_cost") is not None:
            cost_usd = hidden.get("response_cost")
            cost_source = "hidden.response_cost"
        if cost_usd is None and litellm_response is not None:
            try:
                cost_usd = completion_cost(litellm_response)
                cost_source = "completion_cost"
            except Exception:
                cost_usd = None
        if cost_usd is None:
            cost_usd = self._estimate_cost(model, input_tokens, output_tokens)
            cost_source = "pricing_fallback"

        latency_ms = (
            metadata.get("latency_ms")
            or hidden.get("response_ms")
            or hidden.get("_response_ms")
            or hidden.get("completion_ms")
        )
        if latency_ms is not None:
            try:
                latency_ms = float(latency_ms)
            except (TypeError, ValueError):
                latency_ms = None

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": float(cost_usd or 0.0),
            "cost_source": cost_source,
            "cache_hit": bool(hidden.get("cache_hit", False)),
            "rate_limited": bool(
                hidden.get("rate_limit_error", False) or metadata.get("rate_limited", False)
            ),
            "latency_ms": latency_ms,
        }

    def set_env_success(self, success: bool) -> None:
        self.system_metrics.env_success = success

    def set_task_success(self, success: bool, source: str = "evaluation") -> None:
        self.system_metrics.task_success = success
        self.system_metrics.task_success_source = source

    def set_completion_reason(self, reason: Optional[str]) -> None:
        self.system_metrics.completion_reason = reason

    def log_communication(
        self,
        sender: str,
        recipients: List[str],
        message_type: str,
        content: str,
        round: Optional[int] = None,
        iteration: Optional[int] = None,
        latency_ms: Optional[float] = None,
        channel: Optional[str] = None,
        timestamp: Optional[float | str] = None,
    ) -> None:
        ts = self._coerce_timestamp(timestamp)
        metric = CommunicationMetrics(
            message_id=f"{sender}_{len(self.communication_log)}_{ts}",
            timestamp=ts,
            sender=sender,
            recipients=recipients,
            message_type=message_type,
            content_length=self._safe_content_length(content),
            round=round,
            iteration=iteration,
            latency_ms=latency_ms,
            channel=channel,
        )
        self.communication_log.append(metric)

        sender_metrics = self._mark_agent_activity(sender, timestamp=ts, round_num=round)
        sender_metrics.communication_sent += 1

        for recipient in recipients:
            recipient_metrics = self._mark_agent_activity(
                recipient, timestamp=ts, round_num=round
            )
            recipient_metrics.communication_received += 1

    def ingest_communication_event(self, event: Any) -> None:
        self.log_communication(
            sender=getattr(event, "sender_id", "unknown"),
            recipients=[getattr(event, "recipient_id", "unknown")],
            message_type=getattr(event, "channel", "communication"),
            content=getattr(event, "message", ""),
            round=getattr(event, "round_num", None),
            iteration=None,
            channel=getattr(event, "channel", None),
            timestamp=getattr(event, "timestamp", None),
        )

    def log_llm_call(
        self,
        agent_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        call_type: str = "reasoning",
        round: Optional[int] = None,
        iteration: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        cost_source: str = "unknown",
        rate_limited: bool = False,
        retry_count: int = 0,
        cache_hit: bool = False,
        error_message: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens
        if cost_usd is None:
            cost_usd = self._estimate_cost(model, input_tokens, output_tokens)
            cost_source = "pricing_fallback"

        metric = LLMMetrics(
            agent_id=agent_id,
            timestamp=timestamp or time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=float(cost_usd),
            latency_ms=float(latency_ms),
            success=success,
            call_type=call_type,
            round=round,
            iteration=iteration,
            rate_limited=rate_limited,
            retry_count=retry_count,
            cache_hit=cache_hit,
            cost_source=cost_source,
            error_message=error_message,
        )
        self.llm_log.append(metric)

        agent = self._mark_agent_activity(agent_id, timestamp=metric.timestamp, round_num=round)
        agent.total_llm_calls += 1
        agent.total_tokens += total_tokens
        agent.total_cost += float(cost_usd)

        self.system_metrics.total_llm_calls += 1
        self.system_metrics.total_tokens_used += total_tokens
        self.system_metrics.total_cost_usd += float(cost_usd)
        if not success:
            self.system_metrics.total_errors += 1

    def log_llm_response(
        self,
        agent_id: str,
        response: AIMessage,
        latency_ms: float,
        call_type: str = "reasoning",
        round: Optional[int] = None,
        iteration: Optional[int] = None,
        retry_count: int = 0,
    ) -> None:
        extracted = self.extract_llm_response_metrics(response)
        self.log_llm_call(
            agent_id=agent_id,
            model=extracted["model"],
            input_tokens=extracted["input_tokens"],
            output_tokens=extracted["output_tokens"],
            total_tokens=extracted["total_tokens"],
            cost_usd=extracted["cost_usd"],
            cost_source=extracted["cost_source"],
            latency_ms=float(extracted["latency_ms"] or latency_ms),
            success=True,
            call_type=call_type,
            round=round,
            iteration=iteration,
            rate_limited=extracted["rate_limited"],
            retry_count=retry_count,
            cache_hit=extracted["cache_hit"],
        )

    def log_llm_failure(
        self,
        agent_id: str,
        model: str,
        latency_ms: float,
        call_type: str = "reasoning",
        round: Optional[int] = None,
        iteration: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.log_llm_call(
            agent_id=agent_id,
            model=model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            call_type=call_type,
            round=round,
            iteration=iteration,
            error_message=error_message,
        )

    def log_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        execution_time_ms: float,
        round: Optional[int],
        iteration: Optional[int],
        error_message: Optional[str] = None,
        result: Optional[Any] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        metric = ToolMetrics(
            agent_id=agent_id,
            tool_name=tool_name,
            timestamp=timestamp or time.time(),
            arguments=arguments,
            success=success,
            error_message=error_message,
            execution_time_ms=float(execution_time_ms),
            round=round,
            iteration=iteration,
            result_length=self._safe_content_length(result),
        )
        self.tool_log.append(metric)

        agent = self._mark_agent_activity(agent_id, timestamp=metric.timestamp, round_num=round)
        agent.total_tool_calls += 1
        if success:
            agent.successful_tool_calls += 1
        else:
            self.system_metrics.total_errors += 1

        self.system_metrics.total_tool_calls += 1
        if success:
            self.system_metrics.successful_tool_calls += 1

    def log_agent_output(
        self,
        agent_id: str,
        output_type: str,
        content: str,
        round: Optional[int] = None,
        iteration: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        ts = timestamp or time.time()
        content = content or ""
        metric = AgentOutputMetrics(
            agent_id=agent_id,
            timestamp=ts,
            output_type=output_type,
            content=content,
            content_length=len(content),
            round=round,
            iteration=iteration,
        )
        self.agent_output_log.append(metric)

        agent = self._mark_agent_activity(agent_id, timestamp=ts, round_num=round)
        agent.total_outputs += 1
        if output_type == "finding":
            agent.total_findings += 1
            normalized = self._normalize_text(content)
            if normalized:
                self._agent_finding_texts.setdefault(agent_id, set()).add(normalized)
                agent.unique_findings = len(self._agent_finding_texts[agent_id])

    def calculate_final_metrics(self) -> None:
        if self.system_metrics.end_time is None:
            self.system_metrics.end_time = time.time()
        self.system_metrics.total_execution_time_s = (
            self.system_metrics.end_time - self.system_metrics.start_time
        )

        self.system_metrics.total_llm_calls = len(self.llm_log)
        self.system_metrics.total_turns = len(self.llm_log)
        self.system_metrics.total_tokens_used = sum(metric.total_tokens for metric in self.llm_log)
        self.system_metrics.total_cost_usd = sum(metric.cost_usd for metric in self.llm_log)
        self.system_metrics.total_tool_calls = len(self.tool_log)
        self.system_metrics.successful_tool_calls = sum(
            1 for metric in self.tool_log if metric.success
        )
        self.system_metrics.total_messages = len(self.communication_log)

        llm_latencies = [metric.latency_ms for metric in self.llm_log if metric.latency_ms > 0]
        if llm_latencies:
            self.system_metrics.avg_llm_latency_ms = sum(llm_latencies) / len(llm_latencies)

        comm_latencies = [
            metric.latency_ms
            for metric in self.communication_log
            if metric.latency_ms is not None
        ]
        if comm_latencies:
            self.system_metrics.avg_message_latency_ms = sum(comm_latencies) / len(comm_latencies)

        if self.system_metrics.total_execution_time_s > 0:
            comm_time = sum(comm_latencies) / 1000
            self.system_metrics.communication_overhead_percent = (
                comm_time / self.system_metrics.total_execution_time_s * 100
            )

        if self.system_metrics.total_turns > 0:
            self.system_metrics.message_density = (
                self.system_metrics.total_messages / self.system_metrics.total_turns
            )

        duplicate_patterns: Dict[str, set[str]] = {}
        for tool_metric in self.tool_log:
            key = json.dumps(
                {
                    "tool_name": tool_metric.tool_name,
                    "arguments": tool_metric.arguments,
                },
                sort_keys=True,
                default=str,
            )
            duplicate_patterns.setdefault(key, set()).add(tool_metric.agent_id)
        if duplicate_patterns:
            duplicate_count = sum(
                1 for agents in duplicate_patterns.values() if len(agents) > 1
            )
            self.system_metrics.duplicate_work_ratio = duplicate_count / len(duplicate_patterns)

        if self.system_metrics.total_errors > 0:
            recovered = sum(
                1
                for idx, tool_metric in enumerate(self.tool_log)
                if not tool_metric.success
                and any(
                    later_metric.agent_id == tool_metric.agent_id
                    and later_metric.tool_name == tool_metric.tool_name
                    and later_metric.success
                    and later_metric.timestamp > tool_metric.timestamp
                    for later_metric in self.tool_log[idx + 1 :]
                )
            )
            self.system_metrics.error_recovery_success_rate = (
                recovered / self.system_metrics.total_errors
            )

        if (
            self.system_metrics.task_success is not None
            and self.system_metrics.total_tokens_used > 0
        ):
            self.system_metrics.success_per_1k_tokens = (
                1000 * int(self.system_metrics.task_success)
            ) / self.system_metrics.total_tokens_used

        for agent_id, agent_metrics in self.agent_metrics.items():
            rounds_seen = self._agent_rounds_seen.get(agent_id, set())
            agent_metrics.active_rounds = len(rounds_seen)
            if (
                agent_metrics.first_activity_ts is not None
                and agent_metrics.last_activity_ts is not None
            ):
                agent_metrics.execution_time_s = max(
                    0.0,
                    agent_metrics.last_activity_ts - agent_metrics.first_activity_ts,
                )

    def export_event_trace(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for metric in self.communication_log:
            records.append({"event_type": "communication", **asdict(metric)})
        for metric in self.llm_log:
            records.append({"event_type": "llm_call", **asdict(metric)})
        for metric in self.tool_log:
            records.append({"event_type": "tool_call", **asdict(metric)})
        for metric in self.agent_output_log:
            records.append({"event_type": "agent_output", **asdict(metric)})
        records.sort(key=lambda record: (record.get("timestamp", 0.0), record["event_type"]))
        return records

    def export_metrics(self) -> Dict[str, Any]:
        self.calculate_final_metrics()
        return {
            "run_metadata": asdict(self.run_metadata),
            "system_metrics": asdict(self.system_metrics),
            "agent_metrics": {
                agent_id: asdict(metrics)
                for agent_id, metrics in sorted(self.agent_metrics.items())
            },
            "communication_log": [asdict(metric) for metric in self.communication_log],
            "llm_log": [asdict(metric) for metric in self.llm_log],
            "tool_log": [asdict(metric) for metric in self.tool_log],
            "agent_output_log": [asdict(metric) for metric in self.agent_output_log],
            "summary": {
                "architecture": self.system_metrics.architecture,
                "agents": self.system_metrics.total_agents,
                "execution_time_s": self.system_metrics.total_execution_time_s,
                "completion_reason": self.system_metrics.completion_reason,
                "env_success": self.system_metrics.env_success,
                "task_success": self.system_metrics.task_success,
                "task_success_source": self.system_metrics.task_success_source,
                "total_turns": self.system_metrics.total_turns,
                "total_messages": self.system_metrics.total_messages,
                "message_density": self.system_metrics.message_density,
                "avg_message_latency_ms": self.system_metrics.avg_message_latency_ms,
                "avg_llm_latency_ms": self.system_metrics.avg_llm_latency_ms,
                "total_llm_calls": self.system_metrics.total_llm_calls,
                "total_tool_calls": self.system_metrics.total_tool_calls,
                "total_tokens": self.system_metrics.total_tokens_used,
                "total_cost_usd": round(self.system_metrics.total_cost_usd, 8),
                "duplicate_work_ratio": self.system_metrics.duplicate_work_ratio,
                "success_per_1k_tokens": self.system_metrics.success_per_1k_tokens,
            },
        }

    def write_artifacts(self, instance_dir: str) -> Dict[str, Any]:
        metrics = self.export_metrics()
        write_json(metrics, osp.join(instance_dir, "runtime_metrics.json"), indent=True)
        with open(osp.join(instance_dir, "runtime_events.jsonl"), "w") as handle:
            for record in self.export_event_trace():
                handle.write(json.dumps(record, default=str))
                handle.write("\n")
        return metrics
