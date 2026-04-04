from __future__ import annotations

import asyncio
import os
import os.path as osp
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.agents.multiagent_utils.metrics_collector import MetricsCollector
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.utils import join_with_leading_dash, write_yaml
from agent_scaling.utils.token_budget import TokenBudgetManager

from .budgeting import MASBudgetAllocator
from .conversation import CommunicationEvent, OrchestrationResult, SubAgentRoundResult
from .mas_lead_agent import LeadAgent
from .mas_subagent import WorkerSubagent
from .memory import EnhancedMemory
from .plan import OrchestrationPlan, Subtask


class BaseMultiAgentSystem(AgentSystemWithTools, ABC):
    """Shared runtime for the multi-agent architecture variants."""

    required_prompts = ["lead_agent", "subagent"]
    architecture_name = "multi-agent"
    default_task_blurb = "task coordinator"

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 3,
        max_rounds: int = 5,
        peer_rounds: int = 1,
        peer_fanout: Optional[int] = None,
        peer_max_iterations: int = 1,
        consensus_threshold: float = 0.67,
        task_blurb: Optional[str] = None,
        max_execution_time: int = 300,
        worker_timeout: int = 120,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.max_rounds = max_rounds
        self.peer_rounds = peer_rounds
        self.peer_max_iterations = peer_max_iterations
        self.consensus_threshold = consensus_threshold
        self.max_execution_time = max_execution_time
        self.worker_timeout = worker_timeout
        self.task_blurb = task_blurb or self.default_task_blurb
        max_peer_targets = max(0, n_base_agents - 1)
        requested_peer_fanout = peer_fanout if peer_fanout is not None else max_peer_targets
        self.peer_fanout = min(max_peer_targets, requested_peer_fanout)

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
        budget_manager: Optional[TokenBudgetManager] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        return asyncio.run(
            self.run_agent_async(
                instance=instance,
                instance_dir=instance_dir,
                llm_params=llm_params,
                instance_idx=instance_idx,
                budget_manager=budget_manager,
            )
        )

    async def run_agent_async(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
        budget_manager: Optional[TokenBudgetManager] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        start_time = time.time()
        llm_params_dict = llm_params.model_dump() if llm_params else {}
        instance_identifier = getattr(instance, "index", getattr(instance, "task_id", None))
        self.metrics_collector = MetricsCollector(
            architecture=self.architecture_name,
            num_agents=self.n_base_agents,
            dataset_id=self.dataset.dataset_id,
            instance_idx=instance_idx,
            instance_id=str(instance_identifier) if instance_identifier is not None else None,
            model_name=getattr(self.llm, "model", None),
            token_budget=budget_manager.total_budget if budget_manager else None,
        )
        budget_allocator = MASBudgetAllocator.create(
            architecture=self.architecture_name,
            budget_manager=budget_manager,
            n_agents=self.n_base_agents,
            max_iterations_per_agent=self.max_iterations_per_agent,
            max_rounds=self.max_rounds,
            peer_rounds=self.peer_rounds,
            peer_max_iterations=self.peer_max_iterations,
        )

        logger.info(
            f"Starting {self.architecture_name} for instance {instance_idx} with "
            f"{self.n_base_agents} agents"
        )
        result = await self._run_architecture(
            instance=instance,
            llm_params_dict=llm_params_dict,
            budget_manager=budget_manager,
            budget_allocator=budget_allocator,
        )
        result.architecture = self.architecture_name
        result.budget_allocation = budget_allocator.snapshot()
        result.communication_events.extend(
            self._extract_non_peer_events(result.subagent_conversations)
        )
        for event in result.communication_events:
            self.metrics_collector.ingest_communication_event(event)

        execution_time = time.time() - start_time
        self.metrics_collector.system_metrics.end_time = (
            self.metrics_collector.system_metrics.start_time + execution_time
        )
        self.metrics_collector.set_env_success(result.combined_env_status.success)
        self.metrics_collector.set_completion_reason(result.completion_reason)
        if result.synthesized_answer:
            self.metrics_collector.log_agent_output(
                agent_id="lead_agent"
                if result.lead_agent_conversation is not None
                else "aggregator",
                output_type="synthesized_answer",
                content=result.synthesized_answer,
                round=result.total_rounds,
            )
        logger.info(
            f"{self.architecture_name} completed in {execution_time:.2f}s with "
            f"{result.total_agent_iterations} total worker iterations"
        )
        if budget_manager is not None:
            budget_manager.log_status()

        runtime_metrics = self.metrics_collector.export_metrics()
        runtime_metrics_path = None
        runtime_events_path = None
        if instance_dir is not None:
            os.makedirs(instance_dir, exist_ok=True)
            write_yaml(
                result.model_dump(),
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )
            runtime_metrics = self.metrics_collector.write_artifacts(instance_dir)
            runtime_metrics_path = osp.join(instance_dir, "runtime_metrics.json")
            runtime_events_path = osp.join(instance_dir, "runtime_events.jsonl")

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=result.synthesized_answer or "",
            trajectory=[],
            final_env_output=result.combined_env_status,
            budget_used=budget_manager.used if budget_manager else 0,
            budget_remaining=budget_manager.remaining if budget_manager else 0,
            budget_exceeded=budget_manager.budget_exceeded if budget_manager else False,
            runtime_metrics=runtime_metrics,
            runtime_metrics_path=runtime_metrics_path,
            runtime_events_path=runtime_events_path,
        )

    @abstractmethod
    async def _run_architecture(
        self,
        instance: DatasetInstance,
        llm_params_dict: Dict[str, Any],
        budget_manager: Optional[TokenBudgetManager],
        budget_allocator: MASBudgetAllocator,
    ) -> OrchestrationResult:
        pass

    def _create_lead_agent(
        self,
        memory: EnhancedMemory,
        llm_params_dict: Dict[str, Any],
        budget_manager: Optional[TokenBudgetManager],
        budget_allocator: MASBudgetAllocator,
    ) -> LeadAgent:
        lead_agent = LeadAgent(
            llm=self.llm,
            dataset=self.dataset,
            prompts=self.prompts,
            tools=self.tools,
            env=self.env_name,
            env_prompts=self.env_prompts,
            metrics_collector=self.metrics_collector,
            memory=memory,
            min_iterations_per_agent=self.min_iterations_per_agent,
            max_rounds=self.max_rounds,
            max_execution_time=self.max_execution_time,
            worker_timeout=self.worker_timeout,
            num_base_agents=self.n_base_agents,
            task_blurb=self.task_blurb,
            max_iterations_per_agent=self.max_iterations_per_agent,
        )
        lead_agent.budget_manager = budget_manager
        lead_agent.budget_allocator = budget_allocator
        lead_agent.llm_params_dict = llm_params_dict
        return lead_agent

    def _create_worker(
        self,
        subtask: Subtask,
        instance: DatasetInstance,
        llm_params_dict: Dict[str, Any],
        budget_manager: Optional[TokenBudgetManager],
        budget_allocator: MASBudgetAllocator,
    ) -> WorkerSubagent:
        worker = WorkerSubagent.init_from_agent(
            agent=self,
            agent_id=subtask.agent_id,
            objective=subtask.objective,
            original_query=subtask.objective,
            strategy=subtask.focus,
            task_instance=instance,
            min_iterations_per_agent=self.min_iterations_per_agent,
            max_iterations_per_agent=self.max_iterations_per_agent,
            llm_params_dict=llm_params_dict,
            budget_allocator=budget_allocator,
        )
        worker.budget_manager = budget_manager
        worker.budget_allocator = budget_allocator
        return worker

    def _build_default_parallel_plan(self, shared_prompt_templates: Dict[str, Any]) -> OrchestrationPlan:
        base_focuses = [
            "Broad exploration and first-pass evidence gathering.",
            "Verification, contradiction checking, and error spotting.",
            "Synthesis of partial evidence and gap discovery.",
            "Tool-efficient search for decisive evidence.",
        ]
        subtasks = []
        for index in range(self.n_base_agents):
            subtasks.append(
                Subtask(
                    agent_id=f"agent_{index + 1}",
                    objective="Solve the task end-to-end using only your own local work.",
                    focus=base_focuses[index % len(base_focuses)],
                )
            )
        return OrchestrationPlan(
            subtasks=subtasks,
            reasoning=(
                "Static focus assignment used to preserve a non-orchestrated topology "
                "while encouraging diverse parallel exploration."
            ),
        )

    def _invoke_system_llm(
        self,
        messages: List[Dict[str, Any]],
        llm_params_dict: Dict[str, Any],
        budget_allocator: MASBudgetAllocator,
        bucket: str,
    ):
        response = self._invoke_with_metrics(
            self.llm,
            messages,
            agent_id="lead_agent",
            llm_kwargs=llm_params_dict,
            call_type=bucket,
        )
        budget_allocator.consume_response(bucket, response)
        return response

    async def _run_parallel_workers(
        self,
        workers: Dict[str, WorkerSubagent],
        runner: Callable[[WorkerSubagent], SubAgentRoundResult],
    ) -> Dict[str, SubAgentRoundResult]:
        tasks = {
            agent_id: asyncio.create_task(asyncio.to_thread(runner, worker))
            for agent_id, worker in workers.items()
        }
        done, pending = await asyncio.wait(
            list(tasks.values()),
            timeout=self.worker_timeout,
        )
        for task in pending:
            task.cancel()

        results: Dict[str, SubAgentRoundResult] = {}
        for agent_id, task in tasks.items():
            worker = workers[agent_id]
            if task in done:
                try:
                    results[agent_id] = await task
                except Exception as exc:
                    results[agent_id] = SubAgentRoundResult(
                        agent_id=agent_id,
                        findings="",
                        env_status=worker.env.env_status(),
                        error_msg=str(exc),
                    )
            else:
                results[agent_id] = SubAgentRoundResult(
                    agent_id=agent_id,
                    findings="",
                    env_status=worker.env.env_status(),
                    error_msg="Timed out waiting for worker response.",
                )
        return results

    def _build_result(
        self,
        plan: OrchestrationPlan,
        workers: Dict[str, WorkerSubagent],
        agent_findings: Dict[str, List[str]],
        synthesized_answer: Optional[str],
        total_rounds: int,
        completion_reason: str,
        communication_events: Optional[List[CommunicationEvent]] = None,
        lead_agent_conversation: Any = None,
        final_candidates: Optional[List[Any]] = None,
    ) -> OrchestrationResult:
        return OrchestrationResult(
            architecture=self.architecture_name,
            plan=plan,
            subagent_conversations={
                agent_id: worker.conv_history for agent_id, worker in workers.items()
            },
            subagent_env_status={
                agent_id: worker.env.env_status() for agent_id, worker in workers.items()
            },
            subagent_findings=agent_findings,
            total_findings=sum(len(findings) for findings in agent_findings.values()),
            lead_agent_conversation=lead_agent_conversation,
            communication_events=communication_events or [],
            final_candidates=final_candidates or [],
            total_rounds=total_rounds,
            completion_reason=completion_reason,
            synthesized_answer=synthesized_answer,
        )

    def _extract_non_peer_events(
        self, conversations: Dict[str, WorkerSubagent | Any]
    ) -> List[CommunicationEvent]:
        events: List[CommunicationEvent] = []
        for conversation in conversations.values():
            if hasattr(conversation, "to_communication_events"):
                for event in conversation.to_communication_events():
                    if event.channel != "peer":
                        events.append(event)
        return events

    def _select_success_answer(
        self, round_results: Dict[str, SubAgentRoundResult]
    ) -> Optional[str]:
        for result in round_results.values():
            if result.env_status.success and result.findings:
                return result.findings
        for result in round_results.values():
            if result.findings:
                return result.findings
        return None

    def _format_findings_for_synthesis(self, findings_by_agent: Dict[str, List[str]]) -> str:
        findings = []
        for agent_id, agent_findings in findings_by_agent.items():
            for finding in agent_findings:
                findings.append(f"{agent_id}: {finding}")
        return join_with_leading_dash(findings)
