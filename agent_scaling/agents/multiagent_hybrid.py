from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMConfig, LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.metrics.orchestration import build_metrics_artifacts_from_orchestration

from .multiagent_components.conversation import OrchestrationResult
from .multiagent_components.mas_lead_agent import LeadAgent
from .multiagent_components.mas_subagent import WorkerSubagent
from .multiagent_components.memory import EnhancedMemory
from .registry import register_agent


@register_agent("multi-agent-hybrid")
class HybridMultiAgentSystem(AgentSystemWithTools):
    """Hybrid MAS: centralized planning + peer exchange."""

    required_prompts = ["lead_agent", "subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        max_rounds: int = 1,
        enable_peer_communication: bool = True,
        orchestrator_llm: LLMConfig | Dict[str, Any] | None = None,
        subagent_llm: LLMConfig | Dict[str, Any] | None = None,
        subagent_llms: Optional[List[LLMConfig | Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.max_rounds = max_rounds
        self.enable_peer_communication = enable_peer_communication

        self.orchestrator_llm = (
            self._build_llm(orchestrator_llm) if orchestrator_llm is not None else self.llm
        )
        self.subagent_llm = (
            self._build_llm(subagent_llm) if subagent_llm is not None else self.llm
        )
        self.subagent_llms = (
            [self._build_llm(cfg) for cfg in subagent_llms]
            if subagent_llms is not None
            else None
        )
        self.memory = EnhancedMemory()

        lead_kwargs = dict(kwargs)
        lead_kwargs["llm"] = self.orchestrator_llm
        self.lead_agent = LeadAgent(
            *args,
            memory=self.memory,
            min_iterations_per_agent=min_iterations_per_agent,
            num_base_agents=n_base_agents,
            orchestrator_llm=self.orchestrator_llm,
            subagent_llm=self.subagent_llm,
            subagent_llms=self.subagent_llms,
            domain_config={
                "task_blurb": lead_kwargs.get("task_blurb", "hybrid coordinator")
            },
            **lead_kwargs,
        )

    def _peer_exchange(
        self, subagents: Dict[str, WorkerSubagent]
    ) -> Dict[str, Any]:
        results = {}
        for agent_id, agent in subagents.items():
            peer_findings = [
                f"{peer_id}: {subagents[peer_id].conv_history.last_outgoing_external_message}"
                for peer_id in subagents.keys()
                if peer_id != agent_id
            ]
            message = (
                "Peer findings:\n"
                + ("\n".join(peer_findings) if peer_findings else "None")
                + "\nUpdate your answer if needed."
            )
            results[agent_id] = agent.process_peer_message(message)
        return results

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        return asyncio.run(
            self.run_agent_async(instance, instance_dir, llm_params, instance_idx)
        )

    async def run_agent_async(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        logger.info(f"Hybrid MAS running for instance {instance_idx}")

        # Centralized orchestration rounds
        result: OrchestrationResult = await self.lead_agent.orchestrate_work(
            task_instance=instance,
            llm_params_dict=llm_params.model_dump() if llm_params else {},
        )

        # Optional peer exchange rounds
        peer_results = None
        if (
            self.enable_peer_communication
            and self.lead_agent.subagents
            and self.max_rounds > 0
        ):
            for round_idx in range(self.max_rounds):
                logger.info(
                    f"Hybrid MAS peer exchange round {round_idx + 1}/{self.max_rounds}"
                )
                peer_results = await asyncio.to_thread(
                    self._peer_exchange, self.lead_agent.subagents
                )
                self.lead_agent._update_memory_with_turn_results(peer_results)

            # Re-synthesize after peer exchange if needed
            if peer_results and all(
                not r.env_status.success for r in peer_results.values()
            ):
                result.synthesized_answer = self.lead_agent._synthesize_findings()

            result = OrchestrationResult(
                plan=result.plan,
                synthesized_answer=result.synthesized_answer,
                subagent_conversations={
                    agent_id: agent.conv_history
                    for agent_id, agent in self.lead_agent.subagents.items()
                },
                subagent_env_status={
                    agent_id: agent.env.env_status()
                    for agent_id, agent in self.lead_agent.subagents.items()
                },
                subagent_findings=self.memory.agent_findings,
                total_findings=len(self.memory.all_findings),
                lead_agent_conversation=self.lead_agent.conv_history,
            )

        metrics_artifacts = build_metrics_artifacts_from_orchestration(result)
        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=result.synthesized_answer,
            trajectory=[],
            final_env_output=result.combined_env_status,
            metrics_artifacts=metrics_artifacts,
        )
