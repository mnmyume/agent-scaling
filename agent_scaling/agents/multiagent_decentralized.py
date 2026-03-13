from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMConfig, LLMParams
from agent_scaling.datasets import (
    DatasetEnvStatus,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.logger import logger
from agent_scaling.metrics.artifacts import MetricsArtifacts
from agent_scaling.metrics.usage import extract_usage_from_model_response

from .multiagent_components.conversation import SubAgentConversationHistory
from .multiagent_components.mas_subagent import WorkerSubagent
from .registry import register_agent


@register_agent("multi-agent-decentralized")
class DecentralizedMultiAgentSystem(AgentSystemWithTools):
    """Decentralized MAS: peer exchange without a central orchestrator."""

    required_prompts = ["subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        max_rounds: int = 2,
        consensus_threshold: Optional[float] = None,
        subagent_llm: LLMConfig | Dict[str, Any] | None = None,
        subagent_llms: Optional[List[LLMConfig | Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.subagent_llm = (
            self._build_llm(subagent_llm) if subagent_llm is not None else self.llm
        )
        self.subagent_llms = (
            [self._build_llm(cfg) for cfg in subagent_llms]
            if subagent_llms is not None
            else None
        )

    def _create_subagents(
        self, task_instance: DatasetInstance
    ) -> Dict[str, WorkerSubagent]:
        subagents: Dict[str, WorkerSubagent] = {}
        for idx in range(self.n_base_agents):
            agent_id = f"agent_{idx + 1}"
            objective = "Solve the task independently and provide a final answer."
            focus = "Independent reasoning"
            llm_override = self.subagent_llm
            if self.subagent_llms is not None and idx < len(self.subagent_llms):
                llm_override = self.subagent_llms[idx]
            subagents[agent_id] = WorkerSubagent.init_from_agent(
                agent=self,
                llm_override=llm_override,
                agent_id=agent_id,
                objective=objective,
                original_query=objective,
                strategy=focus,
                task_instance=task_instance,
                guidance_template="start_with_peer_guidance",
                min_iterations_per_agent=self.min_iterations_per_agent,
                max_iterations_per_agent=self.max_iterations_per_agent,
            )
        return subagents

    def _majority_vote(self, answers: List[str]) -> str:
        if not answers:
            return ""
        counts = Counter(answers)
        best, count = counts.most_common(1)[0]
        if self.consensus_threshold is not None:
            ratio = count / max(1, len(answers))
            if ratio < self.consensus_threshold:
                return answers[0]
        return best

    def _build_metrics_artifacts(
        self,
        subagent_conversations: Dict[str, SubAgentConversationHistory],
        findings_by_agent: Dict[str, List[str]],
        final_answer: str,
    ) -> MetricsArtifacts:
        artifacts = MetricsArtifacts()

        for conv in subagent_conversations.values():
            for round_msgs in conv.internal_comms:
                for msg in round_msgs:
                    if msg.litellm_message is not None:
                        artifacts.llm_calls.append(
                            extract_usage_from_model_response(msg.litellm_message)
                        )

        artifacts.inter_agent_messages = sum(
            len(conv.external_comms) for conv in subagent_conversations.values()
        )

        # Subagent outputs (last non-empty findings per agent)
        for agent_id, findings in findings_by_agent.items():
            last_non_empty = ""
            for finding in reversed(findings):
                if finding and finding.strip():
                    last_non_empty = finding.strip()
                    break
            if last_non_empty:
                artifacts.subagent_outputs.append(last_non_empty)
                artifacts.subagent_outputs_by_id[agent_id] = last_non_empty

        artifacts.final_output = final_answer
        if artifacts.subagent_outputs_by_id:
            artifacts.pre_state_text = "\n".join(
                f"{agent_id}: {text}"
                for agent_id, text in artifacts.subagent_outputs_by_id.items()
            )
        artifacts.post_state_text = final_answer
        return artifacts

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        logger.info(
            f"Decentralized MAS running {self.n_base_agents} agents for instance {instance_idx}"
        )
        subagents = self._create_subagents(instance)
        findings_by_agent: Dict[str, List[str]] = {k: [] for k in subagents.keys()}

        last_results = {}
        for round_idx in range(self.max_rounds):
            logger.info(f"Decentralized round {round_idx + 1}/{self.max_rounds}")
            for agent_id, agent in subagents.items():
                if round_idx == 0:
                    message = "Solve the task independently. Provide a final answer summary."
                else:
                    peer_findings = [
                        f"{peer_id}: {findings_by_agent[peer_id][-1]}"
                        for peer_id in subagents.keys()
                        if peer_id != agent_id and findings_by_agent[peer_id]
                    ]
                    message = (
                        "Peer findings:\n"
                        + ("\n".join(peer_findings) if peer_findings else "None")
                        + "\nUpdate your answer if needed."
                    )
                result = agent.process_peer_message(message)
                last_results[agent_id] = result
                if result.findings:
                    findings_by_agent[agent_id].append(result.findings)

        answers = [res.findings for res in last_results.values() if res.findings]
        final_answer = self._majority_vote(answers)

        metrics_artifacts = self._build_metrics_artifacts(
            {
                agent_id: agent.conv_history
                for agent_id, agent in subagents.items()
            },
            findings_by_agent,
            final_answer,
        )
        metrics_artifacts.tool_calls = sum(
            agent.env.env_status().num_steps for agent in subagents.values()
        )
        metrics_artifacts.reasoning_turns = metrics_artifacts.tool_calls
        metrics_artifacts.finalize()

        combined_env_status = DatasetEnvStatus(
            success=any(
                agent.env.env_status().success for agent in subagents.values()
            ),
            num_steps=metrics_artifacts.tool_calls,
        )

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],
            final_env_output=combined_env_status,
            metrics_artifacts=metrics_artifacts,
        )
