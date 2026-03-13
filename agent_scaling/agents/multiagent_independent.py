from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.agents.single_agent import SingleAgent
from agent_scaling.config.llm import LLMConfig, LLMParams
from agent_scaling.datasets import DatasetEnvStatus, DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.metrics.artifacts import MetricsArtifacts

from .registry import register_agent


@register_agent("multi-agent-independent")
class IndependentMultiAgentSystem(AgentSystemWithTools):
    """Independent MAS: multiple single agents solve the task without coordination."""

    required_prompts = ["main"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        consensus_threshold: Optional[float] = None,
        subagent_llm: LLMConfig | Dict[str, Any] | None = None,
        subagent_llms: Optional[List[LLMConfig | Dict[str, Any]]] = None,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_base_agents = n_base_agents
        self.max_steps = max_steps or max_iterations_per_agent
        if consensus_threshold is not None:
            logger.warning(
                "Independent MAS uses synthesis-only aggregation; ignoring consensus_threshold=%s",
                consensus_threshold,
            )
        self.subagent_llm = (
            self._build_llm(subagent_llm) if subagent_llm is not None else self.llm
        )
        self.subagent_llms = (
            [self._build_llm(cfg) for cfg in subagent_llms]
            if subagent_llms is not None
            else None
        )

    def _build_agent(self, llm) -> SingleAgent:
        return SingleAgent(
            llm=llm,
            dataset=self.dataset,
            prompts=self.prompts,
            env=self.env_name,
            env_prompts=self.env_prompts,
            tools=self.tools,
            max_steps=self.max_steps,
        )

    def _synthesize_outputs(
        self,
        outputs: List[DatasetInstanceOutputWithTrajectory],
    ) -> str:
        if not outputs:
            return ""
        # Synthesis-only aggregation: concatenate sub-agent outputs without analysis.
        return "\n".join(
            f"agent_{idx + 1}: {str(output.agent_output)}"
            for idx, output in enumerate(outputs)
        )

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        logger.info(
            f"Independent MAS running {self.n_base_agents} agents for instance {instance_idx}"
        )
        outputs: List[DatasetInstanceOutputWithTrajectory] = []
        for idx in range(self.n_base_agents):
            llm = self.subagent_llm
            if self.subagent_llms is not None and idx < len(self.subagent_llms):
                llm = self.subagent_llms[idx]
            agent = self._build_agent(llm)
            output = agent.run_agent(
                instance,
                instance_dir=None,
                llm_params=llm_params,
                instance_idx=instance_idx,
            )
            outputs.append(output)

        final_answer = self._synthesize_outputs(outputs)
        combined_env_status = DatasetEnvStatus(
            success=any(
                out.final_env_output is not None and out.final_env_output.success
                for out in outputs
            ),
            num_steps=sum(
                out.final_env_output.num_steps
                for out in outputs
                if out.final_env_output is not None
            ),
        )

        metrics_artifacts = MetricsArtifacts()
        for idx, output in enumerate(outputs):
            if output.metrics_artifacts is not None:
                metrics_artifacts.llm_calls.extend(output.metrics_artifacts.llm_calls)
                metrics_artifacts.tool_calls += output.metrics_artifacts.tool_calls
                metrics_artifacts.reasoning_turns += (
                    output.metrics_artifacts.reasoning_turns
                )
            metrics_artifacts.subagent_outputs.append(str(output.agent_output))
            metrics_artifacts.subagent_outputs_by_id[f"agent_{idx + 1}"] = str(
                output.agent_output
            )

        metrics_artifacts.inter_agent_messages = 0
        metrics_artifacts.final_output = final_answer
        metrics_artifacts.pre_state_text = "\n".join(
            f"agent_{idx + 1}: {str(output.agent_output)}"
            for idx, output in enumerate(outputs)
        )
        metrics_artifacts.post_state_text = final_answer
        metrics_artifacts.finalize()

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],
            final_env_output=combined_env_status,
            metrics_artifacts=metrics_artifacts,
        )
