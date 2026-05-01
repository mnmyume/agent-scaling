"""Independent multi-agent system: N agents work independently, best result wins."""
import asyncio
import os.path as osp
import time
from typing import Dict, Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.tracing import (
    make_trace_event,
    subagent_trace_events,
    write_trace_events,
)
from agent_scaling.utils import write_yaml

from .multiagent_components.mas_subagent import WorkerSubagent
from .multiagent_components.memory import EnhancedMemory
from .registry import register_agent


@register_agent("multi-agent-independent")
class IndependentMultiAgentSystem(AgentSystemWithTools):
    """Independent multi-agent system: N agents work in parallel with NO coordination.

    Each agent independently attempts to solve the full task. The first agent
    to successfully complete the task determines the result. This is a best-of-N
    baseline that measures the value of pure parallelism without coordination.
    """

    required_prompts = ["subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = EnhancedMemory()
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.subagents: Dict[str, WorkerSubagent] = {}

        logger.info(
            f"IndependentMultiAgentSystem: {n_base_agents} agents, "
            f"NO coordination (best-of-N)"
        )

    def _create_subagents(self, task_instance: DatasetInstance):
        """Create N fully independent agents, each working on the full task."""
        self.subagents = {}

        for i in range(self.n_base_agents):
            agent_id = f"agent_{i + 1}"
            subagent = WorkerSubagent.init_from_agent(
                agent=self,
                agent_id=agent_id,
                objective="Solve the task completely on your own",
                original_query=self.memory.original_task,
                strategy=f"Independent agent {i + 1}",
                task_instance=task_instance,
                min_iterations_per_agent=self.min_iterations_per_agent,
                max_iterations_per_agent=self.max_iterations_per_agent,
            )
            self.subagents[agent_id] = subagent

        logger.info(f"Created {len(self.subagents)} independent agents (no coordination)")

    def _auto_submit(self) -> str:
        """Auto-submit using the first available agent's environment."""
        for agent_id, agent in self.subagents.items():
            env = agent.env
            if env.env_done():
                continue
            submit_tool_name = None
            if "submit_patch" in env.tools:
                submit_tool_name = "submit_patch"
            elif "submit" in env.tools:
                submit_tool_name = "submit"
            if submit_tool_name:
                logger.info(f"Auto-submitting via {agent_id} using {submit_tool_name}")
                try:
                    tool_call = {
                        "name": submit_tool_name,
                        "args": {"reasoning": "Auto-submit: independent agent budget exhausted"},
                        "id": "auto_submit_independent",
                        "type": "tool_call",
                    }
                    tool_msg = env.execute_tool(tool_call)
                    return str(tool_msg.content)
                except Exception as e:
                    logger.warning(f"Auto {submit_tool_name} failed: {e}")
            break
        return ""

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
        start_time = time.time()

        # Get shared templates
        shared_prompt_templates = self.get_dataset_prompt_templates(
            dataset_instance=instance
        )
        self.memory.original_task = shared_prompt_templates.get(
            "task_instance", str(instance.get_prompt_info())
        )

        # Create independent agents
        self._create_subagents(instance)

        # Run ALL agents in parallel with NO coordination between rounds
        # Each agent gets a single "go" message and works independently
        message = (
            "Solve the task completely. Work systematically and submit when done.\n\n"
            "CRITICAL: You MUST make code changes and submit within your iteration budget. "
            "Do NOT spend more than half your iterations on exploration/analysis. "
            "After understanding the problem, immediately start implementing the fix. "
            "If you reach 50% of your budget without editing any files, STOP exploring "
            "and START implementing your best solution immediately."
        )

        tasks = []
        for agent_id, subagent in self.subagents.items():
            task = asyncio.create_task(
                asyncio.to_thread(subagent.process_orchestrator_message, message)
            )
            tasks.append((agent_id, task))

        # Wait for all agents (generous timeout since they work independently)
        time_limit = getattr(instance, "time_limit", 600)
        done, pending = await asyncio.wait(
            [t for _, t in tasks], timeout=time_limit
        )
        for t in pending:
            t.cancel()

        # Collect results
        results = {}
        for agent_id, task in tasks:
            if task in done:
                try:
                    results[agent_id] = await task
                except Exception as e:
                    logger.warning(f"Agent {agent_id} failed: {e}")

        # Find winning agent (first one that solved it)
        final_answer = ""
        final_env_status = None
        winner = None

        for agent_id, agent in self.subagents.items():
            if agent.env.env_done():
                status = agent.env.env_status()
                if hasattr(status, 'success') and status.success:
                    winner = agent_id
                    final_answer = agent.conv_history.last_outgoing_external_message or ""
                    final_env_status = status
                    break

        # If no successful agent, take the first done agent
        if final_env_status is None:
            for agent_id, agent in self.subagents.items():
                if agent.env.env_done():
                    final_answer = agent.conv_history.last_outgoing_external_message or ""
                    final_env_status = agent.env.env_status()
                    winner = agent_id
                    break

        # If no agent submitted, auto-submit
        if final_env_status is None:
            final_answer = self._auto_submit()
            for agent in self.subagents.values():
                if agent.env.env_done():
                    final_env_status = agent.env.env_status()
                    break
            if final_env_status is None:
                final_env_status = next(iter(self.subagents.values())).env.env_status()

        execution_time = time.time() - start_time
        total_iterations = sum(
            a.conv_history.total_iterations for a in self.subagents.values()
        )
        logger.info(
            f"Independent processing completed in {execution_time:.2f}s "
            f"with {total_iterations} total iterations across {len(self.subagents)} agents. "
            f"Winner: {winner}"
        )

        if instance_dir is not None:
            output_data = {
                "architecture": "independent",
                "n_agents": self.n_base_agents,
                "total_iterations": total_iterations,
                "execution_time": execution_time,
                "winner": winner,
            }
            write_yaml(
                output_data,
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )
            trace_events = subagent_trace_events(
                self.subagents,
                instance_idx=instance_idx,
            )
            trace_events.append(
                make_trace_event(
                    "final_answer",
                    instance_idx=instance_idx,
                    agent_id=winner or "system",
                    content=final_answer,
                    metadata={"winner": winner},
                )
            )
            write_trace_events(
                trace_events,
                osp.join(instance_dir, "trace_events.jsonl"),
            )

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],
            final_env_output=final_env_status,
        )
