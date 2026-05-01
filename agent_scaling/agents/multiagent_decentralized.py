"""Decentralized multi-agent system: no orchestrator, peer debate with consensus."""
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

from .multiagent_components.conversation import SubAgentRoundResult
from .multiagent_components.mas_subagent import WorkerSubagent
from .multiagent_components.memory import EnhancedMemory
from .registry import register_agent


@register_agent("multi-agent-decentralized")
class DecentralizedMultiAgentSystem(AgentSystemWithTools):
    """Decentralized multi-agent system with peer debate and consensus.

    Unlike centralized: no lead agent. All agents work independently on the
    full task, share findings between rounds, and the first to solve wins.
    """

    required_prompts = ["subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 25,
        max_rounds: int = 10,
        consensus_threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = EnhancedMemory()
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.subagents: Dict[str, WorkerSubagent] = {}

        logger.info(
            f"DecentralizedMultiAgentSystem: {n_base_agents} agents, "
            f"{max_rounds} max rounds, {consensus_threshold} consensus threshold"
        )

    def _create_subagents(self, task_instance: DatasetInstance):
        """Create N independent agents, each working on the full task."""
        self.subagents = {}
        filtered_kwargs = {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in [
                "llm_w_tools", "env", "llm", "dataset", "prompts",
                "env_prompts", "tools", "memory", "subagents",
                "n_base_agents", "min_iterations_per_agent", "max_rounds",
                "consensus_threshold",
            ]
            and not k.startswith("_")
        }

        for i in range(self.n_base_agents):
            agent_id = f"agent_{i + 1}"
            # Each agent gets a slightly different strategy to promote diversity
            strategies = [
                "Analyze the problem systematically and implement a targeted fix",
                "Explore the codebase broadly first, then identify and fix the root cause",
                "Focus on understanding the test expectations, then work backwards to the fix",
            ]
            strategy = strategies[i % len(strategies)]

            subagent = WorkerSubagent.init_from_agent(
                agent=self,
                agent_id=agent_id,
                objective="Solve the full task independently",
                original_query=self.memory.original_task,
                strategy=strategy,
                task_instance=task_instance,
                min_iterations_per_agent=self.min_iterations_per_agent,
                max_iterations_per_agent=self.max_iterations_per_agent,
            )
            self.subagents[agent_id] = subagent

        logger.info(f"Created {len(self.subagents)} independent agents")

    def _build_peer_context(self, current_agent_id: str) -> str:
        """Build a summary of peer findings for an agent."""
        peer_findings = []
        for agent_id, agent in self.subagents.items():
            if agent_id == current_agent_id:
                continue
            findings = agent.conv_history.last_outgoing_external_message
            if findings:
                peer_findings.append(f"- {agent_id}: {findings[:300]}")
        if peer_findings:
            return "Peer agent findings:\n" + "\n".join(peer_findings)
        return "No peer findings yet."

    def _auto_submit(self, synthesized_answer: str) -> str:
        """Auto-submit using the first available agent's environment."""
        for agent_id, agent in self.subagents.items():
            env = agent.env
            if env.env_done():
                break
            reasoning = (synthesized_answer or "Decentralized consensus")[:500]
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
                        "args": {"reasoning": reasoning},
                        "id": "auto_submit_decentralized",
                        "type": "tool_call",
                    }
                    tool_msg = env.execute_tool(tool_call)
                    return str(tool_msg.content)
                except Exception as e:
                    logger.warning(f"Auto {submit_tool_name} failed: {e}")
            break
        return synthesized_answer

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

        # Run rounds of independent work with peer sharing
        for round_num in range(1, self.max_rounds + 1):
            time_limit = getattr(instance, "time_limit", 600)
            if time.time() - start_time > time_limit:
                logger.warning("Execution timeout reached")
                break

            logger.info(f"\n=== Decentralized Round {round_num} ===")

            # Build peer context for each agent
            messages_for_agents = {}
            for agent_id in self.subagents:
                peer_context = self._build_peer_context(agent_id)
                if round_num == 1:
                    msg = f"Work on your assigned strategy. {peer_context}"
                else:
                    msg = (
                        f"Round {round_num}: Continue working. "
                        f"Consider your peers' progress:\n{peer_context}"
                    )
                messages_for_agents[agent_id] = msg

            # Run all agents in parallel
            active_agents = [
                aid for aid, a in self.subagents.items()
                if a.conv_history.status == "active"
                and not a.should_stop_due_to_rate_limiting()
            ]

            if not active_agents:
                logger.info("No active agents remaining")
                break

            tasks = []
            for agent_id in active_agents:
                subagent = self.subagents[agent_id]
                msg = messages_for_agents[agent_id]
                task = asyncio.create_task(
                    asyncio.to_thread(subagent.process_orchestrator_message, msg)
                )
                tasks.append((agent_id, task))

            # Wait for results
            done, pending = await asyncio.wait(
                [t for _, t in tasks], timeout=300
            )
            for t in pending:
                t.cancel()

            results: Dict[str, SubAgentRoundResult] = {}
            for agent_id, task in tasks:
                if task in done:
                    try:
                        results[agent_id] = await task
                    except Exception as e:
                        logger.warning(f"Agent {agent_id} failed: {e}")

            # Check if any agent solved it
            any_done = any(
                self.subagents[aid].env.env_done() for aid in results
            )
            if any_done:
                logger.info(f"Solution found in round {round_num}")
                break

            # Update memory
            for agent_id, result in results.items():
                if result.findings:
                    self.memory.add_findings(agent_id, result.findings)

        # Auto-submit if no agent submitted
        any_done = any(a.env.env_done() for a in self.subagents.values())
        final_answer = ""
        if not any_done:
            # Synthesize from all findings
            all_findings = []
            for agent_id, findings_list in self.memory.agent_findings.items():
                for f in findings_list:
                    all_findings.append(f"{agent_id}: {f}")
            synthesis = "\n".join(all_findings) if all_findings else "No findings"
            final_answer = self._auto_submit(synthesis)
        else:
            for a in self.subagents.values():
                if a.env.env_done():
                    final_answer = a.conv_history.last_outgoing_external_message or ""
                    break

        # Get final env status
        final_env_status = None
        for agent in self.subagents.values():
            if agent.env.env_done():
                final_env_status = agent.env.env_status()
                break
        if final_env_status is None:
            for agent in self.subagents.values():
                final_env_status = agent.env.env_status()
                break

        execution_time = time.time() - start_time
        total_iterations = sum(
            a.conv_history.total_iterations for a in self.subagents.values()
        )
        logger.info(
            f"Decentralized processing completed in {execution_time:.2f}s "
            f"with {total_iterations} total iterations across {len(self.subagents)} agents"
        )

        if instance_dir is not None:
            output_data = {
                "architecture": "decentralized",
                "n_agents": self.n_base_agents,
                "total_iterations": total_iterations,
                "execution_time": execution_time,
                "agent_findings": {
                    aid: findings
                    for aid, findings in self.memory.agent_findings.items()
                },
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
                    agent_id="system",
                    content=final_answer,
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
