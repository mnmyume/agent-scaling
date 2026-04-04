import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_scaling.datasets import DatasetInstance
from agent_scaling.utils.token_budget import TokenBudgetManager

from .multiagent_components.budgeting import MASBudgetAllocator
from .multiagent_components.conversation import CommunicationEvent
from .multiagent_components.memory import EnhancedMemory
from .multiagent_components.system import BaseMultiAgentSystem
from .registry import register_agent


@register_agent("multi-agent-hybrid")
class HybridMultiAgentSystem(BaseMultiAgentSystem):
    """Orchestrator-led execution with explicit bounded peer exchange."""

    architecture_name = "multi-agent-hybrid"
    default_task_blurb = "hybrid coordinator"

    async def _run_architecture(
        self,
        instance: DatasetInstance,
        llm_params_dict: Dict[str, Any],
        budget_manager: Optional[TokenBudgetManager],
        budget_allocator: MASBudgetAllocator,
    ):
        memory = EnhancedMemory()
        lead_agent = self._create_lead_agent(
            memory=memory,
            llm_params_dict=llm_params_dict,
            budget_manager=budget_manager,
            budget_allocator=budget_allocator,
        )
        shared_prompt_templates = self.get_dataset_prompt_templates(dataset_instance=instance)
        lead_agent.shared_prompt_templates = shared_prompt_templates
        lead_agent.memory.original_task = shared_prompt_templates["task_instance"]

        plan = lead_agent.analyze_query_and_plan(shared_prompt_templates)
        lead_agent.memory.execution_plan = plan
        lead_agent._create_subagents(plan, instance)
        workers = lead_agent.subagents
        subtasks = {subtask.agent_id: subtask for subtask in plan.subtasks}

        communication_events: List[CommunicationEvent] = []
        latest_results: Dict[str, Any] = {}
        completion_reason = "max_rounds_reached"
        start_time = time.time()
        rounds_completed = 0

        for round_num in range(1, self.max_rounds + 1):
            rounds_completed = round_num
            if time.time() - start_time > self.max_execution_time:
                completion_reason = "timeout"
                break

            active_workers = {
                agent_id: worker
                for agent_id, worker in workers.items()
                if worker.conv_history.status == "active"
                and not worker.should_stop_due_to_rate_limiting()
            }
            if not active_workers:
                completion_reason = "no_active_workers"
                break

            latest_results = await self._run_parallel_workers(
                active_workers,
                lambda worker: worker.process_orchestrator_message(
                    lead_agent._create_message_for_agent(
                        worker, subtasks[worker.agent_id], round_num
                    )
                ),
            )
            lead_agent._update_memory_with_turn_results(latest_results)
            if any(result.env_status.success for result in latest_results.values()):
                completion_reason = "worker_success"
                break

            if round_num <= self.peer_rounds and self.peer_fanout > 0:
                peer_results, peer_events = await self._run_peer_phase(
                    active_workers,
                    latest_results,
                    round_num=round_num,
                )
                communication_events.extend(peer_events)
                lead_agent._update_memory_with_turn_results(peer_results)
                latest_results = peer_results
                if any(result.env_status.success for result in latest_results.values()):
                    completion_reason = "worker_success"
                    break

            if lead_agent._should_stop_orchestration(round_num, latest_results):
                completion_reason = "orchestrator_stop"
                break

        synthesized_answer = self._select_success_answer(latest_results)
        if synthesized_answer is None:
            synthesized_answer = lead_agent._synthesize_findings()

        return self._build_result(
            plan=plan,
            workers=workers,
            agent_findings=lead_agent.memory.agent_findings,
            synthesized_answer=synthesized_answer,
            total_rounds=rounds_completed,
            completion_reason=completion_reason,
            communication_events=communication_events,
            lead_agent_conversation=lead_agent.conv_history,
        )

    async def _run_peer_phase(
        self,
        workers,
        round_results,
        round_num: int,
    ):
        peer_messages = self._build_limited_peer_messages(round_results)
        peer_events = self._peer_events_from_messages(peer_messages, round_num)
        peer_results = await self._run_parallel_workers(
            workers,
            lambda worker: worker.process_peer_message(
                peer_messages.get(worker.agent_id, "No peer findings available."),
                sender_id="peer_network",
                max_iterations=self.peer_max_iterations,
            ),
        )
        return peer_results, peer_events

    def _build_limited_peer_messages(self, round_results) -> Dict[str, str]:
        agent_ids = list(round_results.keys())
        peer_messages: Dict[str, str] = {agent_id: "" for agent_id in agent_ids}
        for index, agent_id in enumerate(agent_ids):
            selected_peers = []
            for offset in range(1, self.peer_fanout + 1):
                peer_id = agent_ids[(index + offset) % len(agent_ids)]
                result = round_results.get(peer_id)
                if result and result.findings:
                    selected_peers.append(f"{peer_id}: {result.findings}")
            peer_messages[agent_id] = "\n".join(selected_peers)
        return peer_messages

    def _peer_events_from_messages(
        self, peer_messages: Dict[str, str], round_num: int
    ) -> List[CommunicationEvent]:
        events: List[CommunicationEvent] = []
        timestamp = datetime.now().isoformat()
        for recipient_id, message in peer_messages.items():
            if not message:
                continue
            for line in message.splitlines():
                sender_id, _, content = line.partition(": ")
                events.append(
                    CommunicationEvent(
                        round_num=round_num,
                        timestamp=timestamp,
                        sender_id=sender_id,
                        recipient_id=recipient_id,
                        channel="peer",
                        message=content,
                    )
                )
        return events
