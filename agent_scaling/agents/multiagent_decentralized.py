import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_scaling.datasets import DatasetInstance
from agent_scaling.utils.token_budget import TokenBudgetManager

from .multiagent_components.budgeting import MASBudgetAllocator
from .multiagent_components.conversation import CommunicationEvent, FinalAnswerCandidate
from .multiagent_components.system import BaseMultiAgentSystem
from .registry import register_agent


@register_agent("multi-agent-decentralized")
class DecentralizedMultiAgentSystem(BaseMultiAgentSystem):
    """Peer-to-peer debate without an orchestrator."""

    architecture_name = "multi-agent-decentralized"
    default_task_blurb = "peer collaborator"

    async def _run_architecture(
        self,
        instance: DatasetInstance,
        llm_params_dict: Dict[str, Any],
        budget_manager: Optional[TokenBudgetManager],
        budget_allocator: MASBudgetAllocator,
    ):
        shared_prompt_templates = self.get_dataset_prompt_templates(dataset_instance=instance)
        plan = self._build_default_parallel_plan(shared_prompt_templates)
        workers = {
            subtask.agent_id: self._create_worker(
                subtask=subtask,
                instance=instance,
                llm_params_dict=llm_params_dict,
                budget_manager=budget_manager,
                budget_allocator=budget_allocator,
            )
            for subtask in plan.subtasks
        }

        communication_events: List[CommunicationEvent] = []
        agent_findings = {agent_id: [] for agent_id in workers}
        round_results: Dict[str, Any] = {}
        completion_reason = "consensus_finalization"
        rounds_completed = 0

        for round_num in range(1, self.max_rounds + 1):
            rounds_completed = round_num
            if round_num == 1:
                round_results = await self._run_parallel_workers(
                    workers,
                    lambda worker: worker.run_independent_round(),
                )
            else:
                peer_messages = self._build_peer_messages(round_results)
                communication_events.extend(
                    self._peer_events_from_round_results(round_results, round_num)
                )
                round_results = await self._run_parallel_workers(
                    workers,
                    lambda worker: worker.process_peer_message(
                        peer_messages.get(worker.agent_id, "No peer findings available."),
                        sender_id="peer_network",
                        max_iterations=self.max_iterations_per_agent,
                    ),
                )

            for agent_id, result in round_results.items():
                if result.findings:
                    agent_findings[agent_id].append(result.findings)
            if any(result.env_status.success for result in round_results.values()):
                completion_reason = "worker_success"
                break

        final_candidates = await self._collect_final_candidates(workers)
        synthesized_answer = self._select_consensus_answer(final_candidates)
        if not synthesized_answer:
            synthesized_answer = self._select_success_answer(round_results)

        return self._build_result(
            plan=plan,
            workers=workers,
            agent_findings=agent_findings,
            synthesized_answer=synthesized_answer,
            total_rounds=rounds_completed,
            completion_reason=completion_reason,
            communication_events=communication_events,
            final_candidates=final_candidates,
        )

    def _build_peer_messages(self, round_results: Dict[str, Any]) -> Dict[str, str]:
        peer_messages: Dict[str, str] = {}
        for recipient_id in round_results:
            messages = []
            for sender_id, result in round_results.items():
                if sender_id == recipient_id or not result.findings:
                    continue
                messages.append(f"{sender_id}: {result.findings}")
            peer_messages[recipient_id] = "\n".join(messages) if messages else ""
        return peer_messages

    def _peer_events_from_round_results(
        self, round_results: Dict[str, Any], round_num: int
    ) -> List[CommunicationEvent]:
        events: List[CommunicationEvent] = []
        timestamp = datetime.now().isoformat()
        agent_ids = list(round_results.keys())
        for sender_id, result in round_results.items():
            if not result.findings:
                continue
            recipient_ids = [
                recipient_id for recipient_id in agent_ids if recipient_id != sender_id
            ]
            if not recipient_ids:
                continue
            events.append(
                CommunicationEvent(
                    round_num=round_num,
                    timestamp=timestamp,
                    sender_id=sender_id,
                    recipient_id="peer_broadcast",
                    recipient_ids=recipient_ids,
                    channel="peer",
                    message=result.findings,
                )
            )
        return events

    async def _collect_final_candidates(self, workers) -> List[FinalAnswerCandidate]:
        tasks = [asyncio.to_thread(worker.propose_final_answer) for worker in workers.values()]
        gathered = await asyncio.gather(*tasks)
        return [candidate for candidate in gathered if candidate.answer]

    def _select_consensus_answer(
        self, candidates: List[FinalAnswerCandidate]
    ) -> Optional[str]:
        if not candidates:
            return None
        normalized: Dict[str, List[FinalAnswerCandidate]] = {}
        for candidate in candidates:
            key = " ".join(candidate.answer.lower().split())
            if not key:
                continue
            normalized.setdefault(key, []).append(candidate)
        if not normalized:
            return None

        _, best_group = max(
            normalized.items(),
            key=lambda item: (len(item[1]), max(c.confidence for c in item[1])),
        )
        if len(best_group) / len(candidates) >= self.consensus_threshold:
            return best_group[0].answer

        fallback = max(
            candidates,
            key=lambda candidate: (candidate.confidence, len(candidate.rationale)),
        )
        return fallback.answer
