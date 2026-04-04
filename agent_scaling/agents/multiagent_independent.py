from typing import Any, Dict, Optional

from agent_scaling.datasets import DatasetInstance
from agent_scaling.utils.token_budget import TokenBudgetManager

from .multiagent_components.budgeting import MASBudgetAllocator
from .multiagent_components.system import BaseMultiAgentSystem
from .registry import register_agent


@register_agent("multi-agent-independent")
class IndependentMultiAgentSystem(BaseMultiAgentSystem):
    """Parallel workers with no peer exchange and only a final aggregation step."""

    architecture_name = "multi-agent-independent"
    default_task_blurb = "independent solver"

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

        round_results = await self._run_parallel_workers(
            workers,
            lambda worker: worker.run_independent_round(),
        )
        agent_findings = {
            agent_id: [result.findings] if result.findings else []
            for agent_id, result in round_results.items()
        }

        synthesized_answer = self._select_success_answer(round_results)
        completion_reason = (
            "worker_success"
            if any(result.env_status.success for result in round_results.values())
            else "final_synthesis"
        )
        if synthesized_answer is None:
            all_findings = self._format_findings_for_synthesis(agent_findings)
            if all_findings:
                synthesis_messages = self.prompts["lead_agent"].get_template(
                    "synthesis"
                ).compile(
                    **shared_prompt_templates,
                    all_findings=all_findings,
                )
                response = self._invoke_system_llm(
                    messages=synthesis_messages,
                    llm_params_dict=llm_params_dict,
                    budget_allocator=budget_allocator,
                    bucket="synthesis",
                )
                synthesized_answer = response.text().strip()

        return self._build_result(
            plan=plan,
            workers=workers,
            agent_findings=agent_findings,
            synthesized_answer=synthesized_answer,
            total_rounds=1,
            completion_reason=completion_reason,
        )
