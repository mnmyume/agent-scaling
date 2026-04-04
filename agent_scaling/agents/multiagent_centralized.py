from typing import Any, Dict, Optional

from agent_scaling.datasets import DatasetInstance
from agent_scaling.utils.token_budget import TokenBudgetManager

from .multiagent_components.budgeting import MASBudgetAllocator
from .multiagent_components.memory import EnhancedMemory
from .multiagent_components.system import BaseMultiAgentSystem
from .registry import register_agent


@register_agent("multi-agent-centralized")
class CentralizedMultiAgentSystem(BaseMultiAgentSystem):
    """Centralized multi-agent system with an orchestrator coordinating workers."""

    architecture_name = "multi-agent-centralized"
    default_task_blurb = "task coordinator"

    async def _run_architecture(
        self,
        instance: DatasetInstance,
        llm_params_dict: Dict[str, Any],
        budget_manager: Optional[TokenBudgetManager],
        budget_allocator: MASBudgetAllocator,
    ):
        lead_agent = self._create_lead_agent(
            memory=EnhancedMemory(),
            llm_params_dict=llm_params_dict,
            budget_manager=budget_manager,
            budget_allocator=budget_allocator,
        )
        return await lead_agent.orchestrate_work(
            task_instance=instance,
            llm_params_dict=llm_params_dict,
        )
