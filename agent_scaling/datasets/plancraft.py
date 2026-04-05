from typing import Any, Dict, List

from plancraft.simple import PlancraftExample

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance

DATASET_IDS = ["plancraft-test"]


@register_dataset_instance(DATASET_IDS)
class PlancraftInstance(PlancraftExample, DatasetInstance):
    def model_post_init(self, context: Any) -> None:
        self.expected_output = self.target
        self.slotted_inventory = {int(k): v for k, v in self.slotted_inventory.items()}

    def get_prompt_info(self) -> Dict[str, Any]:
        return {
            "inventory": self.inventory,
            "target": self.target,
        }


@register_dataset(DATASET_IDS)
class PlancraftDataset(Dataset):
    dataset_id: str = "plancraft-test"
    instances: List[PlancraftInstance]

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[PlancraftInstance]
    ) -> Dict[str, Any]:
        return {
            "success": instance_output.final_env_output.success
            if instance_output.final_env_output
            else False,
            "num_steps": instance_output.final_env_output.num_steps
            if instance_output.final_env_output
            else -1,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[PlancraftInstance]
    ) -> Dict[str, Any]:
        return self.get_instance_eval_output(instance_output)

    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        num_instances = len(eval_outputs)
        if num_instances == 0:
            return {
                "avg_success": 0.0,
                "avg_num_steps": 0.0,
                "num_instances": 0,
            }

        return {
            "avg_success": sum(
                bool(e.get("success", False))
                for e in eval_outputs
                if isinstance(e, dict)
            )
            / num_instances,
            "avg_num_steps": sum(
                e.get("num_steps", -1) if isinstance(e, dict) else -1
                for e in eval_outputs
            )
            / num_instances,
            "num_instances": num_instances,
        }
