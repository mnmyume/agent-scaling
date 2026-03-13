from __future__ import annotations

import ast
import csv
import os
from typing import Any, Dict, List, Optional, Union

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance
from agent_scaling.utils import get_root_dir
from agent_scaling.workbench.sandbox import (
    WorkbenchSandbox,
    end_date_minor_error,
    evaluate_actions,
    format_func_call,
    meeting_start_time_error,
    parse_action_to_tool_and_args,
)

DATASET_IDS = ["workbench"]


@register_dataset_instance(DATASET_IDS)
class WorkbenchInstance(DatasetInstance):
    query: str
    answer: List[str]
    domains: Optional[List[str]] = None
    subset: Optional[str] = None
    base_template: Optional[str] = None
    chosen_template: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        self.expected_output = self.answer

    def get_prompt_info(self) -> Dict[str, str]:
        return {"query": self.query}


@register_dataset(DATASET_IDS)
class WorkbenchDataset(Dataset):
    """WorkBench dataset.

    Source: https://github.com/olly-styles/WorkBench (MIT License)
    Paper: https://arxiv.org/abs/2405.00823
    """

    dataset_id: str = "workbench"
    instances: List[WorkbenchInstance]

    @classmethod
    def from_csv(cls, csv_path: str, **kwargs) -> "WorkbenchDataset":
        instances: List[WorkbenchInstance] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = (row.get("query") or "").strip()
                if not query:
                    continue
                raw_answer = row.get("answer") or "[]"
                raw_domains = row.get("domains") or "[]"
                try:
                    answer = ast.literal_eval(raw_answer)
                except Exception:
                    answer = []
                try:
                    domains = ast.literal_eval(raw_domains)
                except Exception:
                    domains = []
                if not isinstance(answer, list):
                    answer = []
                if not isinstance(domains, list):
                    domains = []
                instances.append(
                    WorkbenchInstance(
                        query=query,
                        answer=[str(a) for a in answer],
                        domains=[str(d) for d in domains],
                        subset=row.get("subset") or None,
                        base_template=row.get("base_template") or None,
                        chosen_template=row.get("chosen_template") or None,
                    )
                )
        return cls(instances=instances, **kwargs)

    def _sandbox_factory(self) -> WorkbenchSandbox:
        repo_root = get_root_dir()
        data_dir = os.path.join(repo_root, "datasets", "workbench")
        return WorkbenchSandbox(data_dir)

    def _extract_predicted_actions(
        self, instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance]
    ) -> List[str]:
        actions: List[str] = []
        for step in getattr(instance_output, "trajectory", []) or []:
            parsed = parse_action_to_tool_and_args(step.action)
            if not parsed:
                continue
            tool_name, args, order = parsed
            if tool_name == "done":
                continue
            actions.append(format_func_call(tool_name, args, arg_order=order))
        return actions

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance]
    ) -> Dict[str, Any]:
        instance = instance_output.data_instance
        predicted = self._extract_predicted_actions(instance_output)
        ground_truth = instance.expected_output or []
        return {
            "query": instance.query,
            "predicted_actions": predicted,
            "ground_truth_actions": ground_truth,
            "subset": instance.subset,
            "domains": instance.domains,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance]
    ) -> Dict[str, Union[int, float, str]]:
        instance = instance_output.data_instance
        ground_truth: List[str] = list(instance.expected_output or [])
        predicted = self._extract_predicted_actions(instance_output)

        # Some multi-agent systems don't currently return a single trajectory.
        # In that case, fall back to environment-reported success (if any).
        if not predicted and instance_output.final_env_output is not None:
            correct = int(bool(instance_output.final_env_output.success))
            return {
                "correct": correct,
                "exact_match": -1,
                "unwanted_side_effects": -1,
                "num_predicted_actions": 0,
                "num_ground_truth_actions": len(ground_truth),
                "subset": instance.subset or "",
            }

        eval_result = evaluate_actions(
            predicted_actions=predicted,
            ground_truth_actions=ground_truth,
            sandbox_factory=self._sandbox_factory,
        )

        correct = int(eval_result.get("correct", 0))
        exact_match = int(eval_result.get("exact_match", 0))
        unwanted_side_effects = int(eval_result.get("unwanted_side_effects", 0))

        # Error analysis fields used in the upstream evaluation script.
        pred_str = str(predicted)
        wrong_email = int(("@example" in pred_str) and ("@atlas" not in pred_str) and (not correct))
        no_actions = int(len(predicted) == 0)
        ed_minor = int(end_date_minor_error(ground_truth, predicted) and (not correct))
        meeting_time_err = int(meeting_start_time_error(ground_truth, predicted) and (not correct))

        return {
            "correct": correct,
            "exact_match": exact_match,
            "unwanted_side_effects": unwanted_side_effects,
            "no_actions": no_actions,
            "wrong_email": wrong_email,
            "end_date_minor_error": ed_minor,
            "meeting_start_time_error": meeting_time_err,
            "num_predicted_actions": len(predicted),
            "num_ground_truth_actions": len(ground_truth),
            "subset": instance.subset or "",
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        rows = [e for e in eval_outputs if isinstance(e, dict)]
        n = len(rows)
        if n == 0:
            return {
                "accuracy": 0.0,
                "exact_match_rate": 0.0,
                "side_effect_rate": 0.0,
                "num_instances": 0,
            }

        correct = [int(r.get("correct", 0)) for r in rows]
        exact = [int(r.get("exact_match", 0)) for r in rows if int(r.get("exact_match", 0)) >= 0]
        side = [
            int(r.get("unwanted_side_effects", 0))
            for r in rows
            if int(r.get("unwanted_side_effects", 0)) >= 0
        ]

        accuracy = sum(correct) / n
        exact_match_rate = (sum(exact) / len(exact)) if exact else 0.0
        side_effect_rate = (sum(side) / len(side)) if side else 0.0

        return {
            "accuracy": accuracy,
            "exact_match_rate": exact_match_rate,
            "side_effect_rate": side_effect_rate,
            "num_instances": n,
            "n_exact_match_measured": len(exact),
            "n_side_effects_measured": len(side),
        }

