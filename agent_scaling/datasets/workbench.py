from __future__ import annotations

import ast
import re
from collections import deque
from typing import Any, Dict, List

import pandas as pd

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance
from agent_scaling.env.workbench_utils import (
    WorkbenchAction,
    WorkbenchSandbox,
    WORKBENCH_REFERENCE_TIME,
    execute_workbench_actions,
    normalize_workbench_tool_name,
    parse_workbench_gold_action,
    workbench_states_equal,
)
from agent_scaling.utils import read_json

DATASET_IDS = [
    "workbench",
    "workbench-analytics",
    "workbench-calendar",
    "workbench-customer-relationship-manager",
    "workbench-email",
    "workbench-multi-domain",
    "workbench-project-management",
]

DOMAIN_ALIASES = {
    "crm": "customer_relationship_manager",
}


@register_dataset_instance(DATASET_IDS)
class WorkbenchInstance(DatasetInstance):
    query: str
    answer: List[str] | str
    base_template: str | None = None
    chosen_template: str | None = None
    domains: List[str] | str | None = None
    index: int | None = None

    def model_post_init(self, context: Any) -> None:
        if isinstance(self.answer, str):
            self.answer = ast.literal_eval(self.answer)
        if isinstance(self.domains, str):
            self.domains = ast.literal_eval(self.domains)
        self.domains = [
            DOMAIN_ALIASES.get(domain, domain) for domain in (self.domains or [])
        ]
        self.expected_output = self.answer

    def get_prompt_info(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "current_datetime": WORKBENCH_REFERENCE_TIME.strftime("%Y-%m-%d %H:%M:%S"),
            "today_date": WORKBENCH_REFERENCE_TIME.strftime("%Y-%m-%d"),
            "tomorrow_date": _get_tomorrow_date(),
        }


def _get_tomorrow_date() -> str:
    return (WORKBENCH_REFERENCE_TIME + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _interleave_instances_by_domain_signature(
    instances: List[WorkbenchInstance],
) -> List[WorkbenchInstance]:
    buckets: Dict[tuple[str, ...], deque[WorkbenchInstance]] = {}
    for instance in instances:
        signature = tuple(instance.domains or [])
        buckets.setdefault(signature, deque()).append(instance)

    interleaved: List[WorkbenchInstance] = []
    while buckets:
        empty_signatures: List[tuple[str, ...]] = []
        for signature, bucket in buckets.items():
            if bucket:
                interleaved.append(bucket.popleft())
            if not bucket:
                empty_signatures.append(signature)
        for signature in empty_signatures:
            buckets.pop(signature, None)
    return interleaved


def _serialize_actions(actions: List[WorkbenchAction]) -> List[str]:
    return [action.to_canonical_call() for action in actions]


def _extract_actions_from_runtime_metrics(
    runtime_metrics: Dict[str, Any],
) -> tuple[List[WorkbenchAction], List[str]]:
    actions: List[WorkbenchAction] = []
    errors: List[str] = []
    for item in runtime_metrics.get("tool_log", []):
        tool_name = item.get("tool_name")
        if tool_name == "done" or not tool_name:
            continue
        try:
            normalized_tool_name = normalize_workbench_tool_name(tool_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        arguments = item.get("arguments") or {}
        if not isinstance(arguments, dict):
            errors.append(f"Invalid tool arguments for {tool_name}: {arguments}")
            continue
        actions.append(
            WorkbenchAction(tool_name=normalized_tool_name, arguments=dict(arguments))
        )
    return actions, errors


def _best_effort_parse_trajectory_action(action: str) -> WorkbenchAction:
    match = re.fullmatch(r"([a-zA-Z0-9_]+)\((.*)\)", action.strip(), re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse trajectory action: {action}")
    tool_name = normalize_workbench_tool_name(match.group(1))
    args_src = match.group(2).strip()
    if not args_src:
        return WorkbenchAction(tool_name=tool_name, arguments={})

    key_matches = list(re.finditer(r"(^|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)=", args_src))
    if not key_matches:
        raise ValueError(f"Could not parse trajectory action args: {action}")

    arguments: Dict[str, Any] = {}
    for idx, key_match in enumerate(key_matches):
        key = key_match.group(2)
        value_start = key_match.end()
        value_end = (
            key_matches[idx + 1].start() if idx + 1 < len(key_matches) else len(args_src)
        )
        raw_value = args_src[value_start:value_end].strip().rstrip(",")
        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {
            '"',
            "'",
        }:
            raw_value = raw_value[1:-1]
        arguments[key] = raw_value
    return WorkbenchAction(tool_name=tool_name, arguments=arguments)


def _extract_actions_from_trajectory(
    instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance],
) -> tuple[List[WorkbenchAction], List[str]]:
    actions: List[WorkbenchAction] = []
    errors: List[str] = []
    for step in instance_output.trajectory:
        action = step.action.strip()
        if not action or action.startswith("done("):
            continue
        try:
            actions.append(_best_effort_parse_trajectory_action(action))
        except Exception as exc:
            errors.append(str(exc))
    return actions, errors


def _get_runtime_metrics(
    instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance],
) -> Dict[str, Any] | None:
    if instance_output.runtime_metrics is not None:
        return instance_output.runtime_metrics
    if instance_output.runtime_metrics_path:
        return read_json(instance_output.runtime_metrics_path)
    return None


def _get_predicted_actions(
    instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance],
) -> tuple[List[WorkbenchAction], str, str | None]:
    runtime_metrics = _get_runtime_metrics(instance_output)
    if runtime_metrics is not None:
        actions, errors = _extract_actions_from_runtime_metrics(runtime_metrics)
        return actions, "runtime_metrics", "; ".join(errors) if errors else None

    actions, errors = _extract_actions_from_trajectory(instance_output)
    return actions, "trajectory", "; ".join(errors) if errors else None


def _get_gold_actions(instance: WorkbenchInstance) -> List[WorkbenchAction]:
    return [parse_workbench_gold_action(action) for action in instance.answer]


def _normalize_exact_match_value(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _exact_match_signature(
    action: WorkbenchAction,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        action.canonical_tool_name.lower(),
        tuple(
            sorted(
                (key.lower(), _normalize_exact_match_value(value))
                for key, value in action.arguments.items()
            )
        ),
    )


def _is_exact_match(
    predicted_actions: List[WorkbenchAction], gold_actions: List[WorkbenchAction]
) -> bool:
    predicted_with_side_effects = sorted(
        _exact_match_signature(action)
        for action in predicted_actions
        if action.has_side_effect()
    )
    gold_with_side_effects = sorted(
        _exact_match_signature(action) for action in gold_actions
    )
    return predicted_with_side_effects == gold_with_side_effects


def _is_correct(
    predicted_actions: List[WorkbenchAction],
    gold_actions: List[WorkbenchAction],
    extraction_error: str | None,
) -> bool:
    if extraction_error:
        return False
    predicted_state = execute_workbench_actions(predicted_actions)
    gold_state = execute_workbench_actions(gold_actions)
    return workbench_states_equal(predicted_state, gold_state)


def _has_unwanted_side_effects(
    predicted_actions: List[WorkbenchAction], gold_actions: List[WorkbenchAction]
) -> bool:
    original_state = WorkbenchSandbox().snapshot_state()
    predicted_state = execute_workbench_actions(predicted_actions)
    state_changed = not workbench_states_equal(predicted_state, original_state)
    return state_changed and not _is_correct(predicted_actions, gold_actions, None)


@register_dataset(DATASET_IDS)
class WorkbenchDataset(Dataset):
    dataset_id: str = "workbench"
    instances: List[WorkbenchInstance]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.dataset_id == "workbench":
            self.instances = _interleave_instances_by_domain_signature(self.instances)

    def _evaluate_instance(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance],
    ) -> Dict[str, Any]:
        predicted_actions, extraction_source, extraction_error = _get_predicted_actions(
            instance_output
        )
        gold_actions = _get_gold_actions(instance_output.data_instance)
        num_steps = (
            instance_output.final_env_output.num_steps
            if instance_output.final_env_output is not None
            else -1
        )
        return {
            "correct": _is_correct(predicted_actions, gold_actions, extraction_error),
            "exact_match": _is_exact_match(predicted_actions, gold_actions),
            "unwanted_side_effects": _has_unwanted_side_effects(
                predicted_actions, gold_actions
            ),
            "predicted_actions": _serialize_actions(predicted_actions),
            "expected_actions": _serialize_actions(gold_actions),
            "num_predicted_actions": len(predicted_actions),
            "num_expected_actions": len(gold_actions),
            "num_steps": num_steps,
            "action_extraction_source": extraction_source,
            "action_extraction_error": extraction_error,
        }

    def get_instance_eval_output(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance],
    ) -> Dict[str, Any]:
        evaluated = self._evaluate_instance(instance_output)
        return {
            "correct": evaluated["correct"],
            "exact_match": evaluated["exact_match"],
            "unwanted_side_effects": evaluated["unwanted_side_effects"],
            "predicted_actions": evaluated["predicted_actions"],
            "expected_actions": evaluated["expected_actions"],
            "action_extraction_source": evaluated["action_extraction_source"],
            "action_extraction_error": evaluated["action_extraction_error"],
        }

    def get_instance_eval_metrics(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance],
    ) -> Dict[str, Any]:
        evaluated = self._evaluate_instance(instance_output)
        return {
            "correct": evaluated["correct"],
            "exact_match": evaluated["exact_match"],
            "unwanted_side_effects": evaluated["unwanted_side_effects"],
            "num_predicted_actions": evaluated["num_predicted_actions"],
            "num_expected_actions": evaluated["num_expected_actions"],
            "num_steps": evaluated["num_steps"],
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        num_instances = len(eval_outputs)
        if num_instances == 0:
            return {
                "avg_correct": 0.0,
                "avg_exact_match": 0.0,
                "avg_unwanted_side_effects": 0.0,
                "avg_num_predicted_actions": 0.0,
                "avg_num_expected_actions": 0.0,
                "avg_num_steps": 0.0,
                "num_instances": 0,
            }

        valid_outputs = [output for output in eval_outputs if isinstance(output, dict)]
        return {
            "avg_correct": sum(bool(output.get("correct", False)) for output in valid_outputs)
            / num_instances,
            "avg_exact_match": sum(
                bool(output.get("exact_match", False)) for output in valid_outputs
            )
            / num_instances,
            "avg_unwanted_side_effects": sum(
                bool(output.get("unwanted_side_effects", False))
                for output in valid_outputs
            )
            / num_instances,
            "avg_num_predicted_actions": sum(
                int(output.get("num_predicted_actions", 0)) for output in valid_outputs
            )
            / num_instances,
            "avg_num_expected_actions": sum(
                int(output.get("num_expected_actions", 0)) for output in valid_outputs
            )
            / num_instances,
            "avg_num_steps": sum(int(output.get("num_steps", -1)) for output in valid_outputs)
            / num_instances,
            "num_instances": num_instances,
        }
