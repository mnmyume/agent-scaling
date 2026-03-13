import ast
import json
import math
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance

DATASET_IDS = ["workbench", "workbench_sampled"]


@register_dataset_instance(DATASET_IDS)
class WorkbenchInstance(DatasetInstance):
    task: str
    expected_calls: Optional[List[Dict[str, Any]]] = None
    expected_tool_calls: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None
    task_id: Optional[str] = None

    def model_post_init(self, __context):
        if self.expected_calls is None and self.expected_tool_calls is not None:
            self.expected_calls = self.expected_tool_calls
        if self.expected_calls is not None:
            self.expected_output = self.expected_calls

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "task": self.task,
            "context": self.context or "",
        }


@register_dataset(DATASET_IDS)
class WorkbenchDataset(Dataset):
    dataset_id: str = "workbench"
    instances: List[WorkbenchInstance]

    def _try_parse_date(self, value: str) -> Optional[date]:
        value = value.strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except Exception:
                continue
        return None

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            parsed_date = self._try_parse_date(value)
            if parsed_date is not None:
                return parsed_date
            return value.strip().lower()
        return value

    def _values_equal(self, a: Any, b: Any) -> bool:
        a_norm = self._normalize_value(a)
        b_norm = self._normalize_value(b)
        if isinstance(a_norm, date) and isinstance(b_norm, date):
            return abs((a_norm - b_norm).days) <= 1
        if isinstance(a_norm, float) and isinstance(b_norm, float):
            return math.isclose(a_norm, b_norm, rel_tol=1e-3, abs_tol=1e-3)
        return a_norm == b_norm

    def _normalize_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        name = call.get("name") or call.get("tool") or call.get("tool_name")
        args = call.get("args") or call.get("arguments") or call.get("parameters") or {}
        if args is None:
            args = {}
        return {
            "name": name,
            "args": {k: self._normalize_value(v) for k, v in args.items()},
        }

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or "(" not in action or not action.endswith(")"):
            return None
        name, _, arg_str = action.partition("(")
        name = name.strip()
        arg_str = arg_str[:-1].strip()
        args: Dict[str, Any] = {}
        if arg_str:
            parts = [p.strip() for p in arg_str.split(",") if p.strip()]
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    value_obj = ast.literal_eval(value)
                except Exception:
                    value_obj = value
                args[key] = value_obj
        return {"name": name, "args": args}

    def _extract_pred_calls(
        self, instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance]
    ) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        if hasattr(instance_output, "trajectory") and instance_output.trajectory:
            for step in instance_output.trajectory:
                parsed = self._parse_action(step.action)
                if parsed:
                    calls.append(parsed)
        else:
            # Try to parse agent_output as JSON list
            agent_output = instance_output.agent_output
            if isinstance(agent_output, list):
                calls = agent_output
            elif isinstance(agent_output, str):
                try:
                    parsed = json.loads(agent_output)
                    if isinstance(parsed, list):
                        calls = parsed
                except Exception:
                    calls = []
        # Filter out done calls
        calls = [c for c in calls if (c.get("name") or c.get("tool")) != "done"]
        return calls

    def _calls_match(
        self, expected: List[Dict[str, Any]], predicted: List[Dict[str, Any]]
    ) -> bool:
        if expected is None:
            return False
        exp_norm = [self._normalize_call(c) for c in expected]
        pred_norm = [self._normalize_call(c) for c in predicted]
        if len(exp_norm) != len(pred_norm):
            return False
        for exp, pred in zip(exp_norm, pred_norm):
            if exp.get("name") != pred.get("name"):
                return False
            exp_args = exp.get("args", {})
            pred_args = pred.get("args", {})
            if set(exp_args.keys()) != set(pred_args.keys()):
                return False
            for key in exp_args.keys():
                if not self._values_equal(exp_args[key], pred_args[key]):
                    return False
        return True

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance]
    ) -> Dict[str, Any]:
        instance = instance_output.data_instance
        expected = instance.expected_calls or []
        predicted = self._extract_pred_calls(instance_output)
        return {
            "predicted_calls": predicted,
            "expected_calls": expected,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[WorkbenchInstance]
    ) -> Dict[str, Union[int, float]]:
        instance = instance_output.data_instance
        expected = instance.expected_calls or []
        predicted = self._extract_pred_calls(instance_output)
        correct = self._calls_match(expected, predicted)
        return {
            "correct": int(correct),
            "num_expected": len(expected),
            "num_predicted": len(predicted),
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        num_instances = len(eval_outputs)
        if num_instances == 0:
            return {"success_rate": 0.0, "num_instances": 0}
        return {
            "success_rate": sum(
                e.get("correct", 0) for e in eval_outputs if isinstance(e, dict)
            )
            / num_instances,
            "num_instances": num_instances,
        }
