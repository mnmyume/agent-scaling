import json
import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance

DATASET_IDS = ["finance-agent", "finance_agent", "financeagent"]


@register_dataset_instance(DATASET_IDS)
class FinanceAgentInstance(DatasetInstance):
    question: str
    answer: Optional[str] = None
    rubric: Optional[str] = None
    context: Optional[str] = None
    task_id: Optional[str] = None

    def model_post_init(self, __context):
        if self.answer is not None:
            self.expected_output = self.answer

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "question": self.question,
            "context": self.context or "",
            "rubric": self.rubric or "",
        }


@register_dataset(DATASET_IDS)
class FinanceAgentDataset(Dataset):
    dataset_id: str = "finance-agent"
    instances: List[FinanceAgentInstance]
    _required_eval_prompts: List[str] = ["grader"]
    _require_llm_eval: bool = True

    def get_instance_eval_output(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[FinanceAgentInstance],
    ) -> Dict[str, Any]:
        instance = instance_output.data_instance
        return {
            "question": instance.question,
            "response": instance_output.agent_output,
            "target": instance.expected_output or "",
            "rubric": instance.rubric or "",
        }

    def _parse_grader_response(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        # Fallback: look for simple correctness hints
        lowered = text.lower()
        if "correct" in lowered and "incorrect" not in lowered:
            return {"correct": True, "score": 1.0}
        if "incorrect" in lowered:
            return {"correct": False, "score": 0.0}
        return {"correct": False, "score": 0.0}

    def get_instance_eval_metrics(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[FinanceAgentInstance],
    ) -> Dict[str, Union[int, float, str]]:
        assert self.eval_prompts is not None, "eval_llm must be set for evaluation"
        assert self.eval_llm is not None, "eval_llm must be set for evaluation"
        instance = instance_output.data_instance
        prompt_message = self.eval_prompts["grader"].compile(
            question=instance.question,
            response=instance_output.agent_output,
            rubric=instance.rubric or "",
            target=instance.expected_output or "",
        )
        response: BaseMessage = self.eval_llm.invoke(prompt_message)
        grader_text = response.text().strip()
        parsed = self._parse_grader_response(grader_text)
        correct = bool(parsed.get("correct", False))
        score = parsed.get("score", 0.0)
        explanation = parsed.get("explanation", "")
        return {
            "correct": int(correct),
            "score": float(score) if score is not None else 0.0,
            "explanation": str(explanation),
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        num_instances = len(eval_outputs)
        if num_instances == 0:
            return {"accuracy": 0.0, "avg_score": 0.0, "num_instances": 0}
        return {
            "accuracy": sum(
                e.get("correct", 0) for e in eval_outputs if isinstance(e, dict)
            )
            / num_instances,
            "avg_score": sum(
                e.get("score", 0.0) for e in eval_outputs if isinstance(e, dict)
            )
            / num_instances,
            "num_instances": num_instances,
        }
