import csv
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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
    answer: str
    rubric: Optional[str] = None
    type: Optional[str] = None
    expert_time_to_completion: Optional[str] = None
    task_id: Optional[str] = None

    def model_post_init(self, __context):
        self.expected_output = self.answer

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "question": self.question,
        }


@register_dataset(DATASET_IDS)
class FinanceAgentDataset(Dataset):
    dataset_id: str = "finance-agent"
    instances: List[FinanceAgentInstance]
    _required_eval_prompts: List[str] = ["grader"]
    _require_llm_eval: bool = True

    @classmethod
    def from_csv(cls, csv_path: str, **kwargs) -> "FinanceAgentDataset":
        instances: List[FinanceAgentInstance] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                rubric = row.get("rubric")
                q_type = row.get("type")
                expert_time = row.get("expert time to completion") or row.get(
                    "expert_time_to_completion"
                )

                if not question or not answer:
                    continue
                instances.append(
                    FinanceAgentInstance(
                        question=question,
                        answer=answer,
                        rubric=rubric,
                        type=q_type,
                        expert_time_to_completion=expert_time,
                    )
                )
        return cls(instances=instances, **kwargs)

    def _extract_model_answer(self, text: Any) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            return str(text)
        s = text.strip()
        # Our done tool returns: "<answer>\t<confidence>"
        if "\t" in s:
            s = s.split("\t", 1)[0].strip()
        match = re.search(r"FINAL\\s+ANSWER\\s*:\\s*(.*)", s, re.IGNORECASE | re.DOTALL)
        if match:
            s = match.group(1).strip()
        # Drop trailing SOURCES blocks if present.
        s = re.split(r"\\n\\s*SOURCES\\s*:", s, flags=re.IGNORECASE)[0].strip()
        return s

    def _split_rubric(self, rubric: str) -> Tuple[List[str], List[str]]:
        text = rubric.replace("\r\n", "\n").replace("\r", "\n")
        m_corr = re.search(r"(?i)\\bcorrectness\\s*:", text)
        m_contra = re.search(r"(?i)\\bcontradiction\\s*:", text)
        corr_block = ""
        contra_block = ""
        if m_corr and m_contra:
            if m_corr.start() < m_contra.start():
                corr_block = text[m_corr.end() : m_contra.start()]
                contra_block = text[m_contra.end() :]
            else:
                contra_block = text[m_contra.end() : m_corr.start()]
                corr_block = text[m_corr.end() :]
        elif m_corr:
            corr_block = text[m_corr.end() :]
        elif m_contra:
            contra_block = text[m_contra.end() :]
        else:
            corr_block = text

        def parse_block(block: str) -> List[str]:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            criteria: List[str] = []
            current: List[str] = []
            for line in lines:
                if re.match(r"^(\\d+\\.|[-*])\\s+", line):
                    if current:
                        criteria.append(" ".join(current).strip())
                    current = [re.sub(r"^(\\d+\\.|[-*])\\s+", "", line).strip()]
                else:
                    if current:
                        current.append(line)
            if current:
                criteria.append(" ".join(current).strip())
            return [c for c in criteria if c]

        return parse_block(corr_block), parse_block(contra_block)

    def get_instance_eval_output(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[FinanceAgentInstance],
    ) -> Dict[str, Any]:
        instance = instance_output.data_instance
        extracted = self._extract_model_answer(instance_output.agent_output)
        return {
            "question": instance.question,
            "response_raw": instance_output.agent_output,
            "response": extracted,
            "target": instance.expected_output or "",
            "rubric": instance.rubric or "",
            "type": instance.type,
        }

    def _parse_grader_response(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        # Fallback: look for simple satisfaction hints
        lowered = text.lower()
        if "satisfied" in lowered and "not satisfied" not in lowered:
            return {"satisfied": True}
        if "yes" in lowered and "no" not in lowered:
            return {"satisfied": True}
        if "not satisfied" in lowered:
            return {"satisfied": False}
        if "no" in lowered and "yes" not in lowered:
            return {"satisfied": False}
        return {"satisfied": False}

    def get_instance_eval_metrics(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[FinanceAgentInstance],
    ) -> Dict[str, Union[int, float, str]]:
        assert self.eval_prompts is not None, "eval_llm must be set for evaluation"
        assert self.eval_llm is not None, "eval_llm must be set for evaluation"
        instance = instance_output.data_instance
        model_answer = self._extract_model_answer(instance_output.agent_output)
        rubric = instance.rubric or ""
        correctness, contradiction = self._split_rubric(rubric) if rubric else ([], [])

        total = len(correctness) + len(contradiction)
        if total == 0:
            return {
                "correct": 0,
                "rubric_score": 0.0,
                "n_criteria": 0,
                "question_type": instance.type or "",
                "explanation": "No rubric provided; cannot evaluate.",
            }

        def check(section: str, criterion: str) -> Tuple[bool, str]:
            prompt_message = self.eval_prompts["grader"].compile(
                question=instance.question,
                response=model_answer,
                rubric_section=section,
                criterion=criterion,
            )
            response_msg: BaseMessage = self.eval_llm.invoke(prompt_message)
            parsed = self._parse_grader_response(response_msg.text().strip())
            satisfied = bool(parsed.get("satisfied", False))
            explanation = str(parsed.get("explanation", "") or "")
            return satisfied, explanation

        satisfied_count = 0
        failed: List[str] = []
        for criterion in correctness:
            ok, _ = check("Correctness", criterion)
            if ok:
                satisfied_count += 1
            else:
                failed.append(f"Correctness: {criterion}")
        for criterion in contradiction:
            ok, _ = check("Contradiction", criterion)
            if ok:
                satisfied_count += 1
            else:
                failed.append(f"Contradiction: {criterion}")

        correct = int(len(failed) == 0)
        rubric_score = satisfied_count / total
        return {
            "correct": correct,
            "rubric_score": float(rubric_score),
            "n_criteria": total,
            "question_type": instance.type or "",
            "explanation": "; ".join(failed) if failed else "",
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any] | str]) -> Dict[str, Any]:
        num_instances = len(eval_outputs)
        if num_instances == 0:
            return {"accuracy": 0.0, "avg_rubric_score": 0.0, "num_instances": 0}

        correct_vals = [e.get("correct", 0) for e in eval_outputs if isinstance(e, dict)]
        rubric_scores = [
            e.get("rubric_score", 0.0) for e in eval_outputs if isinstance(e, dict)
        ]

        by_type: Dict[str, List[int]] = {}
        for e in eval_outputs:
            if not isinstance(e, dict):
                continue
            t = str(e.get("question_type", "") or "unknown")
            by_type.setdefault(t, []).append(int(e.get("correct", 0)))

        accuracy = sum(correct_vals) / num_instances
        balanced_accuracy = None
        if by_type:
            balanced_accuracy = sum(
                (sum(v) / len(v)) for v in by_type.values() if v
            ) / len(by_type)

        out: Dict[str, Any] = {
            "accuracy": accuracy,
            "avg_rubric_score": sum(rubric_scores) / num_instances,
            "num_instances": num_instances,
        }
        if balanced_accuracy is not None:
            out["balanced_accuracy"] = balanced_accuracy
            out["accuracy_by_type"] = {k: (sum(v) / len(v)) for k, v in by_type.items()}
        return out
