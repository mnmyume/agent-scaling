from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent_scaling.config.prompts import Prompt
from agent_scaling.logger import logger

from .litellm_lc import ChatLiteLLMLC


class GraderParseError(ValueError):
    """Raised when an LLM grader response cannot be parsed into the expected schema."""


def _strip_json_code_block(text: str) -> str:
    """Remove markdown code fence formatting from an LLM JSON response."""
    text = text.strip()
    code_block_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(code_block_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _message_text(response: Any) -> str:
    if hasattr(response, "text"):
        return str(response.text() or "")
    if hasattr(response, "content"):
        return str(response.content or "")
    return str(response or "")


def parse_json_grade_response(text: str) -> Dict[str, Any]:
    cleaned = _strip_json_code_block(text)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise GraderParseError("Expected a JSON object from the grader.")
    if "correct" not in payload:
        raise GraderParseError("JSON grader response is missing 'correct'.")
    payload["correct"] = int(bool(payload.get("correct", False)))
    payload["score"] = float(payload.get("score", float(payload["correct"])))
    payload["explanation"] = str(payload.get("explanation", "")).strip()
    return payload


def parse_browsecomp_grade_response(text: str) -> Dict[str, Any]:
    correct_match = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    if correct_match is None:
        raise GraderParseError("BrowseComp grader response is missing 'correct: yes/no'.")

    extracted_answer = "Not found"
    answer_match = re.search(
        r"extracted_final_answer:\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if answer_match is not None:
        candidate = answer_match.group(1).strip()
        if candidate and candidate.lower() != "none":
            extracted_answer = candidate

    confidence = 100.0
    confidence_match = re.search(
        r"confidence:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if confidence_match is not None:
        confidence = float(confidence_match.group(1))

    return {
        "is_correct": correct_match.group(1).lower() == "yes",
        "extracted_answer": extracted_answer,
        "confidence": confidence,
    }


def _json_grade_failure(raw_text: str, error: Exception, attempts: int) -> Dict[str, Any]:
    return {
        "correct": 0,
        "score": 0.0,
        "explanation": f"Failed to parse LLM grader response after {attempts} attempt(s): {raw_text}",
        "grader_attempts": attempts,
        "grader_parse_error": str(error),
        "grader_response": raw_text,
    }


def _browsecomp_grade_failure(
    raw_text: str, error: Exception, attempts: int
) -> Dict[str, Any]:
    return {
        "is_correct": False,
        "extracted_answer": "Not found",
        "confidence": 0.0,
        "grader_attempts": attempts,
        "grader_parse_error": str(error),
        "grader_response": raw_text,
    }


@dataclass
class LLMGrader:
    llm: Any
    prompt: Prompt
    parser: Callable[[str], Dict[str, Any]]
    parse_failure_factory: Callable[[str, Exception, int], Dict[str, Any]]
    max_retries: int = 1
    retry_instruction: str = "Return only the required schema with no extra text."
    invoke_kwargs: Optional[Dict[str, Any]] = None

    def _build_retry_messages(
        self,
        messages: List[Dict[str, Any]],
        raw_response: str,
        error: Exception,
    ) -> List[Dict[str, Any]]:
        return messages + [
            {"role": "assistant", "content": raw_response},
            {
                "role": "user",
                "content": (
                    f"Your previous response could not be parsed: {error}. "
                    f"{self.retry_instruction}"
                ),
            },
        ]

    def grade(
        self,
        *,
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        current_messages = messages or self.prompt.compile(**(prompt_kwargs or {}))
        last_text = ""
        last_error: Exception = GraderParseError("LLM grader returned empty response.")

        for attempt in range(1, self.max_retries + 2):
            response = self.llm.invoke(current_messages, **(self.invoke_kwargs or {}))
            last_text = _message_text(response).strip()
            if not last_text:
                last_error = GraderParseError("LLM grader returned empty response.")
            else:
                try:
                    result = self.parser(last_text)
                    result["grader_attempts"] = attempt
                    result["grader_response"] = last_text
                    return result
                except Exception as exc:
                    last_error = exc

            if attempt <= self.max_retries:
                logger.warning(f"Retrying grader parse after attempt {attempt}: {last_error}")
                current_messages = self._build_retry_messages(
                    current_messages,
                    last_text,
                    last_error,
                )

        logger.warning(f"Grader failed after {self.max_retries + 1} attempt(s): {last_error}")
        return self.parse_failure_factory(last_text, last_error, self.max_retries + 1)


def build_json_llm_grader(
    llm: Any,
    prompt: Prompt,
    *,
    max_retries: int = 1,
) -> LLMGrader:
    return LLMGrader(
        llm=llm,
        prompt=prompt,
        parser=parse_json_grade_response,
        parse_failure_factory=_json_grade_failure,
        max_retries=max_retries,
        retry_instruction=(
            'Return valid JSON with keys "correct", "score", and "explanation".'
        ),
    )


def build_browsecomp_grader(
    llm: Any,
    prompt: Prompt,
    *,
    max_retries: int = 1,
) -> LLMGrader:
    return LLMGrader(
        llm=llm,
        prompt=prompt,
        parser=parse_browsecomp_grade_response,
        parse_failure_factory=_browsecomp_grade_failure,
        max_retries=max_retries,
        retry_instruction=(
            "Return exactly the fields extracted_final_answer, reasoning, correct, and "
            "confidence in plain text."
        ),
    )


def llm_grade(
    model_output: str,
    reference: str,
    task_type: str = "qa",
    model_name: str = "gpt-4o",
) -> Dict[str, Any]:
    """Convenience JSON grader retained for existing call sites."""
    grader = build_json_llm_grader(
        llm=ChatLiteLLMLC(model=model_name, temperature=0),
        prompt=Prompt(name="grader", local_path="prompts/eval/grader.yaml"),
    )
    return grader.grade(
        prompt_kwargs={
            "model_output": model_output,
            "reference": reference,
            "task_type": task_type,
        }
    )
