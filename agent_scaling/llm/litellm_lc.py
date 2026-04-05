import json
from typing import Any, Dict, List, Mapping, Optional, cast

import langfuse
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_litellm.chat_models.litellm import ChatLiteLLM
from litellm.types.utils import ModelResponse


class ChatLiteLLMLC(ChatLiteLLM):
    log_langfuse: bool = False

    @staticmethod
    def _stringify_message_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=True, sort_keys=True)
        except TypeError:
            return str(content)

    @classmethod
    def _sanitize_tool_history_for_text_only_completion(
        cls, message_dicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rewrite tool-call history into plain text for text-only follow-up turns."""

        tool_calls_by_id: Dict[str, Dict[str, str]] = {}
        sanitized_messages: List[Dict[str, Any]] = []

        for message in message_dicts:
            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                content_parts: List[str] = []
                text_content = cls._stringify_message_content(
                    message.get("content")
                ).strip()
                if text_content:
                    content_parts.append(text_content)

                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function") or {}
                    tool_name = (
                        function.get("name")
                        or tool_call.get("name")
                        or "unknown_tool"
                    )
                    tool_args = cls._stringify_message_content(
                        function.get("arguments", tool_call.get("args"))
                    ).strip()
                    tool_call_id = tool_call.get("id")
                    if tool_call_id:
                        tool_calls_by_id[tool_call_id] = {
                            "name": tool_name,
                            "arguments": tool_args,
                        }

                    if tool_args:
                        content_parts.append(
                            f"Tool call issued: {tool_name} with arguments {tool_args}"
                        )
                    else:
                        content_parts.append(f"Tool call issued: {tool_name}")

                sanitized_messages.append(
                    {
                        "role": "assistant",
                        "content": "\n".join(content_parts) or "Tool call issued.",
                    }
                )
                continue

            if role == "assistant" and message.get("function_call"):
                function = message["function_call"]
                tool_name = function.get("name") or "unknown_function"
                tool_args = cls._stringify_message_content(
                    function.get("arguments")
                ).strip()
                content_parts: List[str] = []
                text_content = cls._stringify_message_content(
                    message.get("content")
                ).strip()
                if text_content:
                    content_parts.append(text_content)
                if tool_args:
                    content_parts.append(
                        f"Function call issued: {tool_name} with arguments {tool_args}"
                    )
                else:
                    content_parts.append(f"Function call issued: {tool_name}")
                sanitized_messages.append(
                    {
                        "role": "assistant",
                        "content": "\n".join(content_parts),
                    }
                )
                continue

            if role in {"tool", "function"}:
                tool_call_id = message.get("tool_call_id")
                matched_call = tool_calls_by_id.get(tool_call_id, {})
                tool_name = (
                    message.get("name")
                    or matched_call.get("name")
                    or "unknown_tool"
                )
                tool_args = matched_call.get("arguments", "")
                content = cls._stringify_message_content(message.get("content")).strip()
                prefix = f"Tool result from {tool_name}"
                if tool_args:
                    prefix += f" with arguments {tool_args}"
                sanitized_messages.append(
                    {
                        "role": "user",
                        "content": f"{prefix}:\n{content}" if content else prefix,
                    }
                )
                continue

            sanitized_messages.append(dict(message))

        return sanitized_messages

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        res: ChatResult = super()._create_chat_result(response)
        if res.llm_output is None:
            res.llm_output = {}
        res.llm_output["litellm_response"] = response
        return res

    def invoke(self, *args, **kwargs) -> AIMessage:
        return cast(AIMessage, super().invoke(*args, **kwargs))

    def _log_langfuse(
        self,
        message_dicts: List[Dict[str, Any]],
        params: Dict[str, Any],
        response: ModelResponse,
    ) -> None:
        client = langfuse.Langfuse()  # type: ignore
        model_params = {
            k: v
            for k, v in params.items()
            if k not in ["model", "stream"] and v is not None
        }

        gen_context = client.start_generation(
            name=f"call {params.get('model')}"
            + (" (from cache)" if response._hidden_params.get("cache_hit", "") else ""),
            input=message_dicts,
            model=params.get("model"),
            model_parameters=model_params,
        )

        gen_context.update(
            output=response.choices[0].message,  # type: ignore
            metadata=response.model_dump(),
            usage_details=response.usage if hasattr(response, "usage") else None,  # type: ignore
            cost_details={"total": response._hidden_params.get("response_cost", 0)},
        )
        gen_context.end()
        client.flush()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        log_langfuse = kwargs.pop("log_langfuse", None)
        if log_langfuse is None:
            log_langfuse = self.log_langfuse

        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        if "tools" not in params and any(
            message.get("role") in {"tool", "function"}
            or (
                message.get("role") == "assistant"
                and (message.get("tool_calls") or message.get("function_call"))
            )
            for message in message_dicts
        ):
            message_dicts = self._sanitize_tool_history_for_text_only_completion(
                message_dicts
            )

        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )

        if log_langfuse:
            self._log_langfuse(message_dicts, params, response)
        return self._create_chat_result(response)
