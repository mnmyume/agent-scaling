import copy
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, cast

import requests

import langfuse
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_litellm.chat_models.litellm import ChatLiteLLM
from litellm.types.utils import ModelResponse

logger = logging.getLogger(__name__)


class _HTTPResponseShim:
    def __init__(self, status_code: int, headers: Mapping[str, Any]) -> None:
        self.status_code = status_code
        self.headers = headers


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

    @staticmethod
    def _uses_anthropic_compatible_proxy(api_base: Optional[str]) -> bool:
        if not api_base:
            return False
        lowered = api_base.lower().rstrip("/")
        return "anthropic.com" not in lowered and "/anthropic" in lowered

    def _is_anthropic_proxy_request(self, **kwargs: Any) -> bool:
        model = cast(str, kwargs.get("model", self.model_name or self.model))
        api_base = cast(Optional[str], kwargs.get("api_base", self.api_base))
        return model.startswith("anthropic/") and self._uses_anthropic_compatible_proxy(
            api_base
        )

    @staticmethod
    def _coerce_tool_input(raw_arguments: Any) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return {"raw_arguments": raw_arguments}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return {}

    @classmethod
    def _coerce_anthropic_content_blocks(cls, raw_content: Any) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        if raw_content is None:
            return blocks
        if isinstance(raw_content, str):
            text = raw_content.strip()
            if text:
                blocks.append({"type": "text", "text": raw_content})
            return blocks
        if isinstance(raw_content, dict):
            if raw_content.get("type") is not None:
                blocks.append(raw_content)
                return blocks
            text = raw_content.get("text")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            return blocks
        if isinstance(raw_content, list):
            for item in raw_content:
                blocks.extend(cls._coerce_anthropic_content_blocks(item))
        return blocks

    @staticmethod
    def _normalize_anthropic_stop_reason(
        stop_reason: Optional[str],
        finish_reason: Optional[str],
        has_tool_calls: bool,
    ) -> str:
        if stop_reason:
            return stop_reason
        if finish_reason == "tool_calls":
            return "tool_use"
        if finish_reason == "length":
            return "max_tokens"
        if finish_reason:
            return finish_reason
        return "tool_use" if has_tool_calls else "end_turn"

    @staticmethod
    def _normalize_anthropic_usage(usage: Any) -> Dict[str, Any]:
        if not isinstance(usage, dict):
            return {"input_tokens": 0, "output_tokens": 0}
        if "input_tokens" in usage or "output_tokens" in usage:
            return usage
        return {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "server_tool_use": usage.get("server_tool_use"),
        }

    @classmethod
    def _normalize_anthropic_compatible_completion(
        cls, completion_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        normalized = dict(completion_response)
        normalized_content = cls._coerce_anthropic_content_blocks(
            normalized.get("content")
        )

        message = normalized.get("message")
        if isinstance(message, dict) and not normalized_content:
            normalized_content.extend(
                cls._coerce_anthropic_content_blocks(message.get("content"))
            )
            if not normalized_content:
                normalized_content.extend(
                    cls._coerce_anthropic_content_blocks(message.get("text"))
                )

        choices = normalized.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else None
        choice_message = (
            first_choice.get("message")
            if isinstance(first_choice, dict) and isinstance(first_choice.get("message"), dict)
            else None
        )
        if choice_message and not normalized_content:
            normalized_content.extend(
                cls._coerce_anthropic_content_blocks(choice_message.get("content"))
            )
            if not normalized_content:
                normalized_content.extend(
                    cls._coerce_anthropic_content_blocks(choice_message.get("text"))
                )
        if isinstance(first_choice, dict) and not normalized_content:
            normalized_content.extend(
                cls._coerce_anthropic_content_blocks(first_choice.get("text"))
            )
        if not normalized_content:
            normalized_content.extend(
                cls._coerce_anthropic_content_blocks(normalized.get("output_text"))
            )
        if not normalized_content:
            normalized_content.extend(
                cls._coerce_anthropic_content_blocks(normalized.get("text"))
            )

        tool_calls = normalized.get("tool_calls")
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        if tool_calls is None and isinstance(choice_message, dict):
            tool_calls = choice_message.get("tool_calls")

        for idx, tool_call in enumerate(tool_calls or []):
            function = tool_call.get("function") or {}
            normalized_content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_proxy_{idx}",
                    "name": function.get("name")
                    or tool_call.get("name")
                    or f"tool_{idx}",
                    "input": cls._coerce_tool_input(
                        function.get("arguments", tool_call.get("args"))
                    ),
                }
            )

        normalized["content"] = normalized_content
        normalized["usage"] = cls._normalize_anthropic_usage(normalized.get("usage"))
        normalized["stop_reason"] = cls._normalize_anthropic_stop_reason(
            cast(Optional[str], normalized.get("stop_reason")),
            cast(
                Optional[str],
                first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
            ),
            bool(tool_calls),
        )
        return normalized

    def _should_use_anthropic_proxy_fallback(
        self, exc: BaseException, **kwargs: Any
    ) -> bool:
        if kwargs.get("stream"):
            return False
        if not self._is_anthropic_proxy_request(**kwargs):
            return False

        formatted_tb = traceback.format_exc()
        error_text = str(exc)
        has_known_parse_signature = (
            "'NoneType' object is not iterable" in error_text
            or "'NoneType' object is not iterable" in formatted_tb
        ) and (
            "extract_response_content" in error_text
            or "transform_parsed_response" in error_text
            or "extract_response_content" in formatted_tb
            or "transform_parsed_response" in formatted_tb
        )

        if has_known_parse_signature:
            return True
        if isinstance(exc, TypeError):
            return True
        return isinstance(exc, (KeyError, AttributeError)) and (
            "extract_response_content" in formatted_tb
            or "transform_parsed_response" in formatted_tb
        )

    def _anthropic_proxy_completion_fallback(self, **kwargs: Any) -> ModelResponse:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        from litellm.utils import ModelResponse as LiteLLMModelResponse

        model = cast(str, kwargs.get("model", self.model_name or self.model))
        api_base = cast(Optional[str], kwargs.get("api_base", self.api_base))
        api_key = cast(
            Optional[str],
            kwargs.get("api_key") or self.api_key or self.anthropic_api_key,
        )
        if api_base is None or api_key is None:
            raise ValueError(
                "Anthropic-compatible fallback requires both api_base and api_key."
            )

        endpoint = api_base.rstrip("/")
        if not endpoint.endswith("/v1/messages"):
            endpoint = f"{endpoint}/v1/messages"

        config = AnthropicConfig()
        request_messages = copy.deepcopy(cast(List[Dict[str, Any]], kwargs["messages"]))
        prefix_messages = copy.deepcopy(request_messages)
        non_default_params = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "messages",
                "model",
                "api_base",
                "api_key",
                "force_timeout",
                "custom_llm_provider",
                "run_manager",
                "extra_headers",
            }
            and value is not None
        }
        optional_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )
        headers = config.validate_environment(
            api_key=api_key,
            headers=dict(cast(Dict[str, Any], kwargs.get("extra_headers") or {})),
            model=model,
            messages=copy.deepcopy(request_messages),
            optional_params=copy.deepcopy(optional_params),
            litellm_params={},
        )
        request_payload = config.transform_request(
            model=model,
            messages=request_messages,
            optional_params=optional_params,
            litellm_params={},
            headers=headers,
        )

        logger.warning(
            "LiteLLM Anthropic parsing failed for %s via %s; retrying with a "
            "direct anthropic-compatible fallback.",
            model,
            api_base,
        )

        timeout = kwargs.get("force_timeout")
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(request_payload),
            timeout=timeout,
        )
        response.raise_for_status()

        raw_completion = response.json()
        normalized_completion = self._normalize_anthropic_compatible_completion(
            raw_completion
        )
        normalized_completion.setdefault("model", model)
        model_response = LiteLLMModelResponse(model=model)
        parsed_response = config.transform_parsed_response(
            completion_response=normalized_completion,
            raw_response=_HTTPResponseShim(
                status_code=response.status_code,
                headers=dict(response.headers),
            ),
            model_response=model_response,
            json_mode=optional_params.get("json_mode"),
            prefix_prompt=config.get_prefix_prompt(messages=prefix_messages),
        )
        parsed_response.model = model
        parsed_response._hidden_params["anthropic_proxy_fallback"] = True
        parsed_response._hidden_params["raw_provider_response"] = raw_completion
        return parsed_response

    @staticmethod
    def _resolve_retry_budget(request_retries: Any, default_retries: int) -> int:
        if request_retries is None:
            return max(default_retries, 0)
        try:
            resolved = int(request_retries)
        except (TypeError, ValueError):
            return max(default_retries, 0)
        return max(resolved, 0)

    @staticmethod
    def _extract_status_code(exc: BaseException) -> Optional[int]:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        status_code = getattr(exc, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    @classmethod
    def _is_retryable_server_error(cls, exc: BaseException) -> bool:
        status_code = cls._extract_status_code(exc)
        if status_code is not None and status_code >= 500:
            return True

        try:
            import litellm
        except Exception:
            litellm = None

        if litellm is None:
            return False

        service_unavailable_error = getattr(
            litellm, "ServiceUnavailableError", None
        )
        retryable_types = tuple(
            error_type
            for error_type in (
                getattr(litellm, "InternalServerError", None),
                service_unavailable_error,
            )
            if isinstance(error_type, type)
        )
        return isinstance(exc, retryable_types)

    @classmethod
    def _retry_delay_seconds(cls, exc: BaseException, retry_index: int) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0.0), 30.0)
            except (TypeError, ValueError):
                pass
        return min(float(2**retry_index), 8.0)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        res: ChatResult = super()._create_chat_result(response)
        if res.llm_output is None:
            res.llm_output = {}
        res.llm_output["litellm_response"] = response
        return res

    def invoke(self, *args, **kwargs) -> AIMessage:
        return cast(AIMessage, super().invoke(*args, **kwargs))

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        max_server_error_retries = self._resolve_retry_budget(
            kwargs.get("num_retries"), self.max_retries
        )
        retry_count = 0
        model = cast(str, kwargs.get("model", self.model_name or self.model))

        while True:
            try:
                return super().completion_with_retry(
                    run_manager=run_manager, **kwargs
                )
            except Exception as exc:
                active_exc: BaseException = exc
                if self._should_use_anthropic_proxy_fallback(exc, **kwargs):
                    try:
                        return self._anthropic_proxy_completion_fallback(**kwargs)
                    except Exception as fallback_exc:
                        active_exc = fallback_exc

                if (
                    not self._is_retryable_server_error(active_exc)
                    or retry_count >= max_server_error_retries
                ):
                    if active_exc is exc:
                        raise
                    raise active_exc

                retry_count += 1
                delay_seconds = self._retry_delay_seconds(active_exc, retry_count)
                logger.warning(
                    "Retrying %s after retryable server error (%s/%s): %s",
                    model,
                    retry_count,
                    max_server_error_retries,
                    active_exc,
                )
                time.sleep(delay_seconds)

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
