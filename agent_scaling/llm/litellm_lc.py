from time import perf_counter
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
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @classmethod
    def _attach_response_latency_ms(cls, response: Any, latency_ms: float) -> None:
        if latency_ms <= 0:
            return

        # Keep provider-populated timings if present; only fill missing/zero values.
        if isinstance(response, dict):
            current = cls._safe_float(response.get("_response_ms"))
            if current is None or current <= 0:
                response["_response_ms"] = latency_ms
            hidden = response.get("_hidden_params")
            if isinstance(hidden, dict):
                hidden_current = cls._safe_float(hidden.get("_response_ms"))
                if hidden_current is None or hidden_current <= 0:
                    hidden["_response_ms"] = latency_ms
                hidden_ms = cls._safe_float(hidden.get("response_ms"))
                if hidden_ms is None or hidden_ms <= 0:
                    hidden["response_ms"] = latency_ms
            return

        current = cls._safe_float(getattr(response, "_response_ms", None))
        if current is None or current <= 0:
            try:
                setattr(response, "_response_ms", latency_ms)
            except Exception:
                pass

        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            hidden_current = cls._safe_float(hidden.get("_response_ms"))
            if hidden_current is None or hidden_current <= 0:
                hidden["_response_ms"] = latency_ms
            hidden_ms = cls._safe_float(hidden.get("response_ms"))
            if hidden_ms is None or hidden_ms <= 0:
                hidden["response_ms"] = latency_ms

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

        start = perf_counter()
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        self._attach_response_latency_ms(response, elapsed_ms)

        if log_langfuse:
            self._log_langfuse(message_dicts, params, response)
        return self._create_chat_result(response)
