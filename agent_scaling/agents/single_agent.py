import os
import os.path as osp
import traceback
from typing import List, Optional, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,  # type: ignore
    ToolMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages  # type: ignore

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.agents.multiagent_utils.metrics_collector import MetricsCollector
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import (
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
    TrajectoryStep,
)
from agent_scaling.env import AgentEnvironment
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml
from agent_scaling.utils.token_budget import (
    TokenBudgetManager,
    extract_token_usage,
)

from .registry import register_agent


@register_agent("single-agent")
class SingleAgent(AgentSystemWithTools[AgentEnvironment]):
    """
    A single agent that can interact with tools. Developed in SWE-Agent framework.
    """

    required_prompts = ["main"]

    def __init__(self, max_steps: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
        budget_manager: Optional[TokenBudgetManager] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        llm_params_dict = llm_params.model_dump() if llm_params else {}
        instance_identifier = getattr(instance, "index", None)
        metrics_collector = MetricsCollector(
            architecture="single-agent",
            num_agents=1,
            dataset_id=self.dataset.dataset_id,
            instance_idx=instance_idx,
            instance_id=str(instance_identifier) if instance_identifier is not None else None,
            model_name=getattr(self.llm, "model", None),
            token_budget=budget_manager.total_budget if budget_manager else None,
        )
        self.metrics_collector = metrics_collector
        env, llm_w_tools = self.init_environment(instance)
        shared_prompt_templates = self.get_dataset_prompt_templates(env)

        messages = cast(
            list,
            self.prompts["main"].compile(**shared_prompt_templates),
        )
        trajectory: List[TrajectoryStep] = []
        final_answer = ""
        final_env_output = {}
        is_done = False
        completion_reason = "max_steps_reached"
        for step in range(self.max_steps):
            env.set_metrics_context(round_num=1, iteration=step + 1)
            response = self._invoke_with_metrics(
                llm_w_tools,
                messages,  # type: ignore[arg-type]
                agent_id="single_agent",
                llm_kwargs=llm_params_dict,
                call_type="agent_step",
                round_num=1,
                iteration=step + 1,
            )
            response = cast(AIMessage, response)

            if budget_manager is not None:
                inp_tok, out_tok = extract_token_usage(response)
                budget_manager.consume(inp_tok, out_tok)

            if response.tool_calls:
                response.tool_calls = [response.tool_calls[0]]

            messages.append(convert_to_openai_messages(response))
            tool_resp: ToolMessage | None = None
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = ""
                try:
                    tool_resp = env.execute_tool(tool_call)
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    action = f"{tool_name}({', '.join([f'{k}={v}' for k, v in tool_input.items()])})"
                    messages.append(convert_to_openai_messages(tool_resp))
                    is_done = tool_name == "done"
                    if is_done:
                        completion_reason = "done_tool"
                except Exception as e:
                    action = ""
                    messages.append(
                        {
                            "role": "user",
                            "content": f"ERROR: Tool **{tool_name}** failed with error: {str(e)}. Please check the tool call.",
                        }
                    )
                    logger.warning(
                        f"Tool **{tool_name}** failed with error: {str(e)}\n{traceback.format_exc()}"
                    )
            else:
                action = ""
                messages.append(
                    {
                        "role": "user",
                        "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                    }
                )
                logger.warning("No tool calls found in the response.")
            trajectory.append(
                TrajectoryStep(
                    action=action,
                    observation=str(tool_resp.content) if tool_resp else "",
                    response=str(response.content),
                    thought=str(response.content),
                )
            )
            if is_done or env.env_done():
                final_answer = trajectory[-1].observation
                if not is_done:
                    completion_reason = "env_done"
                break
        final_env_output = env.env_status()
        metrics_collector.set_env_success(final_env_output.success)
        metrics_collector.set_completion_reason(completion_reason)
        if final_answer:
            metrics_collector.log_agent_output(
                agent_id="single_agent",
                output_type="final_answer",
                content=final_answer,
                round=1,
                iteration=len(trajectory) if trajectory else None,
            )

        if budget_manager is not None:
            budget_manager.log_status()

        runtime_metrics = metrics_collector.export_metrics()
        runtime_metrics_path = None
        runtime_events_path = None
        if instance_dir is not None:
            os.makedirs(instance_dir, exist_ok=True)
            out = {
                "trajectory": [t.model_dump() for t in trajectory],
                "final_answer": final_answer,
            }
            write_yaml(
                out,
                osp.join(instance_dir, "agent_output.yaml"),
                use_long_str_representer=True,
            )
            runtime_metrics = metrics_collector.write_artifacts(instance_dir)
            runtime_metrics_path = osp.join(instance_dir, "runtime_metrics.json")
            runtime_events_path = osp.join(instance_dir, "runtime_events.jsonl")
        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=trajectory,
            final_env_output=final_env_output,
            budget_used=budget_manager.used if budget_manager else 0,
            budget_remaining=budget_manager.remaining if budget_manager else 0,
            budget_exceeded=budget_manager.budget_exceeded if budget_manager else False,
            runtime_metrics=runtime_metrics,
            runtime_metrics_path=runtime_metrics_path,
            runtime_events_path=runtime_events_path,
        )
