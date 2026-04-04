import inspect
import json
import time
from typing import Dict, List, Optional

from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from agent_scaling.config.prompts import Prompt
from agent_scaling.datasets.base import Dataset, DatasetEnvStatus, DatasetInstance
from agent_scaling.env.tools import enhance_tool, get_tool, list_registered_tools


class AgentEnvironment:
    required_prompts: List[str] = []
    env_tools: Dict[str, BaseTool] = {}

    def __init__(
        self,
        env_prompts: Dict[str, Prompt] | None = None,
        additional_env_tools: Dict[str, BaseTool] | None = None,
        tools: List[str] | None = None,
        log_langfuse: bool = False,
        dataset: Dataset | None = None,
        dataset_instance: DatasetInstance | None = None,
        agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.log_langfuse = log_langfuse
        self.env_prompts: Dict[str, Prompt] | None = env_prompts
        self.check_required_prompts(self.env_prompts or {})
        self._tools: Dict[str, BaseTool] = {}
        env_tools = self.env_tools or {
            tool_name: get_tool(tool_name) for tool_name in list_registered_tools()
        }
        env_tools.update(additional_env_tools or {})
        tools = tools or list(env_tools.keys())
        self._tools = {
            tool_name: enhance_tool(
                env_tools[tool_name], use_langfuse=self.log_langfuse
            )
            for tool_name in tools
        }
        self.dataset = dataset
        self.dataset_instance = dataset_instance
        self.success = False
        self.num_steps = 0
        self.metrics_collector = None
        self._metrics_round: Optional[int] = None
        self._metrics_iteration: Optional[int] = None

    def env_status(self) -> DatasetEnvStatus:
        """
        Get the final output of the environment.
        """
        return DatasetEnvStatus(success=self.success, num_steps=self.num_steps)

    def get_instance_prompt_info(self) -> Dict[str, str]:
        """
        Get the prompt info for the instance.
        """
        return {
            **(
                self.dataset_instance.get_prompt_info() if self.dataset_instance else {}
            ),
            "tools_description": self.tools_description,
        }

    def env_done(self) -> bool:
        """
        Check if the environment is done.
        """
        return False

    @classmethod
    def check_required_prompts(cls, prompts: Dict[str, Prompt]) -> None:
        required_but_not_found = []
        for prompt_name in cls.required_prompts:
            if prompt_name not in prompts:
                required_but_not_found.append(prompt_name)
        if required_but_not_found:
            raise ValueError(
                f"{cls.__name__} Environment Prompt Template(s) **{', '.join(required_but_not_found)}** are required but not found."
            )

    @property
    def tools(self) -> Dict[str, BaseTool]:
        """
        A dictionary of tools that the agent can use.
        """
        return self._tools

    @property
    def tools_list(self) -> List[BaseTool]:
        """
        A list of tools that the agent can use.
        """
        return list(self._tools.values())

    @property
    def tools_description(self) -> str:
        ret = ""
        for tool_name, tool in self.tools.items():
            tool_call = f"- {tool.name}: {tool.description}"
            try:
                assert tool.args_schema is not None
                if isinstance(tool.args_schema, dict):
                    tool_args = tool.args_schema
                else:
                    tool_args = tool.args_schema.model_json_schema()
                    tool_args = {
                        "args": tool_args.get("properties", {}),
                        "required": tool_args.get("required", []),
                    }
                if tool_args:
                    tool_call += f"with arguments:\n{json.dumps(tool_args, indent=2)}"
            except Exception:
                pass
            ret += f"{tool_call}\n"
        return ret

    def attach_metrics_collector(self, metrics_collector: object) -> None:
        self.metrics_collector = metrics_collector

    def set_metrics_context(
        self,
        round_num: Optional[int] = None,
        iteration: Optional[int] = None,
    ) -> None:
        self._metrics_round = round_num
        self._metrics_iteration = iteration

    def execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        """
        Execute a single tool.
        """
        start_time = time.perf_counter()
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        try:
            ret = self.tools[tool_name].invoke(tool_call)
        except Exception as exc:
            if self.metrics_collector is not None:
                self.metrics_collector.log_tool_call(
                    agent_id=self.agent_id or "single_agent",
                    tool_name=tool_name,
                    arguments=tool_args,
                    success=False,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    round=self._metrics_round,
                    iteration=self._metrics_iteration,
                    error_message=str(exc),
                )
            raise

        self.num_steps += 1
        if self.metrics_collector is not None:
            self.metrics_collector.log_tool_call(
                agent_id=self.agent_id or "single_agent",
                tool_name=tool_name,
                arguments=tool_args,
                success=True,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                round=self._metrics_round,
                iteration=self._metrics_iteration,
                result=ret.content,
            )
        return ret

    def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolMessage]:
        """
        Execute multiple tools.
        """
        tool_msgs = []
        for tool_call in tool_calls:
            tool_msg = self.execute_tool(tool_call)
            tool_msgs.append(tool_msg)
        return tool_msgs


class AgentEnvironmentTools(AgentEnvironment):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.env_tools: Dict[str, BaseTool] = {
            name: tool
            for name, tool in inspect.getmembers(
                self, predicate=lambda x: isinstance(x, BaseTool)
            )
        }
        super().__init__(*args, **kwargs)
