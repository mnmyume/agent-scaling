import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

from agent_scaling.datasets.workbench import WorkbenchInstance
from agent_scaling.env.base import AgentEnvironment
from agent_scaling.env.registry import register_env


@register_env("workbench")
class WorkbenchEnvironment(AgentEnvironment):
    """Workbench environment with dynamic tools defined per instance."""

    def __init__(self, *args, **kwargs):
        self.is_done = False
        self.executed_calls: List[Dict[str, Any]] = []
        dataset_instance = kwargs.get("dataset_instance")
        additional_tools: Dict[str, StructuredTool] = {}
        if isinstance(dataset_instance, WorkbenchInstance):
            additional_tools.update(self._build_dynamic_tools(dataset_instance))
        super().__init__(*args, additional_env_tools=additional_tools, **kwargs)

    def _build_dynamic_tools(
        self, instance: WorkbenchInstance
    ) -> Dict[str, StructuredTool]:
        tools: Dict[str, StructuredTool] = {}
        for tool_spec in instance.tools or []:
            name = tool_spec.get("name") or tool_spec.get("tool_name")
            if not name:
                continue
            description = tool_spec.get("description", "")
            params = tool_spec.get("parameters") or tool_spec.get("args_schema") or {}
            args_schema = self._build_args_schema(name, params)
            tool_fn = self._make_tool_fn(name)
            tools[name] = StructuredTool.from_function(
                func=tool_fn,
                name=name,
                description=description,
                args_schema=args_schema,
            )
        return tools

    def _build_args_schema(self, name: str, params: Dict[str, Any]) -> Optional[type[BaseModel]]:
        if not params:
            return None
        properties = params.get("properties", {}) if isinstance(params, dict) else {}
        required = set(params.get("required", [])) if isinstance(params, dict) else set()
        fields: Dict[str, Any] = {}
        for prop, spec in properties.items():
            spec_type = spec.get("type") if isinstance(spec, dict) else None
            if spec_type == "integer":
                field_type = int
            elif spec_type == "number":
                field_type = float
            elif spec_type == "boolean":
                field_type = bool
            elif spec_type == "array":
                field_type = list
            elif spec_type == "object":
                field_type = dict
            else:
                field_type = str
            default = ... if prop in required else None
            fields[prop] = (field_type, default)
        if not fields:
            return None
        return create_model(f"{name}_Args", **fields)  # type: ignore[misc]

    def _make_tool_fn(self, tool_name: str):
        def _tool(**kwargs):
            return json.dumps({"tool": tool_name, "args": kwargs})

        return _tool

    def env_done(self) -> bool:
        return self.is_done

    def execute_tool(self, tool_call):  # type: ignore[override]
        self.executed_calls.append(
            {"name": tool_call.get("name"), "args": tool_call.get("args", {})}
        )
        if tool_call.get("name") == "done":
            self.is_done = True
        return super().execute_tool(tool_call)
