from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, StructuredTool

from agent_scaling.env.base import AgentEnvironment
from agent_scaling.env.registry import register_env
from agent_scaling.logger import logger
from agent_scaling.utils import get_root_dir
from agent_scaling.workbench.sandbox import (
    HARDCODED_CURRENT_TIME,
    WorkbenchSandbox,
    evaluate_actions,
    format_func_call,
)


def _missing_data_hint(data_dir: str) -> str:
    return (
        f"WorkBench data dir not found: {data_dir}. "
        "Expected datasets/workbench/* to exist. "
        "You can populate it from upstream WorkBench by running: "
        "python run_scripts/fetch_workbench.py"
    )


def _build_workbench_tools(sandbox: WorkbenchSandbox) -> Dict[str, BaseTool]:
    """Create WorkBench tool set with names matching the upstream benchmark."""
    # NOTE: Tool names must match the dataset ground-truth actions (minus `.func`).
    tool_defs: List[Tuple[str, Any, str]] = [
        # Calendar
        (
            "calendar.get_event_information_by_id",
            sandbox.calendar_get_event_information_by_id,
            "Get a single calendar event field by event_id.",
        ),
        (
            "calendar.search_events",
            sandbox.calendar_search_events,
            "Search calendar events by query (event_name/participant_email) and optional time range.",
        ),
        (
            "calendar.create_event",
            sandbox.calendar_create_event,
            "Create a calendar event (side effect).",
        ),
        (
            "calendar.delete_event",
            sandbox.calendar_delete_event,
            "Delete a calendar event by event_id (side effect).",
        ),
        (
            "calendar.update_event",
            sandbox.calendar_update_event,
            "Update a calendar event field by event_id (side effect).",
        ),
        # Email
        (
            "email.get_email_information_by_id",
            sandbox.email_get_email_information_by_id,
            "Get a single email field by email_id.",
        ),
        (
            "email.search_emails",
            sandbox.email_search_emails,
            "Search emails by keyword query and optional date range.",
        ),
        (
            "email.send_email",
            sandbox.email_send_email,
            "Send an email (side effect).",
        ),
        (
            "email.delete_email",
            sandbox.email_delete_email,
            "Delete an email by email_id (side effect).",
        ),
        (
            "email.forward_email",
            sandbox.email_forward_email,
            "Forward an email to a recipient (side effect).",
        ),
        (
            "email.reply_email",
            sandbox.email_reply_email,
            "Reply to an email (side effect).",
        ),
        # Analytics
        (
            "analytics.get_visitor_information_by_id",
            sandbox.analytics_get_visitor_information_by_id,
            "Get analytics record(s) by visitor_id.",
        ),
        (
            "analytics.create_plot",
            sandbox.analytics_create_plot,
            "Create a plot record (side effect).",
        ),
        (
            "analytics.total_visits_count",
            sandbox.analytics_total_visits_count,
            "Count total visits per day over a date range.",
        ),
        (
            "analytics.engaged_users_count",
            sandbox.analytics_engaged_users_count,
            "Count engaged users per day over a date range.",
        ),
        (
            "analytics.traffic_source_count",
            sandbox.analytics_traffic_source_count,
            "Count visits per day filtered by traffic source over a date range.",
        ),
        (
            "analytics.get_average_session_duration",
            sandbox.analytics_get_average_session_duration,
            "Average session duration per day over a date range.",
        ),
        # Project management
        (
            "project_management.get_task_information_by_id",
            sandbox.project_management_get_task_information_by_id,
            "Get a single project task field by task_id.",
        ),
        (
            "project_management.search_tasks",
            sandbox.project_management_search_tasks,
            "Search project tasks by one or more fields.",
        ),
        (
            "project_management.create_task",
            sandbox.project_management_create_task,
            "Create a project task (side effect).",
        ),
        (
            "project_management.delete_task",
            sandbox.project_management_delete_task,
            "Delete a project task by task_id (side effect).",
        ),
        (
            "project_management.update_task",
            sandbox.project_management_update_task,
            "Update a project task field by task_id (side effect).",
        ),
        # CRM
        (
            "customer_relationship_manager.search_customers",
            sandbox.customer_relationship_manager_search_customers,
            "Search customer records with optional filters.",
        ),
        (
            "customer_relationship_manager.update_customer",
            sandbox.customer_relationship_manager_update_customer,
            "Update a customer record field (side effect).",
        ),
        (
            "customer_relationship_manager.add_customer",
            sandbox.customer_relationship_manager_add_customer,
            "Add a customer record (side effect).",
        ),
        (
            "customer_relationship_manager.delete_customer",
            sandbox.customer_relationship_manager_delete_customer,
            "Delete a customer record by customer_id (side effect).",
        ),
        # Company directory
        (
            "company_directory.find_email_address",
            sandbox.company_directory_find_email_address,
            "Find email address(es) for an employee by name substring.",
        ),
    ]

    tools: Dict[str, BaseTool] = {}
    for name, fn, desc in tool_defs:
        tools[name] = StructuredTool.from_function(
            func=fn,
            name=name,
            description=desc,
        )
    return tools


@register_env("workbench")
class WorkbenchEnvironment(AgentEnvironment):
    """WorkBench environment (local sandbox + deterministic evaluation)."""

    def __init__(self, *args, **kwargs):
        self.is_done = False
        self._predicted_actions: List[str] = []
        self._ground_truth_actions: List[str] = []

        repo_root = get_root_dir()
        self.data_dir = os.path.join(repo_root, "datasets", "workbench")
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(_missing_data_hint(self.data_dir))

        self.sandbox = WorkbenchSandbox(self.data_dir)
        additional_env_tools = _build_workbench_tools(self.sandbox)

        dataset_instance = kwargs.get("dataset_instance")
        expected = getattr(dataset_instance, "expected_output", None)
        if isinstance(expected, list):
            self._ground_truth_actions = [str(a) for a in expected]

        requested_tools = kwargs.get("tools", None)
        # In this repo, an empty list currently means "no tools". For WorkBench, we
        # treat it as "use the full WorkBench tool set + done".
        if not requested_tools:
            requested_tools = list(additional_env_tools.keys()) + ["done"]
        elif "done" not in requested_tools:
            requested_tools = list(requested_tools) + ["done"]
        kwargs["tools"] = requested_tools

        super().__init__(*args, additional_env_tools=additional_env_tools, **kwargs)

    def get_instance_prompt_info(self) -> Dict[str, str]:
        base = super().get_instance_prompt_info()
        base["current_weekday"] = HARDCODED_CURRENT_TIME.strftime("%A")
        base["current_date"] = str(HARDCODED_CURRENT_TIME.date())
        base["current_time"] = str(HARDCODED_CURRENT_TIME.time())
        base["meeting_constraints"] = (
            "Meetings must not start before 09:00 or end after 18:00."
        )
        return base

    def env_done(self) -> bool:
        return self.is_done

    def execute_tool(self, tool_call):  # type: ignore[override]
        name = tool_call.get("name")
        args = tool_call.get("args", {}) or {}

        # Record tool calls in WorkBench's canonical '<tool>.func(...)' form.
        if name and name != "done":
            self._predicted_actions.append(
                format_func_call(name, dict(args), arg_order=list(args.keys()))
            )

        tool_msg = super().execute_tool(tool_call)

        if name == "done":
            self.is_done = True
            try:
                eval_result = evaluate_actions(
                    predicted_actions=self._predicted_actions,
                    ground_truth_actions=self._ground_truth_actions,
                    sandbox_factory=lambda: WorkbenchSandbox(self.data_dir),
                )
                self.success = bool(eval_result.get("correct", 0))
            except Exception as exc:
                logger.warning("Workbench evaluation failed: %s", exc)
                self.success = False

        return tool_msg

