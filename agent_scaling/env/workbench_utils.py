from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

# WorkBench includes many relative-time requests ("today", "tomorrow", "past fortnight").
# The upstream benchmark anchors those requests to a fixed reference timestamp so the
# sandbox state and gold actions stay deterministic across runs.
WORKBENCH_REFERENCE_TIME = pd.Timestamp("2023-11-30T00:00:00")
WORKBENCH_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "workbench"
STATEFUL_DATAFRAME_NAMES = (
    "calendar_events",
    "emails",
    "plots_data",
    "project_tasks",
    "crm_data",
)
CASE_SENSITIVE_COMPARE_COLUMNS = {"status", "list_name", "board"}

CANONICAL_TO_INTERNAL_TOOL_NAMES = {
    "calendar.get_event_information_by_id": "calendar_get_event_information_by_id",
    "calendar.search_events": "calendar_search_events",
    "calendar.create_event": "calendar_create_event",
    "calendar.delete_event": "calendar_delete_event",
    "calendar.update_event": "calendar_update_event",
    "email.get_email_information_by_id": "email_get_email_information_by_id",
    "email.search_emails": "email_search_emails",
    "email.send_email": "email_send_email",
    "email.delete_email": "email_delete_email",
    "email.forward_email": "email_forward_email",
    "email.reply_email": "email_reply_email",
    "analytics.get_visitor_information_by_id": "analytics_get_visitor_information_by_id",
    "analytics.create_plot": "analytics_create_plot",
    "analytics.total_visits_count": "analytics_total_visits_count",
    "analytics.engaged_users_count": "analytics_engaged_users_count",
    "analytics.traffic_source_count": "analytics_traffic_source_count",
    "analytics.get_average_session_duration": "analytics_get_average_session_duration",
    "project_management.get_task_information_by_id": "project_management_get_task_information_by_id",
    "project_management.search_tasks": "project_management_search_tasks",
    "project_management.create_task": "project_management_create_task",
    "project_management.delete_task": "project_management_delete_task",
    "project_management.update_task": "project_management_update_task",
    "customer_relationship_manager.search_customers": "customer_relationship_manager_search_customers",
    "customer_relationship_manager.update_customer": "customer_relationship_manager_update_customer",
    "customer_relationship_manager.add_customer": "customer_relationship_manager_add_customer",
    "customer_relationship_manager.delete_customer": "customer_relationship_manager_delete_customer",
    "company_directory.find_email_address": "company_directory_find_email_address",
}
INTERNAL_TO_CANONICAL_TOOL_NAMES = {
    internal: canonical
    for canonical, internal in CANONICAL_TO_INTERNAL_TOOL_NAMES.items()
}
SIDE_EFFECT_TOOL_NAMES = {
    "calendar_create_event",
    "calendar_delete_event",
    "calendar_update_event",
    "email_send_email",
    "email_delete_email",
    "email_forward_email",
    "email_reply_email",
    "analytics_create_plot",
    "project_management_create_task",
    "project_management_delete_task",
    "project_management_update_task",
    "customer_relationship_manager_update_customer",
    "customer_relationship_manager_add_customer",
    "customer_relationship_manager_delete_customer",
}


def _read_workbench_csv(filename: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(WORKBENCH_DATA_DIR / filename, **kwargs)


def normalize_workbench_tool_name(tool_name: str) -> str:
    normalized = tool_name.strip()
    if normalized.endswith(".func"):
        normalized = normalized[:-5]
    if normalized in CANONICAL_TO_INTERNAL_TOOL_NAMES:
        return CANONICAL_TO_INTERNAL_TOOL_NAMES[normalized]
    if normalized in INTERNAL_TO_CANONICAL_TOOL_NAMES:
        return normalized
    raise ValueError(f"Unknown WorkBench tool name: {tool_name}")


def canonical_workbench_tool_name(tool_name: str) -> str:
    internal_name = normalize_workbench_tool_name(tool_name)
    return INTERNAL_TO_CANONICAL_TOOL_NAMES[internal_name]


def _serialize_workbench_value(value: Any) -> str:
    return json.dumps("" if value is None else str(value))


@dataclass(frozen=True)
class WorkbenchAction:
    tool_name: str
    arguments: Dict[str, Any]

    @property
    def canonical_tool_name(self) -> str:
        return canonical_workbench_tool_name(self.tool_name)

    def to_canonical_call(self) -> str:
        args = ", ".join(
            f"{key}={_serialize_workbench_value(value)}"
            for key, value in self.arguments.items()
        )
        return f"{self.canonical_tool_name}.func({args})"

    def has_side_effect(self) -> bool:
        return normalize_workbench_tool_name(self.tool_name) in SIDE_EFFECT_TOOL_NAMES


def parse_workbench_gold_action(action: str) -> WorkbenchAction:
    match = re.fullmatch(r"([a-z_]+\.[a-z_]+)\.func\((.*)\)", action.strip(), re.DOTALL)
    if match is None:
        raise ValueError(f"Invalid WorkBench action: {action}")
    tool_name = normalize_workbench_tool_name(match.group(1))
    args_src = match.group(2).strip()
    if not args_src:
        return WorkbenchAction(tool_name=tool_name, arguments={})

    parsed = ast.parse(f"f({args_src})", mode="eval")
    assert isinstance(parsed.body, ast.Call)
    arguments: Dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is None:
            raise ValueError(f"Unsupported positional argument in action: {action}")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return WorkbenchAction(tool_name=tool_name, arguments=arguments)


def normalize_state_frame_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in normalized.columns:
        if column in CASE_SENSITIVE_COMPARE_COLUMNS:
            continue
        if normalized[column].dtype == "O" or pd.api.types.is_string_dtype(normalized[column]):
            normalized[column] = normalized[column].str.lower()
    return normalized


class WorkbenchSandbox:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calendar_events = _read_workbench_csv("calendar_events.csv", dtype=str)
        self.emails = _read_workbench_csv("emails.csv", dtype=str)
        self.analytics_data = _read_workbench_csv("analytics_data.csv", dtype=str)
        self.analytics_data["user_engaged"] = (
            self.analytics_data["user_engaged"] == "True"
        )
        self.plots_data = pd.DataFrame(columns=["file_path"])
        self.project_tasks = _read_workbench_csv("project_tasks.csv", dtype=str)
        self.crm_data = _read_workbench_csv(
            "customer_relationship_manager_data.csv", dtype=str
        )
        self.company_directory = _read_workbench_csv(
            "email_addresses.csv",
            header=None,
            names=["email_address"],
            dtype=str,
        )

    def snapshot_state(self) -> Dict[str, pd.DataFrame]:
        return {
            "calendar_events": self.calendar_events.copy(),
            "emails": self.emails.copy(),
            "plots_data": self.plots_data.copy(),
            "project_tasks": self.project_tasks.copy(),
            "crm_data": self.crm_data.copy(),
        }

    def execute_action(self, action: WorkbenchAction) -> Any:
        tool_name = normalize_workbench_tool_name(action.tool_name)
        tool = getattr(self, tool_name, None)
        if tool is None:
            raise ValueError(f"Unsupported WorkBench action: {tool_name}")
        return tool(**action.arguments)

    def execute_actions(self, actions: Iterable[WorkbenchAction]) -> None:
        for action in actions:
            try:
                self.execute_action(action)
            except Exception:
                continue

    def calendar_get_event_information_by_id(
        self, event_id: str | None = None, field: str | None = None
    ) -> Any:
        if not event_id:
            return "Event ID not provided."
        if not field:
            return "Field not provided."
        event = self.calendar_events[
            self.calendar_events["event_id"] == event_id
        ].to_dict(orient="records")
        if not event:
            return "Event not found."
        if field not in event[0]:
            return "Field not found."
        return {field: event[0][field]}

    def calendar_search_events(
        self,
        query: str = "",
        time_min: str | None = None,
        time_max: str | None = None,
    ) -> Any:
        events = self.calendar_events[
            self.calendar_events["event_name"].str.contains(query, case=False, na=False)
            | self.calendar_events["participant_email"].str.contains(
                query, case=False, na=False
            )
        ].to_dict(orient="records")
        if time_min:
            events = [
                event
                for event in events
                if pd.Timestamp(event["event_start"]) >= pd.Timestamp(time_min)
            ]
        if time_max:
            events = [
                event
                for event in events
                if pd.Timestamp(event["event_start"]) <= pd.Timestamp(time_max)
            ]
        return events[:5] if events else "No events found."

    def calendar_create_event(
        self,
        event_name: str | None = None,
        participant_email: str | None = None,
        event_start: str | None = None,
        duration: str | None = None,
    ) -> str:
        if not event_name:
            return "Event name not provided."
        if not participant_email:
            return "Participant email not provided."
        if not event_start:
            return "Event start not provided."
        if not duration:
            return "Event duration not provided."

        participant_email = participant_email.lower()
        event_id = str(int(self.calendar_events["event_id"].max()) + 1).zfill(8)
        new_event = pd.DataFrame(
            {
                "event_id": [event_id],
                "event_name": [event_name],
                "participant_email": [participant_email],
                "event_start": [event_start],
                "duration": [duration],
            }
        )
        self.calendar_events = pd.concat([self.calendar_events, new_event])
        return event_id

    def calendar_delete_event(self, event_id: str | None = None) -> str:
        if not event_id:
            return "Event ID not provided."
        if event_id not in self.calendar_events["event_id"].values:
            return "Event not found."
        self.calendar_events = self.calendar_events[
            self.calendar_events["event_id"] != event_id
        ]
        return "Event deleted successfully."

    def calendar_update_event(
        self,
        event_id: str | None = None,
        field: str | None = None,
        new_value: str | None = None,
    ) -> str:
        if not event_id or not field or not new_value:
            return "Event ID, field, or new value not provided."
        if event_id not in self.calendar_events["event_id"].values:
            return "Event not found."
        if field == "participant_email":
            new_value = new_value.lower()
        self.calendar_events.loc[
            self.calendar_events["event_id"] == event_id, field
        ] = new_value
        return "Event updated successfully."

    def email_get_email_information_by_id(
        self, email_id: str | None = None, field: str | None = None
    ) -> Any:
        if not email_id:
            return "Email ID not provided."
        if not field:
            return "Field not provided."
        email = self.emails[self.emails["email_id"] == email_id].to_dict(
            orient="records"
        )
        if not email:
            return "Email not found."
        if field not in email[0]:
            return "Field not found."
        return {field: email[0][field]}

    def email_search_emails(
        self,
        query: str = "",
        date_min: str | None = None,
        date_max: str | None = None,
    ) -> Any:
        query_words = query.lower().split()

        def filter_email_row(row: pd.Series) -> bool:
            combined_fields = (
                f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
            )
            return all(word in combined_fields for word in query_words)

        filtered_emails = self.emails.apply(filter_email_row, axis=1)
        emails = (
            self.emails[filtered_emails]
            .sort_values("sent_datetime", ascending=False)
            .to_dict(orient="records")
        )
        if date_min:
            emails = [
                email
                for email in emails
                if pd.Timestamp(email["sent_datetime"]).date()
                >= pd.Timestamp(date_min).date()
            ]
        if date_max:
            emails = [
                email
                for email in emails
                if pd.Timestamp(email["sent_datetime"]).date()
                <= pd.Timestamp(date_max).date()
            ]
        return emails[:5] if emails else "No emails found."

    def email_send_email(
        self,
        recipient: str | None = None,
        subject: str | None = None,
        body: str | None = None,
    ) -> str:
        if not recipient or not subject or not body:
            return "Recipient, subject, or body not provided."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."

        recipient = recipient.lower()
        email_id = str(int(self.emails["email_id"].max()) + 1)
        sent_datetime = WORKBENCH_REFERENCE_TIME.strftime("%Y-%m-%d %H:%M:%S")
        self.emails.loc[len(self.emails)] = [
            email_id,
            "outbox",
            recipient,
            subject,
            sent_datetime,
            body,
        ]
        return "Email sent successfully."

    def email_delete_email(self, email_id: str | None = None) -> str:
        if not email_id:
            return "Email ID not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        self.emails = self.emails[self.emails["email_id"] != email_id]
        return "Email deleted successfully."

    def email_forward_email(
        self, email_id: str | None = None, recipient: str | None = None
    ) -> str:
        if not email_id or not recipient:
            return "Email ID or recipient not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."

        recipient = recipient.lower()
        email = self.emails[self.emails["email_id"] == email_id].to_dict(
            orient="records"
        )[0]
        result = self.email_send_email(
            recipient=recipient,
            subject=f"FW: {email['subject']}",
            body=email["body"],
        )
        return "Email forwarded successfully." if result == "Email sent successfully." else result

    def email_reply_email(
        self, email_id: str | None = None, body: str | None = None
    ) -> str:
        if not email_id or not body:
            return "Email ID or body not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."

        email = self.emails[self.emails["email_id"] == email_id].to_dict(
            orient="records"
        )[0]
        result = self.email_send_email(
            recipient=email["sender/recipient"],
            subject=email["subject"],
            body=body,
        )
        return "Email replied successfully." if result == "Email sent successfully." else result

    def analytics_get_visitor_information_by_id(
        self, visitor_id: str | None = None
    ) -> Any:
        if not visitor_id:
            return "Visitor ID not provided."
        visitor_data = self.analytics_data[
            self.analytics_data["visitor_id"] == visitor_id
        ].to_dict(orient="records")
        return visitor_data if visitor_data else "Visitor not found."

    def analytics_create_plot(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        value_to_plot: str | None = None,
        plot_type: str | None = None,
    ) -> str:
        if not time_min:
            return "Start date not provided."
        if not time_max:
            return "End date not provided."
        if value_to_plot not in {
            "total_visits",
            "session_duration_seconds",
            "user_engaged",
            "visits_direct",
            "visits_referral",
            "visits_search_engine",
            "visits_social_media",
        }:
            return (
                "Value to plot must be one of 'total_visits', 'session_duration_seconds', "
                "'user_engaged', 'direct', 'referral', 'search engine', 'social media'"
            )
        if plot_type not in {"bar", "line", "scatter", "histogram"}:
            return "Plot type must be one of 'bar', 'line', 'scatter', or 'histogram'"

        file_path = f"plots/{time_min}_{time_max}_{value_to_plot}_{plot_type}.png"
        self.plots_data.loc[len(self.plots_data)] = [file_path]
        return file_path

    def analytics_total_visits_count(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> Dict[str, Any]:
        data = self.analytics_data
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        return data.groupby("date_of_visit").size().to_dict()

    def analytics_engaged_users_count(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> Dict[str, Any]:
        data = self.analytics_data.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["user_engaged"] = data["user_engaged"].astype(bool).astype(int)
        return data.groupby("date_of_visit").sum()["user_engaged"].to_dict()

    def analytics_traffic_source_count(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        traffic_source: str | None = None,
    ) -> Dict[str, Any]:
        data = self.analytics_data.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        if traffic_source:
            data["visits_from_source"] = (
                data["traffic_source"] == traffic_source
            ).astype(int)
            return data.groupby("date_of_visit").sum()["visits_from_source"].to_dict()
        return data.groupby("date_of_visit").size().to_dict()

    def analytics_get_average_session_duration(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> Dict[str, Any]:
        data = self.analytics_data.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["session_duration_seconds"] = data["session_duration_seconds"].astype(float)
        return (
            data[["date_of_visit", "session_duration_seconds"]]
            .groupby("date_of_visit")
            .mean()["session_duration_seconds"]
            .to_dict()
        )

    def project_management_get_task_information_by_id(
        self, task_id: str | None = None, field: str | None = None
    ) -> Any:
        if not task_id:
            return "Task ID not provided."
        if not field:
            return "Field not provided."
        task = self.project_tasks[self.project_tasks["task_id"] == task_id].to_dict(
            orient="records"
        )
        if not task:
            return "Task not found."
        if field not in task[0]:
            return "Field not found."
        return {field: task[0][field]}

    def project_management_search_tasks(
        self,
        task_name: str | None = None,
        assigned_to_email: str | None = None,
        list_name: str | None = None,
        due_date: str | None = None,
        board: str | None = None,
    ) -> Any:
        if not any([task_name, assigned_to_email, list_name, due_date, board]):
            return "No search parameters provided."
        tasks = self.project_tasks.copy()
        if task_name:
            tasks = tasks[tasks["task_name"].str.contains(task_name, case=False, na=False)]
        if assigned_to_email:
            tasks = tasks[
                tasks["assigned_to_email"].str.contains(
                    assigned_to_email, case=False, na=False
                )
            ]
        if list_name:
            tasks = tasks[tasks["list_name"].str.contains(list_name, case=False, na=False)]
        if due_date:
            tasks = tasks[tasks["due_date"].str.contains(due_date, case=False, na=False)]
        if board:
            tasks = tasks[tasks["board"].str.contains(board, case=False, na=False)]
        return tasks.to_dict(orient="records")

    def project_management_create_task(
        self,
        task_name: str | None = None,
        assigned_to_email: str | None = None,
        list_name: str | None = None,
        due_date: str | None = None,
        board: str | None = None,
    ) -> str:
        if not all([task_name, assigned_to_email, list_name, due_date, board]):
            return "Missing task details."

        assigned_to_email = assigned_to_email.lower()
        if assigned_to_email not in self.project_tasks["assigned_to_email"].str.lower().values:
            return "Assignee email not valid. Please choose from the list of team members."
        if list_name not in {"Backlog", "In Progress", "In Review", "Completed"}:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if board not in {"Back end", "Front end", "Design"}:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."

        task_id = str(int(self.project_tasks["task_id"].max()) + 1).zfill(8)
        new_task = pd.DataFrame(
            {
                "task_id": [task_id],
                "task_name": [task_name],
                "assigned_to_email": [assigned_to_email],
                "list_name": [list_name],
                "due_date": [due_date],
                "board": [board],
            }
        )
        self.project_tasks = pd.concat(
            [self.project_tasks, new_task], ignore_index=True
        )
        return task_id

    def project_management_delete_task(self, task_id: str | None = None) -> str:
        if not task_id:
            return "Task ID not provided."
        if task_id not in self.project_tasks["task_id"].values:
            return "Task not found."
        self.project_tasks = self.project_tasks[
            self.project_tasks["task_id"] != task_id
        ]
        return "Task deleted successfully."

    def project_management_update_task(
        self,
        task_id: str | None = None,
        field: str | None = None,
        new_value: str | None = None,
    ) -> str:
        if not task_id or not field or not new_value:
            return "Task ID, field, or new value not provided."
        if field == "assigned_to_email":
            new_value = new_value.lower()
        if field == "board" and new_value not in {"Back end", "Front end", "Design"}:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
        if field == "list_name" and new_value not in {
            "Backlog",
            "In Progress",
            "In Review",
            "Completed",
        }:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if (
            field == "assigned_to_email"
            and new_value not in self.project_tasks["assigned_to_email"].str.lower().values
        ):
            return "Assignee email not valid. Please choose from the list of team members."
        if task_id not in self.project_tasks["task_id"].values:
            return "Task not found."
        if field not in self.project_tasks.columns:
            return "Field not valid."
        self.project_tasks.loc[self.project_tasks["task_id"] == task_id, field] = new_value
        return "Task updated successfully."

    def customer_relationship_manager_search_customers(
        self,
        customer_name: str | None = None,
        customer_email: str | None = None,
        product_interest: str | None = None,
        status: str | None = None,
        assigned_to_email: str | None = None,
        last_contact_date_min: str | None = None,
        last_contact_date_max: str | None = None,
        follow_up_by_min: str | None = None,
        follow_up_by_max: str | None = None,
    ) -> Any:
        if not any(
            [
                customer_name,
                customer_email,
                product_interest,
                status,
                assigned_to_email,
                last_contact_date_min,
                last_contact_date_max,
                follow_up_by_min,
                follow_up_by_max,
            ]
        ):
            return "No search parameters provided. Please provide at least one parameter."
        customers = self.crm_data.copy()
        if customer_name:
            customers = customers[
                customers["customer_name"].str.contains(customer_name, case=False, na=False)
            ]
        if customer_email:
            customers = customers[
                customers["customer_email"].str.contains(customer_email, case=False, na=False)
            ]
        if product_interest:
            customers = customers[
                customers["product_interest"].str.contains(
                    product_interest, case=False, na=False
                )
            ]
        if status:
            customers = customers[customers["status"].str.contains(status, case=False, na=False)]
        if assigned_to_email:
            customers = customers[
                customers["assigned_to_email"].str.contains(
                    assigned_to_email, case=False, na=False
                )
            ]
        if last_contact_date_min:
            customers = customers[customers["last_contact_date"] >= last_contact_date_min]
        if last_contact_date_max:
            customers = customers[customers["last_contact_date"] <= last_contact_date_max]
        if follow_up_by_min:
            customers = customers[customers["follow_up_by"] >= follow_up_by_min]
        if follow_up_by_max:
            customers = customers[customers["follow_up_by"] <= follow_up_by_max]
        return customers.to_dict(orient="records")[:5]

    def customer_relationship_manager_update_customer(
        self,
        customer_id: str | None = None,
        field: str | None = None,
        new_value: str | None = None,
    ) -> str:
        if not customer_id or not field or not new_value:
            return "Customer ID, field, or new value not provided."
        if field == "status" and new_value not in {
            "Qualified",
            "Won",
            "Lost",
            "Lead",
            "Proposal",
        }:
            return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"
        if field == "product_interest" and new_value not in {
            "Software",
            "Hardware",
            "Services",
            "Consulting",
            "Training",
        }:
            return "Product interest not valid. Please choose from: 'Software', 'Hardware', 'Services', 'Consulting', 'Training'"
        if field in {"customer_email", "assigned_to_email"}:
            new_value = new_value.lower()
        if customer_id not in self.crm_data["customer_id"].values:
            return "Customer not found."
        if field not in self.crm_data.columns:
            return (
                "Field not valid. Please choose from: 'customer_name', 'assigned_to_email', "
                "'customer_email', 'customer_phone', 'last_contact_date', 'product_interest', "
                "'status', 'notes', 'follow_up_by'"
            )
        self.crm_data.loc[self.crm_data["customer_id"] == customer_id, field] = new_value
        return "Customer updated successfully."

    def customer_relationship_manager_add_customer(
        self,
        customer_name: str | None = None,
        assigned_to_email: str | None = None,
        status: str | None = None,
        customer_email: str | None = None,
        customer_phone: str | None = None,
        last_contact_date: str | None = None,
        product_interest: str | None = None,
        notes: str = "",
        follow_up_by: str | None = None,
    ) -> str:
        if not all([customer_name, assigned_to_email, status]):
            return "Please provide all required fields: customer_name, assigned_to_email, status."

        assigned_to_email = assigned_to_email.lower()
        if customer_email:
            customer_email = customer_email.lower()
        customer_id = str(int(self.crm_data["customer_id"].max()) + 1).zfill(8)
        new_customer = pd.DataFrame(
            {
                "customer_id": [customer_id],
                "customer_name": [customer_name],
                "customer_email": [customer_email],
                "customer_phone": [customer_phone],
                "last_contact_date": [last_contact_date],
                "product_interest": [product_interest],
                "status": [status],
                "assigned_to_email": [assigned_to_email],
                "notes": [notes],
                "follow_up_by": [follow_up_by],
            }
        )
        self.crm_data = pd.concat([self.crm_data, new_customer], ignore_index=True)
        return customer_id

    def customer_relationship_manager_delete_customer(
        self, customer_id: str | None = None
    ) -> str:
        if not customer_id:
            return "Customer ID not provided."
        if customer_id not in self.crm_data["customer_id"].values:
            return "Customer not found."
        self.crm_data = self.crm_data[self.crm_data["customer_id"] != customer_id]
        return "Customer deleted successfully."

    def company_directory_find_email_address(self, name: str = "") -> Any:
        if name == "":
            return "Name not provided."
        email_matches = self.company_directory[
            self.company_directory["email_address"].str.contains(name.lower(), na=False)
        ]
        return email_matches["email_address"].tolist()


def execute_workbench_actions(
    actions: Iterable[WorkbenchAction],
) -> Dict[str, pd.DataFrame]:
    sandbox = WorkbenchSandbox()
    sandbox.execute_actions(actions)
    return sandbox.snapshot_state()


def workbench_states_equal(
    left_state: Dict[str, pd.DataFrame], right_state: Dict[str, pd.DataFrame]
) -> bool:
    return all(
        normalize_state_frame_for_comparison(left_state[name]).equals(
            normalize_state_frame_for_comparison(right_state[name])
        )
        for name in STATEFUL_DATAFRAME_NAMES
    )
