from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# WorkBench hard-codes the current time for all tasks.
#
# Upstream: src/data_generation/data_generation_utils.py
HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T00:00:00")


# Tools with side effects (used for the "exact match" metric).
#
# Upstream: src/tools/toolkits.py
TOOLS_WITH_SIDE_EFFECTS = [
    "calendar.create_event",
    "calendar.delete_event",
    "calendar.update_event",
    "email.send_email",
    "email.delete_email",
    "email.forward_email",
    "email.reply_email",
    "analytics.create_plot",
    "project_management.create_task",
    "project_management.delete_task",
    "project_management.update_task",
    "customer_relationship_manager.update_customer",
    "customer_relationship_manager.add_customer",
    "customer_relationship_manager.delete_customer",
]


_FIELDS_NOT_TO_LOWER = {"status", "list_name", "board"}


def get_default_workbench_data_dir(repo_root: str) -> str:
    # Expected on-disk layout:
    # datasets/workbench/
    #   processed/*.csv
    #   raw/email_addresses.csv
    return os.path.join(repo_root, "datasets", "workbench")


def _safe_to_datetime(value: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def _convert_df_strs_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    # Mirrors WorkBench's evaluation: lowercase almost everything to be
    # case-insensitive, except for a few fields where they keep case.
    df = df.copy()
    for col in df.columns:
        if col in _FIELDS_NOT_TO_LOWER:
            continue
        # Upstream uses `df[col] = df[col].str.lower()`; keep NaNs as NaNs.
        if df[col].dtype == object:
            df[col] = df[col].str.lower()
    return df


@dataclass(frozen=True)
class WorkbenchState:
    calendar: pd.DataFrame
    email: pd.DataFrame
    analytics: pd.DataFrame
    project_management: pd.DataFrame
    customer_relationship_manager: pd.DataFrame


class WorkbenchSandbox:
    """In-memory sandbox used by WorkBench tools + evaluation.

    This is intentionally close to upstream behavior to keep metric parity.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        processed = os.path.join(data_dir, "processed")
        raw = os.path.join(data_dir, "raw")

        self._orig_emails = pd.read_csv(
            os.path.join(processed, "emails.csv"), dtype=str
        )
        self._orig_calendar_events = pd.read_csv(
            os.path.join(processed, "calendar_events.csv"), dtype=str
        )
        self._orig_analytics_data = pd.read_csv(
            os.path.join(processed, "analytics_data.csv"), dtype=str
        )
        # Upstream converts user_engaged to bool.
        self._orig_analytics_data["user_engaged"] = (
            self._orig_analytics_data["user_engaged"] == "True"
        )
        self._orig_project_tasks = pd.read_csv(
            os.path.join(processed, "project_tasks.csv"), dtype=str
        )
        self._orig_crm_data = pd.read_csv(
            os.path.join(processed, "customer_relationship_manager_data.csv"), dtype=str
        )
        self._email_addresses = pd.read_csv(
            os.path.join(raw, "email_addresses.csv"),
            header=None,
            names=["email_address"],
            dtype=str,
        )

        self.reset()

    def reset(self) -> None:
        self.emails = self._orig_emails.copy()
        self.calendar_events = self._orig_calendar_events.copy()
        self.analytics_data = self._orig_analytics_data.copy()
        self.plots_data = pd.DataFrame(columns=["file_path"])
        self.project_tasks = self._orig_project_tasks.copy()
        self.crm_data = self._orig_crm_data.copy()

    def snapshot(self) -> WorkbenchState:
        return WorkbenchState(
            calendar=self.calendar_events.copy(),
            email=self.emails.copy(),
            analytics=self.plots_data.copy(),
            project_management=self.project_tasks.copy(),
            customer_relationship_manager=self.crm_data.copy(),
        )

    # ---------------------------
    # Calendar tools
    # ---------------------------
    def calendar_get_event_information_by_id(
        self, event_id: Optional[str] = None, field: Optional[str] = None
    ) -> Any:
        if not event_id:
            return "Event ID not provided."
        if not field:
            return "Field not provided."
        event = self.calendar_events[self.calendar_events["event_id"] == event_id].to_dict(
            orient="records"
        )
        if not event:
            return "Event not found."
        if field not in event[0]:
            return "Field not found."
        return {field: event[0][field]}

    def calendar_search_events(
        self,
        query: str = "",
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
    ) -> Any:
        events = self.calendar_events[
            (self.calendar_events["event_name"].str.contains(query, case=False))
            | (self.calendar_events["participant_email"].str.contains(query, case=False))
        ].to_dict(orient="records")
        if time_min:
            tmin = _safe_to_datetime(time_min)
            if tmin is not None:
                events = [
                    e for e in events if _safe_to_datetime(e.get("event_start", "")) is not None and _safe_to_datetime(e["event_start"]) >= tmin
                ]
        if time_max:
            tmax = _safe_to_datetime(time_max)
            if tmax is not None:
                events = [
                    e for e in events if _safe_to_datetime(e.get("event_start", "")) is not None and _safe_to_datetime(e["event_start"]) <= tmax
                ]
        return events[:5] if events else "No events found."

    def calendar_create_event(
        self,
        event_name: Optional[str] = None,
        participant_email: Optional[str] = None,
        event_start: Optional[str] = None,
        duration: Optional[str] = None,
    ) -> Any:
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

    def calendar_delete_event(self, event_id: Optional[str] = None) -> Any:
        if not event_id:
            return "Event ID not provided."
        if event_id in self.calendar_events["event_id"].values:
            self.calendar_events = self.calendar_events[self.calendar_events["event_id"] != event_id]
            return "Event deleted successfully."
        return "Event not found."

    def calendar_update_event(
        self,
        event_id: Optional[str] = None,
        field: Optional[str] = None,
        new_value: Optional[str] = None,
    ) -> Any:
        if not event_id or not field or not new_value:
            return "Event ID, field, or new value not provided."
        if event_id not in self.calendar_events["event_id"].values:
            return "Event not found."
        if field == "participant_email":
            new_value = new_value.lower()
        self.calendar_events.loc[self.calendar_events["event_id"] == event_id, field] = new_value
        return "Event updated successfully."

    # ---------------------------
    # Email tools
    # ---------------------------
    def email_get_email_information_by_id(
        self, email_id: Optional[str] = None, field: Optional[str] = None
    ) -> Any:
        if not email_id:
            return "Email ID not provided."
        if not field:
            return "Field not provided."
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")
        if not email:
            return "Email not found."
        if field not in email[0]:
            return "Field not found."
        return {field: email[0][field]}

    def email_search_emails(
        self, query: str = "", date_min: Optional[str] = None, date_max: Optional[str] = None
    ) -> Any:
        query_words = query.lower().split()

        def matches(row: pd.Series) -> bool:
            combined = f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
            return all(w in combined for w in query_words)

        filtered = self.emails.apply(matches, axis=1)
        emails = (
            self.emails[filtered]
            .sort_values("sent_datetime", ascending=False)
            .to_dict(orient="records")
        )

        if date_min:
            dmin = _safe_to_datetime(date_min)
            if dmin is not None:
                emails = [
                    e
                    for e in emails
                    if _safe_to_datetime(e.get("sent_datetime", "")) is not None
                    and _safe_to_datetime(e["sent_datetime"]).date() >= dmin.date()
                ]
        if date_max:
            dmax = _safe_to_datetime(date_max)
            if dmax is not None:
                emails = [
                    e
                    for e in emails
                    if _safe_to_datetime(e.get("sent_datetime", "")) is not None
                    and _safe_to_datetime(e["sent_datetime"]).date() <= dmax.date()
                ]

        return emails[:5] if emails else "No emails found."

    def email_send_email(
        self,
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
    ) -> Any:
        if not recipient or not subject or not body:
            return "Recipient, subject, or body not provided."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."
        recipient = recipient.lower()

        email_id = str(int(self.emails["email_id"].max()) + 1)
        sent_datetime = HARDCODED_CURRENT_TIME
        # Keep upstream behavior: append via `.loc[len(df)]` (index may collide if there were deletions).
        self.emails.loc[len(self.emails)] = [
            email_id,
            "outbox",
            recipient,
            subject,
            sent_datetime,
            body,
        ]
        return "Email sent successfully."

    def email_delete_email(self, email_id: Optional[str] = None) -> Any:
        if not email_id:
            return "Email ID not provided."
        if email_id in self.emails["email_id"].values:
            self.emails = self.emails[self.emails["email_id"] != email_id]
            return "Email deleted successfully."
        return "Email not found."

    def email_forward_email(
        self, email_id: Optional[str] = None, recipient: Optional[str] = None
    ) -> Any:
        if not email_id or not recipient:
            return "Email ID or recipient not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."
        recipient = recipient.lower()
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")[0]
        result = self.email_send_email(recipient, f"FW: {email['subject']}", email["body"])
        return "Email forwarded successfully." if result == "Email sent successfully." else result

    def email_reply_email(self, email_id: Optional[str] = None, body: Optional[str] = None) -> Any:
        if not email_id or not body:
            return "Email ID or body not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")[0]
        result = self.email_send_email(email["sender/recipient"], f"{email['subject']}", body)
        return "Email replied successfully." if result == "Email sent successfully." else result

    # ---------------------------
    # Analytics tools
    # ---------------------------
    def analytics_get_visitor_information_by_id(self, visitor_id: Optional[str] = None) -> Any:
        if not visitor_id:
            return "Visitor ID not provided."
        visitor_data = self.analytics_data[self.analytics_data["visitor_id"] == visitor_id].to_dict(
            orient="records"
        )
        return visitor_data if visitor_data else "Visitor not found."

    def analytics_create_plot(
        self,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        value_to_plot: Optional[str] = None,
        plot_type: Optional[str] = None,
    ) -> Any:
        if not time_min:
            return "Start date not provided."
        if not time_max:
            return "End date not provided."
        if value_to_plot not in [
            "total_visits",
            "session_duration_seconds",
            "user_engaged",
            "visits_direct",
            "visits_referral",
            "visits_search_engine",
            "visits_social_media",
        ]:
            return (
                "Value to plot must be one of 'total_visits', 'session_duration_seconds', "
                "'user_engaged', 'direct', 'referral', 'search engine', 'social media'"
            )
        if plot_type not in ["bar", "line", "scatter", "histogram"]:
            return "Plot type must be one of 'bar', 'line', 'scatter', or 'histogram'"

        file_path = f"plots/{time_min}_{time_max}_{value_to_plot}_{plot_type}.png"
        self.plots_data.loc[len(self.plots_data)] = [file_path]
        return file_path

    def analytics_total_visits_count(
        self, time_min: Optional[str] = None, time_max: Optional[str] = None
    ) -> Any:
        if time_min:
            data = self.analytics_data[self.analytics_data["date_of_visit"] >= time_min]
        else:
            data = self.analytics_data
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        return data.groupby("date_of_visit").size().to_dict()

    def analytics_engaged_users_count(
        self, time_min: Optional[str] = None, time_max: Optional[str] = None
    ) -> Any:
        if time_min:
            data = self.analytics_data[self.analytics_data["date_of_visit"] >= time_min]
        else:
            data = self.analytics_data.copy()
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["user_engaged"] = data["user_engaged"].astype(bool).astype(int)
        return data.groupby("date_of_visit").sum()["user_engaged"].to_dict()

    def analytics_traffic_source_count(
        self,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        traffic_source: Optional[str] = None,
    ) -> Any:
        if time_min:
            data = self.analytics_data[self.analytics_data["date_of_visit"] >= time_min]
        else:
            data = self.analytics_data.copy()
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        if traffic_source:
            data["visits_from_source"] = (data["traffic_source"] == traffic_source).astype(int)
            return data.groupby("date_of_visit").sum()["visits_from_source"].to_dict()
        return data.groupby("date_of_visit").size().to_dict()

    def analytics_get_average_session_duration(
        self, time_min: Optional[str] = None, time_max: Optional[str] = None
    ) -> Any:
        if time_min:
            data = self.analytics_data[self.analytics_data["date_of_visit"] >= time_min]
        else:
            data = self.analytics_data
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data = data.copy()
        data["session_duration_seconds"] = data["session_duration_seconds"].astype(float)
        return (
            data[["date_of_visit", "session_duration_seconds"]]
            .groupby("date_of_visit")
            .mean()["session_duration_seconds"]
            .to_dict()
        )

    # ---------------------------
    # Project management tools
    # ---------------------------
    def project_management_get_task_information_by_id(
        self, task_id: Optional[str] = None, field: Optional[str] = None
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
        task_name: Optional[str] = None,
        assigned_to_email: Optional[str] = None,
        list_name: Optional[str] = None,
        due_date: Optional[str] = None,
        board: Optional[str] = None,
    ) -> Any:
        if not any([task_name, assigned_to_email, list_name, due_date, board]):
            return "No search parameters provided."
        tasks = self.project_tasks.copy()
        if task_name:
            tasks = tasks[tasks["task_name"].str.contains(task_name, case=False)]
        if assigned_to_email:
            tasks = tasks[tasks["assigned_to_email"].str.contains(assigned_to_email, case=False)]
        if list_name:
            tasks = tasks[tasks["list_name"].str.contains(list_name, case=False)]
        if due_date:
            tasks = tasks[tasks["due_date"].str.contains(due_date, case=False)]
        if board:
            tasks = tasks[tasks["board"].str.contains(board, case=False)]
        return tasks.to_dict(orient="records")

    def project_management_create_task(
        self,
        task_name: Optional[str] = None,
        assigned_to_email: Optional[str] = None,
        list_name: Optional[str] = None,
        due_date: Optional[str] = None,
        board: Optional[str] = None,
    ) -> Any:
        if not all([task_name, assigned_to_email, list_name, due_date, board]):
            return "Missing task details."

        assigned_to_email = assigned_to_email.lower()
        if assigned_to_email not in self.project_tasks["assigned_to_email"].str.lower().values:
            return "Assignee email not valid. Please choose from the list of team members."
        if list_name not in ["Backlog", "In Progress", "In Review", "Completed"]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if board not in ["Back end", "Front end", "Design"]:
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
        self.project_tasks = pd.concat([self.project_tasks, new_task], ignore_index=True)
        return task_id

    def project_management_delete_task(self, task_id: Optional[str] = None) -> Any:
        if not task_id:
            return "Task ID not provided."
        if task_id in self.project_tasks["task_id"].values:
            self.project_tasks = self.project_tasks[self.project_tasks["task_id"] != task_id]
            return "Task deleted successfully."
        return "Task not found."

    def project_management_update_task(
        self,
        task_id: Optional[str] = None,
        field: Optional[str] = None,
        new_value: Optional[str] = None,
    ) -> Any:
        if not task_id or not field or not new_value:
            return "Task ID, field, or new value not provided."

        if field == "assigned_to_email":
            new_value = new_value.lower()

        if field == "board" and new_value not in ["Back end", "Front end", "Design"]:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
        if field == "list_name" and new_value not in ["Backlog", "In Progress", "In Review", "Completed"]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if field == "assigned_to_email" and new_value not in self.project_tasks["assigned_to_email"].str.lower().values:
            return "Assignee email not valid. Please choose from the list of team members."

        if task_id not in self.project_tasks["task_id"].values:
            return "Task not found."
        if field not in self.project_tasks.columns:
            return "Field not valid."
        self.project_tasks.loc[self.project_tasks["task_id"] == task_id, field] = new_value
        return "Task updated successfully."

    # ---------------------------
    # CRM tools (customer_relationship_manager)
    # ---------------------------
    def customer_relationship_manager_search_customers(
        self,
        customer_name: Optional[str] = None,
        customer_email: Optional[str] = None,
        product_interest: Optional[str] = None,
        status: Optional[str] = None,
        assigned_to_email: Optional[str] = None,
        last_contact_date_min: Optional[str] = None,
        last_contact_date_max: Optional[str] = None,
        follow_up_by_min: Optional[str] = None,
        follow_up_by_max: Optional[str] = None,
    ) -> Any:
        customers = self.crm_data.copy()
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

        if customer_name:
            customers = customers[customers["customer_name"].str.contains(customer_name, case=False)]
        if customer_email:
            customers = customers[customers["customer_email"].str.contains(customer_email, case=False)]
        if product_interest:
            customers = customers[customers["product_interest"].str.contains(product_interest, case=False)]
        if status:
            customers = customers[customers["status"].str.contains(status, case=False)]
        if assigned_to_email:
            customers = customers[customers["assigned_to_email"].str.contains(assigned_to_email, case=False)]
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
        customer_id: Optional[str] = None,
        field: Optional[str] = None,
        new_value: Optional[str] = None,
    ) -> Any:
        if not customer_id or not field or not new_value:
            return "Customer ID, field, or new value not provided."

        if field == "status" and new_value not in ["Qualified", "Won", "Lost", "Lead", "Proposal"]:
            return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"

        if field == "product_interest" and new_value not in ["Software", "Hardware", "Services", "Consulting", "Training"]:
            return (
                "Product interest not valid. Please choose from: 'Software', 'Hardware', "
                "'Services', 'Consulting', 'Training'"
            )

        if field in {"customer_email", "assigned_to_email"}:
            new_value = new_value.lower()

        if customer_id not in self.crm_data["customer_id"].values:
            return "Customer not found."
        if field not in self.crm_data.columns:
            return (
                "Field not valid. Please choose from: 'customer_name', 'assigned_to_email', 'customer_email', "
                "'customer_phone', 'last_contact_date', 'product_interest', 'status', 'notes', 'follow_up_by'"
            )
        self.crm_data.loc[self.crm_data["customer_id"] == customer_id, field] = new_value
        return "Customer updated successfully."

    def customer_relationship_manager_add_customer(
        self,
        customer_name: Optional[str] = None,
        assigned_to_email: Optional[str] = None,
        status: Optional[str] = None,
        customer_email: Optional[str] = None,
        customer_phone: Optional[str] = None,
        last_contact_date: Optional[str] = None,
        product_interest: Optional[str] = None,
        notes: str = "",
        follow_up_by: Optional[str] = None,
    ) -> Any:
        if not all([customer_name, assigned_to_email, status]):
            return "Please provide all required fields: customer_name, assigned_to_email, status."

        assigned_to_email = assigned_to_email.lower()
        if customer_email:
            customer_email = customer_email.lower()

        new_id = str(int(self.crm_data["customer_id"].max()) + 1).zfill(8)
        new_customer = pd.DataFrame(
            {
                "customer_id": [new_id],
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
        return new_id

    def customer_relationship_manager_delete_customer(self, customer_id: Optional[str] = None) -> Any:
        if not customer_id:
            return "Customer ID not provided."
        if customer_id not in self.crm_data["customer_id"].values:
            return "Customer not found."
        self.crm_data = self.crm_data[self.crm_data["customer_id"] != customer_id]
        return "Customer deleted successfully."

    # ---------------------------
    # Company directory (always included)
    # ---------------------------
    def company_directory_find_email_address(self, name: str = "") -> Any:
        if name == "":
            return "Name not provided."
        name = name.lower()
        email_address = self._email_addresses[
            self._email_addresses["email_address"].str.contains(name)
        ]
        return email_address["email_address"].values


def get_function_name(action: str) -> str:
    # Matches upstream get_function_name(): first two dot segments before '('.
    # Examples:
    #   email.delete_email.func(...) -> email.delete_email
    #   email.delete_email(...)      -> email.delete_email
    head = action.split("(", 1)[0]
    parts = head.split(".")[0:2]
    return ".".join(parts)


def parse_action_to_tool_and_args(action: str) -> Optional[Tuple[str, Dict[str, Any], List[str]]]:
    """Parse an action string into (tool_name, args_dict, arg_order).

    Supports both:
    - WorkBench format:  email.delete_email.func(email_id="00000001")
    - Our trajectory:    email.delete_email(email_id='00000001')
    """
    s = (action or "").strip()
    if not s:
        return None

    # Some agent logs may wrap tool calls with additional text; keep it strict.
    try:
        node = ast.parse(s, mode="eval").body
    except Exception:
        return None
    if not isinstance(node, ast.Call):
        return None

    def func_name(expr: ast.expr) -> str:
        parts: List[str] = []
        cur: ast.expr = expr
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)

    full = func_name(node.func)
    if full.endswith(".func"):
        full = full[:-5]

    args: Dict[str, Any] = {}
    order: List[str] = []
    for kw in node.keywords:
        if kw.arg is None:
            continue
        order.append(kw.arg)
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            # Fallback to a string-ish representation.
            args[kw.arg] = getattr(kw.value, "id", None) or str(kw.value)

    return full, args, order


def format_func_call(tool_name: str, args: Dict[str, Any], arg_order: Optional[Iterable[str]] = None) -> str:
    """Format a tool call as WorkBench-style '<tool>.func(k=\"v\", ...)'."""
    parts: List[str] = []
    keys = list(arg_order) if arg_order is not None else list(args.keys())
    for k in keys:
        v = args.get(k)
        v_str = str(v)
        v_str = v_str.replace('"', '\\"')
        parts.append(f'{k}="{v_str}"')
    return f"{tool_name}.func(" + ", ".join(parts) + ")"


def execute_actions_and_get_state(actions: List[str], sandbox: WorkbenchSandbox) -> WorkbenchState:
    """Execute actions on the sandbox (errors are ignored) and return the final state snapshot."""
    for action in actions:
        parsed = parse_action_to_tool_and_args(action)
        if not parsed:
            continue
        tool_name, args, _ = parsed
        method_name = tool_name.replace(".", "_")
        fn: Optional[Callable[..., Any]] = getattr(sandbox, method_name, None)
        if fn is None:
            continue
        try:
            fn(**args)
        except Exception:
            # Upstream ignores tool execution errors when calculating metrics.
            continue
    return sandbox.snapshot()


def is_exact_match(predicted_actions: List[str], ground_truth_actions: List[str]) -> bool:
    tools_with_side_effects = set(TOOLS_WITH_SIDE_EFFECTS)
    pred = [a for a in predicted_actions if get_function_name(a) in tools_with_side_effects]
    pred = sorted([a.lower() for a in pred])
    gt = sorted([a.lower() for a in ground_truth_actions])
    return pred == gt


def evaluate_actions(
    *,
    predicted_actions: List[str],
    ground_truth_actions: List[str],
    sandbox_factory: Callable[[], WorkbenchSandbox],
    error: str = "",
) -> Dict[str, Any]:
    """Evaluate WorkBench metrics for one instance.

    Returns a dict with keys compatible with WorkBench's evaluation:
    - correct
    - exact_match
    - unwanted_side_effects
    """
    if error:
        return {
            "correct": 0,
            "exact_match": 0,
            "unwanted_side_effects": 0,
            "error": error,
        }

    original = sandbox_factory().snapshot()

    pred_sb = sandbox_factory()
    pred_state = execute_actions_and_get_state(predicted_actions, pred_sb)

    gt_sb = sandbox_factory()
    gt_state = execute_actions_and_get_state(ground_truth_actions, gt_sb)

    pred_calendar = _convert_df_strs_to_lowercase(pred_state.calendar)
    pred_email = _convert_df_strs_to_lowercase(pred_state.email)
    pred_analytics = _convert_df_strs_to_lowercase(pred_state.analytics)
    pred_pm = _convert_df_strs_to_lowercase(pred_state.project_management)
    pred_crm = _convert_df_strs_to_lowercase(pred_state.customer_relationship_manager)

    gt_calendar = _convert_df_strs_to_lowercase(gt_state.calendar)
    gt_email = _convert_df_strs_to_lowercase(gt_state.email)
    gt_analytics = _convert_df_strs_to_lowercase(gt_state.analytics)
    gt_pm = _convert_df_strs_to_lowercase(gt_state.project_management)
    gt_crm = _convert_df_strs_to_lowercase(gt_state.customer_relationship_manager)

    correct = (
        pred_calendar.equals(gt_calendar)
        and pred_email.equals(gt_email)
        and pred_analytics.equals(gt_analytics)
        and pred_pm.equals(gt_pm)
        and pred_crm.equals(gt_crm)
    )

    # Side effects: any state change relative to original state AND not correct.
    state_changed = not pred_state.calendar.equals(original.calendar)
    state_changed |= not pred_state.email.equals(original.email)
    state_changed |= not pred_state.analytics.equals(original.analytics)
    state_changed |= not pred_state.project_management.equals(original.project_management)
    state_changed |= not pred_state.customer_relationship_manager.equals(original.customer_relationship_manager)

    return {
        "correct": int(correct),
        "exact_match": int(is_exact_match(predicted_actions, ground_truth_actions)),
        "unwanted_side_effects": int(state_changed and (not correct)),
        "error": "",
    }


_DATE_MINOR_ERROR = "2023-11-29"
_DATE_MINOR_ERROR_REPLACEMENT = "2023-11-30"


def end_date_minor_error(ground_truth_actions: List[str], predicted_actions: List[str]) -> bool:
    matches = 0
    for func in ground_truth_actions:
        if _DATE_MINOR_ERROR in func:
            if func.replace(_DATE_MINOR_ERROR, _DATE_MINOR_ERROR_REPLACEMENT) in predicted_actions:
                matches += 1
    if len(ground_truth_actions) == 0:
        return False
    return matches == len(ground_truth_actions)


def meeting_start_time_error(ground_truth_actions: List[str], predicted_actions: List[str]) -> bool:
    matches = 0
    next_free_time_ground_truth = "13:00:00"
    common_error_times = ["09:00:00", "11:00:00", "15:00:00", "15:30:00"]
    for func in ground_truth_actions:
        if next_free_time_ground_truth in func:
            for t in common_error_times:
                if func.replace(next_free_time_ground_truth, t) in predicted_actions:
                    matches += 1
                    break
    if len(ground_truth_actions) == 0:
        return False
    return matches == len(ground_truth_actions)
