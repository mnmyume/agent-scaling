from __future__ import annotations

from typing import Any, Dict

from agent_scaling.env.workbench_utils import WorkbenchSandbox

from .base import AgentEnvironmentTools
from .registry import register_env
from .tools import cls_tool


@register_env("workbench")
class WorkbenchEnvironment(AgentEnvironmentTools):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sandbox = WorkbenchSandbox()
        self.done_called = False

    def env_done(self) -> bool:
        return self.done_called

    @cls_tool
    def calendar_get_event_information_by_id(
        self,
        event_id: str | None = None,
        field: str | None = None,
    ) -> Any:
        """Get a specific calendar field for a single event ID."""
        return self.sandbox.calendar_get_event_information_by_id(
            event_id=event_id, field=field
        )

    @cls_tool
    def calendar_search_events(
        self,
        query: str = "",
        time_min: str | None = None,
        time_max: str | None = None,
    ) -> Any:
        """Search calendar events by query text and optional datetime bounds."""
        return self.sandbox.calendar_search_events(
            query=query, time_min=time_min, time_max=time_max
        )

    @cls_tool
    def calendar_create_event(
        self,
        event_name: str | None = None,
        participant_email: str | None = None,
        event_start: str | None = None,
        duration: str | None = None,
    ) -> str:
        """Create a new calendar event and return its event ID."""
        return self.sandbox.calendar_create_event(
            event_name=event_name,
            participant_email=participant_email,
            event_start=event_start,
            duration=duration,
        )

    @cls_tool
    def calendar_delete_event(self, event_id: str | None = None) -> str:
        """Delete a calendar event by ID."""
        return self.sandbox.calendar_delete_event(event_id=event_id)

    @cls_tool
    def calendar_update_event(
        self,
        event_id: str | None = None,
        field: str | None = None,
        new_value: str | None = None,
    ) -> str:
        """Update a single field on an existing calendar event."""
        return self.sandbox.calendar_update_event(
            event_id=event_id, field=field, new_value=new_value
        )

    @cls_tool
    def email_get_email_information_by_id(
        self,
        email_id: str | None = None,
        field: str | None = None,
    ) -> Any:
        """Get a specific field for one email by ID."""
        return self.sandbox.email_get_email_information_by_id(
            email_id=email_id, field=field
        )

    @cls_tool
    def email_search_emails(
        self,
        query: str = "",
        date_min: str | None = None,
        date_max: str | None = None,
    ) -> Any:
        """Search emails by text and optional inclusive date bounds."""
        return self.sandbox.email_search_emails(
            query=query, date_min=date_min, date_max=date_max
        )

    @cls_tool
    def email_send_email(
        self,
        recipient: str | None = None,
        subject: str | None = None,
        body: str | None = None,
    ) -> str:
        """Send an email from the sandbox user to a recipient."""
        return self.sandbox.email_send_email(
            recipient=recipient, subject=subject, body=body
        )

    @cls_tool
    def email_delete_email(self, email_id: str | None = None) -> str:
        """Delete an email by ID."""
        return self.sandbox.email_delete_email(email_id=email_id)

    @cls_tool
    def email_forward_email(
        self, email_id: str | None = None, recipient: str | None = None
    ) -> str:
        """Forward an existing email to another recipient."""
        return self.sandbox.email_forward_email(email_id=email_id, recipient=recipient)

    @cls_tool
    def email_reply_email(
        self, email_id: str | None = None, body: str | None = None
    ) -> str:
        """Reply to an existing email thread."""
        return self.sandbox.email_reply_email(email_id=email_id, body=body)

    @cls_tool
    def analytics_get_visitor_information_by_id(
        self, visitor_id: str | None = None
    ) -> Any:
        """Retrieve analytics row(s) for a visitor ID."""
        return self.sandbox.analytics_get_visitor_information_by_id(visitor_id=visitor_id)

    @cls_tool
    def analytics_create_plot(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        value_to_plot: str | None = None,
        plot_type: str | None = None,
    ) -> str:
        """Create a plot artifact path for analytics data over a date range."""
        return self.sandbox.analytics_create_plot(
            time_min=time_min,
            time_max=time_max,
            value_to_plot=value_to_plot,
            plot_type=plot_type,
        )

    @cls_tool
    def analytics_total_visits_count(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> Dict[str, Any]:
        """Count total visits per day across an optional date range."""
        return self.sandbox.analytics_total_visits_count(
            time_min=time_min, time_max=time_max
        )

    @cls_tool
    def analytics_engaged_users_count(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> Dict[str, Any]:
        """Count engaged users per day across an optional date range."""
        return self.sandbox.analytics_engaged_users_count(
            time_min=time_min, time_max=time_max
        )

    @cls_tool
    def analytics_traffic_source_count(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        traffic_source: str | None = None,
    ) -> Dict[str, Any]:
        """Count visits per day for a traffic source across an optional date range."""
        return self.sandbox.analytics_traffic_source_count(
            time_min=time_min, time_max=time_max, traffic_source=traffic_source
        )

    @cls_tool
    def analytics_get_average_session_duration(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> Dict[str, Any]:
        """Return average session duration per day across an optional date range."""
        return self.sandbox.analytics_get_average_session_duration(
            time_min=time_min, time_max=time_max
        )

    @cls_tool
    def project_management_get_task_information_by_id(
        self,
        task_id: str | None = None,
        field: str | None = None,
    ) -> Any:
        """Get a specific task field for a task ID."""
        return self.sandbox.project_management_get_task_information_by_id(
            task_id=task_id, field=field
        )

    @cls_tool
    def project_management_search_tasks(
        self,
        task_name: str | None = None,
        assigned_to_email: str | None = None,
        list_name: str | None = None,
        due_date: str | None = None,
        board: str | None = None,
    ) -> Any:
        """Search project-management tasks by one or more task fields."""
        return self.sandbox.project_management_search_tasks(
            task_name=task_name,
            assigned_to_email=assigned_to_email,
            list_name=list_name,
            due_date=due_date,
            board=board,
        )

    @cls_tool
    def project_management_create_task(
        self,
        task_name: str | None = None,
        assigned_to_email: str | None = None,
        list_name: str | None = None,
        due_date: str | None = None,
        board: str | None = None,
    ) -> str:
        """Create a new task in the project-management board."""
        return self.sandbox.project_management_create_task(
            task_name=task_name,
            assigned_to_email=assigned_to_email,
            list_name=list_name,
            due_date=due_date,
            board=board,
        )

    @cls_tool
    def project_management_delete_task(self, task_id: str | None = None) -> str:
        """Delete a task by ID."""
        return self.sandbox.project_management_delete_task(task_id=task_id)

    @cls_tool
    def project_management_update_task(
        self,
        task_id: str | None = None,
        field: str | None = None,
        new_value: str | None = None,
    ) -> str:
        """Update a single field on an existing task."""
        return self.sandbox.project_management_update_task(
            task_id=task_id, field=field, new_value=new_value
        )

    @cls_tool
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
        """Search CRM customers using one or more CRM fields and date filters."""
        return self.sandbox.customer_relationship_manager_search_customers(
            customer_name=customer_name,
            customer_email=customer_email,
            product_interest=product_interest,
            status=status,
            assigned_to_email=assigned_to_email,
            last_contact_date_min=last_contact_date_min,
            last_contact_date_max=last_contact_date_max,
            follow_up_by_min=follow_up_by_min,
            follow_up_by_max=follow_up_by_max,
        )

    @cls_tool
    def customer_relationship_manager_update_customer(
        self,
        customer_id: str | None = None,
        field: str | None = None,
        new_value: str | None = None,
    ) -> str:
        """Update a single field on a CRM customer record."""
        return self.sandbox.customer_relationship_manager_update_customer(
            customer_id=customer_id, field=field, new_value=new_value
        )

    @cls_tool
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
        """Create a new CRM customer record."""
        return self.sandbox.customer_relationship_manager_add_customer(
            customer_name=customer_name,
            assigned_to_email=assigned_to_email,
            status=status,
            customer_email=customer_email,
            customer_phone=customer_phone,
            last_contact_date=last_contact_date,
            product_interest=product_interest,
            notes=notes,
            follow_up_by=follow_up_by,
        )

    @cls_tool
    def customer_relationship_manager_delete_customer(
        self, customer_id: str | None = None
    ) -> str:
        """Delete a CRM customer record by ID."""
        return self.sandbox.customer_relationship_manager_delete_customer(
            customer_id=customer_id
        )

    @cls_tool
    def company_directory_find_email_address(self, name: str = "") -> Any:
        """Find one or more @atlas.com employee email addresses by person name."""
        return self.sandbox.company_directory_find_email_address(name=name)

    @cls_tool
    def done(self) -> str:
        """Finish the task after all required state changes have been made."""
        self.done_called = True
        self.success = True
        return "Done."
