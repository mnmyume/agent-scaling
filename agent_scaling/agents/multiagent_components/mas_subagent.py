import threading
import traceback
from typing import Any, Dict, Literal, Optional, cast

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from agent_scaling.agents.base import BaseAgentWithTools
from agent_scaling.agents.output_validation import validate_json
from agent_scaling.datasets import DatasetInstance
from agent_scaling.logger import logger
from agent_scaling.utils.token_budget import TokenBudgetManager, extract_token_usage

from .budgeting import MASBudgetAllocator
from .conversation import (
    FinalAnswerCandidate,
    SubAgentConversationHistory,
    SubAgentRoundResult,
)


class WorkerSubagent(BaseAgentWithTools):
    """Generic worker subagent that works with proper environment access"""

    required_prompts = ["subagent"]

    def __init__(
        self,
        agent_id: str,
        objective: str,
        original_query: str,
        strategy: str,
        task_instance: DatasetInstance,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        llm_params_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.agent_id = agent_id
        # Core agent attributes
        self.objective = objective
        self.original_query = original_query
        self.strategy = strategy
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.task_instance = task_instance
        self.llm_params_dict = llm_params_dict or {}
        self.env, self.llm_w_tools = self.init_environment(task_instance, agent_id)
        self.shared_prompt_templates = self.get_dataset_prompt_templates(self.env)
        # Conversation management
        self.conv_history = SubAgentConversationHistory(agent_id=agent_id)

        self._execution_lock = threading.Lock()
        self.budget_manager: Optional[TokenBudgetManager] = None
        self.budget_allocator: Optional[MASBudgetAllocator] = None

        logger.info(
            f"WorkerSubagent {agent_id} initialized with objective: {objective[:100]}..."
        )

    @classmethod
    def init_from_agent(
        cls,
        agent: BaseAgentWithTools,
        agent_id: str,
        objective: str,
        original_query: str,
        strategy: str,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        llm_params_dict: Optional[Dict[str, Any]] = None,
        budget_allocator: Optional[MASBudgetAllocator] = None,
        **kwargs,
    ):
        metrics_collector = kwargs.pop(
            "metrics_collector", getattr(agent, "metrics_collector", None)
        )
        worker = cls(
            agent_id=agent_id,
            objective=objective,
            original_query=original_query,
            strategy=strategy,
            min_iterations_per_agent=min_iterations_per_agent,
            max_iterations_per_agent=max_iterations_per_agent,
            llm_params_dict=llm_params_dict,
            llm=agent.llm,
            dataset=agent.dataset,
            prompts=agent.prompts,
            tools=agent.tools,
            env=agent.env_name,
            env_prompts=agent.env_prompts,
            metrics_collector=metrics_collector,
            **kwargs,
        )
        worker.budget_allocator = budget_allocator
        return worker

    def _prepare_llm_kwargs(
        self,
        messages: Any,
        budget_bucket: str,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs = {**self.llm_params_dict, **(extra_kwargs or {})}
        if self.budget_allocator is not None:
            return self.budget_allocator.prepare_call(budget_bucket, messages, kwargs)
        return kwargs

    def _consume_response(self, budget_bucket: str, response: AIMessage) -> None:
        if self.budget_allocator is not None:
            self.budget_allocator.consume_response(budget_bucket, response)
            return
        if self.budget_manager is not None:
            inp_tok, out_tok = extract_token_usage(response)
            self.budget_manager.consume(inp_tok, out_tok)

    def _compile_round_messages(
        self,
        message: Optional[str],
        instruction_type: Literal["lead_agent", "peer", "independent"],
    ):
        if instruction_type == "lead_agent":
            return (
                self.prompts["subagent"]
                .get_template("start_with_orchestrator_guidance")
                .compile(
                    orchestrator_objective=self.objective,
                    orchestrator_guidance=message or "",
                    **self.shared_prompt_templates,
                )
            )
        if instruction_type == "peer":
            return (
                self.prompts["subagent"]
                .get_template("start_with_peer_context")
                .compile(
                    agent_objective=self.objective,
                    agent_strategy=self.strategy,
                    peer_context=message or "No peer feedback available.",
                    **self.shared_prompt_templates,
                )
            )
        return (
            self.prompts["subagent"]
            .get_template("start_independent_work")
            .compile(
                agent_objective=self.objective,
                agent_strategy=self.strategy,
                **self.shared_prompt_templates,
            )
        )

    def _run_one_round(
        self,
        message: Optional[str],
        instruction_type: Literal["lead_agent", "peer", "independent"],
        budget_bucket: str,
        max_iterations: Optional[int] = None,
    ) -> SubAgentRoundResult:
        """Process the task by calling tools with proper conversation state management"""
        logger.info(f"Agent {self.agent_id} starting process")
        logger.info(f"Agent {self.agent_id} objective: {self.objective}")

        messages = self._compile_round_messages(message, instruction_type)
        # Continue from previous conversation state if exists and valid
        if len(self.conv_history.internal_comms) > 0:
            logger.info(
                f"Agent {self.agent_id} continuing from previous conversation state with {len(self.conv_history.internal_comms)} messages"
            )
            last_n_messages = self.conv_history.last_n_iterations_messages(n=20)
            messages.extend(last_n_messages)

        # Simple iteration loop with conversation persistence
        curr_iteration = 0
        iteration_limit = max_iterations or self.max_iterations_per_agent
        for iteration in range(iteration_limit):
            curr_iteration += 1
            logger.info(
                f"Agent {self.agent_id} iteration {iteration}/{iteration_limit}"
            )
            # Invoke LLM with tools using retry logic
            self.env.set_metrics_context(
                round_num=self.conv_history.current_round,
                iteration=curr_iteration,
            )
            response: AIMessage = cast(
                AIMessage,
                self._invoke_with_metrics(
                    self.llm_w_tools,
                    messages,  # type: ignore
                    agent_id=self.agent_id,
                    llm_kwargs=self._prepare_llm_kwargs(
                        messages=messages,
                        budget_bucket=budget_bucket,
                        extra_kwargs={"num_retries": 2},
                    ),
                    call_type=f"{budget_bucket}_tool_loop",
                    round_num=self.conv_history.current_round,
                    iteration=curr_iteration,
                ),
            )
            self._consume_response(budget_bucket, response)
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                response.tool_calls = [response.tool_calls[0]]
                tool_name = ""

                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)  # type: ignore
                self.conv_history.add_internal_message(
                    message=response_msg,  # type: ignore
                    iteration_num=curr_iteration,
                )
                try:
                    # Execute tool with retry logic
                    tool_name = tool_call["name"]
                    tool_resp = self.env.execute_tool(tool_call)
                except Exception as e:
                    # Add error message to conversation state (consistent with single_agent.py)
                    error_msg = {
                        "role": "user",
                        "content": f"ERROR: Tool **{tool_name}** failed with error: {str(e)}. Please check the tool call.",
                    }
                    messages.append(error_msg)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=error_msg,
                        iteration_num=curr_iteration,  # type: ignore
                    )
                    logger.warning(
                        f"Tool **{tool_name}** failed with error: {str(e)}\n{traceback.format_exc()}"
                    )
                else:
                    # Add tool response to messages and state
                    tool_msg = convert_to_openai_messages(tool_resp)
                    messages.append(tool_msg)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=tool_msg,  # type: ignore
                        iteration_num=curr_iteration,
                    )
                    # Check if done
                    if tool_name == "done":
                        logger.info(
                            f"Agent {self.agent_id} decided to finish with 'done' tool"
                        )
                        break
                    elif self.env.env_done():
                        logger.info(
                            f"Environment {self.env_name} is done for agent {self.agent_id}"
                        )
                        break
            else:
                # No tool calls - add error message to conversation state
                error_msg = {
                    "role": "user",
                    "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                }
                messages.append(error_msg)  # type: ignore
                self.conv_history.add_internal_message(
                    message=error_msg,  # type: ignore
                    iteration_num=curr_iteration,
                )
                logger.warning(
                    f"Agent {self.agent_id}: No tool calls found in iteration {iteration}"
                )
        curr_iteration += 1
        findings_message = (
            self.prompts["subagent"]
            .get_template("summarize_findings", with_base=False)
            .compile()
        )[0]
        messages.append(findings_message)  # type: ignore
        self.conv_history.add_internal_message(
            message=findings_message,  # type: ignore
            iteration_num=curr_iteration,
        )

        llm_response = self._invoke_with_metrics(
            self.llm,
            messages,
            agent_id=self.agent_id,
            llm_kwargs=self._prepare_llm_kwargs(messages, budget_bucket),
            call_type=f"{budget_bucket}_summary",
            round_num=self.conv_history.current_round,
            iteration=curr_iteration,
        )
        self._consume_response(budget_bucket, llm_response)

        # Update agent's conversation
        self.conv_history.add_internal_message(
            message=convert_to_openai_messages(llm_response),  # type: ignore
            iteration_num=curr_iteration,
        )
        logger.info(
            f"Agent {self.agent_id} completed round {self.conv_history.current_round} (n_iterations={self.conv_history.curr_iteration}). Findings:\n{llm_response.text()} "
        )
        if self.metrics_collector is not None:
            self.metrics_collector.log_agent_output(
                agent_id=self.agent_id,
                output_type="finding",
                content=llm_response.text(),
                round=self.conv_history.current_round,
                iteration=curr_iteration,
            )

        return SubAgentRoundResult(
            agent_id=self.agent_id,
            findings=llm_response.text(),
            env_status=self.env.env_status(),
        )

    def _process_external_message(
        self,
        message: Optional[str],
        sender_role: Literal["lead_agent", "peer", "independent"],
        sender_id: Optional[str] = None,
        budget_bucket: str = "worker",
        max_iterations: Optional[int] = None,
    ) -> SubAgentRoundResult:
        """Process a round with external guidance while preserving conversation state."""
        # Increment round counter for new orchestrator message
        self.conv_history.start_new_round()
        logger.info(
            f"Agent {self.agent_id} ({self.strategy}) processing {sender_role} input for round {self.conv_history.current_round}"
        )
        if sender_role != "independent":
            self.conv_history.add_external_message(
                sender_role,
                message or "",
                sender_id=sender_id,
                recipient_id=self.agent_id,
                channel="peer" if sender_role == "peer" else "orchestrator",
            )
        round_result = self._run_one_round(
            message=message,
            instruction_type=sender_role,
            budget_bucket=budget_bucket,
            max_iterations=max_iterations,
        )
        self.conv_history.add_external_message(
            "subagent",
            round_result.findings,
            sender_id=self.agent_id,
            recipient_id=(
                "aggregator"
                if sender_role == "independent"
                else ("peer_network" if sender_role == "peer" else "lead_agent")
            ),
            channel=(
                "aggregation"
                if sender_role == "independent"
                else ("peer" if sender_role == "peer" else "orchestrator")
            ),
        )

        if self.env.env_done():
            self.conv_history.status = "completed"
        elif self.should_stop_due_to_rate_limiting():
            self.conv_history.status = "rate_limited"

        return round_result

    def process_orchestrator_message(self, message: str) -> SubAgentRoundResult:
        return self._process_external_message(
            message=message,
            sender_role="lead_agent",
            sender_id="lead_agent",
            budget_bucket="worker",
        )

    def run_independent_round(self) -> SubAgentRoundResult:
        return self._process_external_message(
            message=None,
            sender_role="independent",
            budget_bucket="worker",
        )

    def process_peer_message(
        self,
        message: str,
        sender_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> SubAgentRoundResult:
        return self._process_external_message(
            message=message,
            sender_role="peer",
            sender_id=sender_id or "peer_network",
            budget_bucket="peer",
            max_iterations=max_iterations,
        )

    def propose_final_answer(self) -> FinalAnswerCandidate:
        messages = self.conv_history.last_n_iterations_messages(n=25)
        messages.extend(
            self.prompts["subagent"]
            .get_template("final_answer", with_base=False)
            .compile(
                agent_objective=self.objective,
                agent_strategy=self.strategy,
            )
        )
        response = self._invoke_with_metrics(
            self.llm,
            messages,
            agent_id=self.agent_id,
            llm_kwargs=self._prepare_llm_kwargs(messages, "consensus"),
            call_type="consensus",
            round_num=self.conv_history.current_round,
            iteration=self.conv_history.curr_iteration,
        )
        self._consume_response("consensus", response)

        try:
            payload = validate_json(response.text())
        except Exception:
            payload = {
                "final_answer": response.text().strip(),
                "confidence": 0,
                "rationale": "Fallback parsing path.",
            }
        if self.metrics_collector is not None:
            self.metrics_collector.log_agent_output(
                agent_id=self.agent_id,
                output_type="final_candidate",
                content=str(payload.get("final_answer", "")).strip(),
                round=self.conv_history.current_round,
                iteration=self.conv_history.curr_iteration,
            )
        return FinalAnswerCandidate(
            agent_id=self.agent_id,
            answer=str(payload.get("final_answer", "")).strip(),
            confidence=int(payload.get("confidence", 0) or 0),
            rationale=str(payload.get("rationale", "")).strip(),
        )

    def should_stop_due_to_rate_limiting(self) -> bool:
        """Check if agent should stop due to excessive rate limiting"""
        if hasattr(self.env, "should_stop_due_to_rate_limiting"):
            return self.env.should_stop_due_to_rate_limiting()
        return False
