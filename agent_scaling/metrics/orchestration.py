from __future__ import annotations

from typing import Dict, List

from agent_scaling.metrics.artifacts import MetricsArtifacts
from agent_scaling.metrics.usage import extract_usage_from_model_response
from agent_scaling.agents.multiagent_components.conversation import OrchestrationResult


def build_metrics_artifacts_from_orchestration(
    result: OrchestrationResult,
) -> MetricsArtifacts:
    artifacts = MetricsArtifacts()

    # LLM usage from lead agent
    for message in result.lead_agent_conversation.messages:
        if message.litellm_message is not None:
            artifacts.llm_calls.append(
                extract_usage_from_model_response(message.litellm_message)
            )

    # LLM usage from subagents
    for conv in result.subagent_conversations.values():
        for round_msgs in conv.internal_comms:
            for msg in round_msgs:
                if msg.litellm_message is not None:
                    artifacts.llm_calls.append(
                        extract_usage_from_model_response(msg.litellm_message)
                    )

    # Inter-agent messages
    artifacts.inter_agent_messages = sum(
        len(conv.external_comms) for conv in result.subagent_conversations.values()
    )

    # Reasoning turns / tool calls
    artifacts.reasoning_turns = result.combined_env_status.num_steps
    artifacts.tool_calls = result.combined_env_status.num_steps

    # Subagent outputs (last findings)
    for agent_id, findings in result.subagent_findings.items():
        last_non_empty = ""
        for finding in reversed(findings):
            if finding and finding.strip():
                last_non_empty = finding.strip()
                break
        if last_non_empty:
            artifacts.subagent_outputs.append(last_non_empty)
            artifacts.subagent_outputs_by_id[agent_id] = last_non_empty

    artifacts.final_output = result.synthesized_answer

    # Pre/post states for info gain proxy
    if artifacts.subagent_outputs:
        artifacts.pre_state_text = "\n".join(
            f"{agent_id}: {text}"
            for agent_id, text in artifacts.subagent_outputs_by_id.items()
        )
    artifacts.post_state_text = artifacts.final_output

    artifacts.finalize()
    return artifacts
