from typing import Any, Dict, Optional, Self

from litellm import _logging as litellm_logging
from pydantic import BaseModel, Field, model_validator

from agent_scaling.agents import AgentSystem, get_agent_cls
from agent_scaling.langfuse_client import get_lf_client
from agent_scaling.metrics import PaperMetricsConfig
from agent_scaling.logger import logger
from agent_scaling.utils import (
    disable_local_cache,
    enable_local_cache,
    enable_local_logging,
)

from .dataset import DatasetConfig
from .llm import LLMConfig, LLMOverride
from .prompts import Prompt


class MultiAgentResearchConfig(BaseModel):
    n_base_agents: int = 3
    max_orchestrator_turns: int = 2
    min_searches_per_agent: int = 3
    min_iterations_per_agent: int = 3
    max_iterations_per_agent: int = 7
    max_rounds: Optional[int] = None
    enable_peer_communication: Optional[bool] = None
    consensus_threshold: Optional[float] = None
    task_blurb: Optional[str] = None
    orchestrator_llm: Optional[LLMOverride] = None
    subagent_llm: Optional[LLMOverride] = None
    subagent_llms: Optional[list[LLMOverride]] = None


class AgentConfig(BaseModel):
    name: str
    prompts: Dict[str, Prompt] = Field(default_factory=dict)
    agent_specific_config: Optional[MultiAgentResearchConfig] = None

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        agent_cls = get_agent_cls(self.name)
        agent_cls.check_required_prompts(self.prompts)
        return self

    @model_validator(mode="before")
    @classmethod
    def move_agent_specific_fields(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        agent_specific = dict(data.get("agent_specific_config") or {})
        for field_name in MultiAgentResearchConfig.model_fields.keys():
            if field_name in data:
                if field_name not in agent_specific:
                    agent_specific[field_name] = data.pop(field_name)
                else:
                    data.pop(field_name)
        if agent_specific:
            data["agent_specific_config"] = agent_specific
        return data

    @model_validator(mode="before")
    @classmethod
    def add_prompt_names(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        prompts = dict(data.get("prompts", {}))
        for k, prompt in prompts.items():
            prompt = dict(prompt)
            if prompt.get("name") is None:
                prompt["name"] = k
            prompts[k] = prompt
        data["prompts"] = prompts
        return data

    def get_run_metadata(self) -> Dict[str, Any]:
        prompts = {}
        assert self.prompts is not None, "Prompts must be defined in AgentConfig"
        for k, prompt in self.prompts.items():
            prompts[k] = {
                "name": prompt.name,
            }
        return {
            "name": self.name,
            "prompts": prompts,
        }


class RunConfig(BaseModel):
    agent: AgentConfig
    dataset: DatasetConfig
    llm: LLMConfig
    run_name: str
    save_dir: Optional[str] = None
    log_langfuse: bool = True
    use_disk_cache: bool = False
    debug: bool = False
    max_instances: Optional[int] = None
    num_workers: int = 1
    metrics: PaperMetricsConfig = Field(default_factory=PaperMetricsConfig)

    @property
    def run_parallel(self) -> bool:
        return self.num_workers > 1

    def model_post_init(self, context: Any) -> None:
        client = get_lf_client()
        if self.log_langfuse and not self.dataset.from_langfuse:
            logger.warning(
                "log_langfuse requested but dataset.from_langfuse is False; disabling Langfuse logging."
            )
        self.log_langfuse = (
            self.log_langfuse
            and self.dataset.from_langfuse
            and client is not None
            and self.dataset.langfuse_dataset is not None
        )
        if not self.log_langfuse and not self.run_parallel:
            enable_local_logging(prompt_only=True)
        litellm_logging._disable_debugging()  # type: ignore
        # Disable litellm debugging logs
        if self.use_disk_cache:
            enable_local_cache()
        else:
            disable_local_cache()

    def get_agent(self) -> AgentSystem:
        agent_kwargs: Dict[str, Any] = {}
        if self.agent.agent_specific_config is not None:
            agent_kwargs = self.agent.agent_specific_config.model_dump(
                exclude_none=True
            )
            if self.agent.agent_specific_config.orchestrator_llm is not None:
                agent_kwargs["orchestrator_llm"] = (
                    self.agent.agent_specific_config.orchestrator_llm.merge(self.llm)
                )
            if self.agent.agent_specific_config.subagent_llm is not None:
                agent_kwargs["subagent_llm"] = (
                    self.agent.agent_specific_config.subagent_llm.merge(self.llm)
                )
            if self.agent.agent_specific_config.subagent_llms is not None:
                agent_kwargs["subagent_llms"] = [
                    override.merge(self.llm)
                    for override in self.agent.agent_specific_config.subagent_llms
                ]
        return get_agent_cls(self.agent.name).from_config(
            llm_config=self.llm,
            dataset_config=self.dataset,
            prompts=self.agent.prompts,
            **agent_kwargs,
        )

    def get_run_metadata(self) -> Dict[str, Any]:
        agent_meta = self.agent.get_run_metadata()
        if self.agent.agent_specific_config is not None:
            agent_meta["config"] = self.agent.agent_specific_config.model_dump(
                exclude_none=True
            )
        agent_meta["effective_llms"] = self._get_effective_agent_llms()
        ret: Dict[str, Any] = {
            "agent": agent_meta,
            "llm": self.llm.model_dump(exclude_none=True),
            "dataset": self.dataset.model_dump(exclude_none=True),
            "metrics": self.metrics.model_dump(exclude_none=True),
        }
        if self.save_dir is not None:
            ret["save_dir"] = self.save_dir
        ret["run_name"] = self.run_name
        ret["num_workers"] = self.num_workers
        return ret

    def _get_effective_agent_llms(self) -> Dict[str, Any]:
        base_llm = self.llm
        effective: Dict[str, Any] = {
            "base": base_llm.model_dump(exclude_none=True),
        }
        cfg = self.agent.agent_specific_config
        if cfg is None:
            return effective

        # Orchestrator (for centralized/hybrid style agents)
        if cfg.orchestrator_llm is not None:
            effective["orchestrator"] = cfg.orchestrator_llm.merge(base_llm).model_dump(
                exclude_none=True
            )
        elif self.agent.name in {
            "multi-agent-centralized",
            "multi-agent-hybrid",
            "multi-agent-research",
        }:
            effective["orchestrator"] = base_llm.model_dump(exclude_none=True)

        # Subagents
        if cfg.subagent_llms is not None:
            effective["subagents"] = [
                override.merge(base_llm).model_dump(exclude_none=True)
                for override in cfg.subagent_llms
            ]
        else:
            default_sub = (
                cfg.subagent_llm.merge(base_llm)
                if cfg.subagent_llm is not None
                else base_llm
            )
            num_agents = cfg.n_base_agents or 0
            if num_agents:
                effective["subagents"] = [
                    default_sub.model_dump(exclude_none=True)
                    for _ in range(num_agents)
                ]
            else:
                effective["subagents"] = [
                    default_sub.model_dump(exclude_none=True)
                ]

        return effective

    def get_baseline_candidate_models(self) -> list[str]:
        """Models eligible for SAS baseline auto-discovery.

        For heterogeneous multi-agent runs, we prioritize subagent models.
        If no subagent override exists, fall back to the run base model.
        """
        candidates: set[str] = set()
        cfg = self.agent.agent_specific_config
        if cfg is not None:
            if cfg.subagent_llms is not None and len(cfg.subagent_llms) > 0:
                for override in cfg.subagent_llms:
                    merged = override.merge(self.llm)
                    if merged.model:
                        candidates.add(str(merged.model))
            elif cfg.subagent_llm is not None:
                merged = cfg.subagent_llm.merge(self.llm)
                if merged.model:
                    candidates.add(str(merged.model))

        if not candidates and self.llm.model:
            candidates.add(str(self.llm.model))
        return sorted(candidates)
