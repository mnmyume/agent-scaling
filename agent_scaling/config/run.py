from typing import Any, Dict, Optional, Self

from litellm import _logging as litellm_logging
from pydantic import BaseModel, Field, model_validator

from agent_scaling.agents import AgentSystem, get_agent_cls
from agent_scaling.langfuse_client import get_lf_client
from agent_scaling.utils import (
    disable_local_cache,
    enable_local_cache,
    enable_local_logging,
)

from .dataset import DatasetConfig
from .llm import LLMConfig
from .prompts import Prompt


class TokenBudgetConfig(BaseModel):
    enabled: bool = True
    total_tokens_per_instance: int = 4800


class MultiAgentConfig(BaseModel):
    max_steps: Optional[int] = None
    n_base_agents: int = 3
    min_iterations_per_agent: int = 3
    max_iterations_per_agent: int = 3
    max_rounds: int = 5
    peer_rounds: int = 1
    peer_fanout: Optional[int] = None
    peer_max_iterations: int = 1
    consensus_threshold: float = 0.67
    task_blurb: Optional[str] = None
    max_execution_time: int = 300
    worker_timeout: int = 120
    max_findings: int = 100
    communication: Optional[Dict[str, Any]] = None


MultiAgentResearchConfig = MultiAgentConfig


class AgentConfig(BaseModel):
    name: str
    prompts: Dict[str, Prompt] = Field(default_factory=dict)
    agent_specific_config: Optional[MultiAgentConfig] = None

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        agent_cls = get_agent_cls(self.name)
        agent_cls.check_required_prompts(self.prompts)
        return self

    @model_validator(mode="before")
    @classmethod
    def add_prompt_names(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        agent_specific_config = dict(data.get("agent_specific_config") or {})

        if "debate_rounds" in data and "max_rounds" not in agent_specific_config:
            agent_specific_config["max_rounds"] = data.pop("debate_rounds")
        if "peer_exchange_rounds" in data and "peer_rounds" not in agent_specific_config:
            agent_specific_config["peer_rounds"] = data.pop("peer_exchange_rounds")

        enable_peer_communication = data.pop("enable_peer_communication", None)
        if enable_peer_communication is False:
            agent_specific_config["peer_rounds"] = 0

        for field_name in MultiAgentConfig.model_fields:
            if field_name in data:
                agent_specific_config[field_name] = data.pop(field_name)

        if agent_specific_config:
            data["agent_specific_config"] = agent_specific_config

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
            "agent_specific_config": (
                self.agent_specific_config.model_dump(exclude_none=True)
                if self.agent_specific_config is not None
                else None
            ),
        }


class RunConfig(BaseModel):
    agent: AgentConfig
    dataset: DatasetConfig
    llm: LLMConfig
    run_name: str
    save_dir: Optional[str] = None
    resume: bool = False
    log_langfuse: bool = True
    use_disk_cache: bool = False
    debug: bool = False
    max_instances: Optional[int] = None
    num_workers: int = 1
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)

    @property
    def run_parallel(self) -> bool:
        return self.num_workers > 1

    def model_post_init(self, context: Any) -> None:
        client = get_lf_client()
        self.log_langfuse = (
            self.log_langfuse and client is not None
            # and self.dataset.langfuse_dataset is not None
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
        return get_agent_cls(self.agent.name).from_config(
            llm_config=self.llm,
            dataset_config=self.dataset,
            prompts=self.agent.prompts,
            **(
                self.agent.agent_specific_config.model_dump(exclude_none=True)
                if self.agent.agent_specific_config is not None
                else {}
            ),
        )

    def get_run_metadata(self) -> Dict[str, Any]:
        ret: Dict[str, Any] = {
            "agent": self.agent.get_run_metadata(),
            "llm": self.llm.model_dump(exclude_none=True),
            "dataset": self.dataset.model_dump(exclude_none=True),
            "token_budget": self.token_budget.model_dump(),
            "log_langfuse": self.log_langfuse,
            "use_disk_cache": self.use_disk_cache,
            "debug": self.debug,
            "max_instances": self.max_instances,
        }
        if self.save_dir is not None:
            ret["save_dir"] = self.save_dir
        ret["resume"] = self.resume
        ret["run_name"] = self.run_name
        ret["num_workers"] = self.num_workers
        return ret
