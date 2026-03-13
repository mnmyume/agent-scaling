"""OmegaConf/Hydra resolvers used by run_conf/*.yaml configs.

Hydra evaluates `hydra.run.dir` early (before entering main), so any custom
resolver must be registered at import time of the entrypoint script.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import OmegaConf


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.+-]+")


def _to_container(cfg: Any) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, (dict, list, tuple, str, int, float, bool)):
        return cfg
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return cfg


def _sanitize_segment(text: str) -> str:
    s = (text or "").strip()
    # Avoid path separators creating nested directories.
    s = s.replace("/", "-").replace("\\", "-")
    s = _SAFE_CHARS_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _model_from_override(override: Any, base_model: str) -> str:
    if override is None:
        return base_model
    ov = _to_container(override)
    if isinstance(ov, str):
        return ov or base_model
    if isinstance(ov, dict):
        model = ov.get("model")
        return model or base_model
    return base_model


def _summarize_models(models: Iterable[str]) -> str:
    # Run-length encode consecutive duplicates to keep paths readable.
    out: List[str] = []
    prev: Optional[str] = None
    count = 0
    for m in models:
        if prev is None:
            prev = m
            count = 1
            continue
        if m == prev:
            count += 1
            continue
        seg = _sanitize_segment(prev)
        if count > 1:
            seg = f"{seg}x{count}"
        out.append(seg)
        prev = m
        count = 1
    if prev is not None:
        seg = _sanitize_segment(prev)
        if count > 1:
            seg = f"{seg}x{count}"
        out.append(seg)
    return "+".join(out)


def make_llm_tag(agent: Any, llm: Any) -> str:
    """Build a filesystem-friendly tag representing the *effective* LLMs used.

    This is used for naming experiment output directories so multi-agent runs
    reflect per-role/per-agent model overrides (e.g. `subagent_llms`).
    """
    agent_cfg = _to_container(agent) or {}
    llm_cfg = _to_container(llm) or {}

    base_model = ""
    if isinstance(llm_cfg, dict):
        base_model = str(llm_cfg.get("model") or "")
    base_model = base_model or "unknown"

    name = ""
    if isinstance(agent_cfg, dict):
        name = str(agent_cfg.get("name") or "")

    # Single-agent: directory segment is just the base model.
    if name.startswith("single-agent") or name == "direct-prompt":
        return _sanitize_segment(base_model)

    # Multi-agent: reflect orchestrator/subagent models.
    n_base_agents = 0
    if isinstance(agent_cfg, dict) and agent_cfg.get("n_base_agents") is not None:
        try:
            n_base_agents = int(agent_cfg.get("n_base_agents") or 0)
        except Exception:
            n_base_agents = 0

    orchestrator_model: Optional[str] = None
    if name in {"multi-agent-centralized", "multi-agent-hybrid", "multi-agent-research"}:
        orchestrator_model = _model_from_override(
            (agent_cfg.get("orchestrator_llm") if isinstance(agent_cfg, dict) else None),
            base_model,
        )

    sub_models: List[str] = []
    fallback_sub_model = base_model
    if isinstance(agent_cfg, dict) and agent_cfg.get("subagent_llm") is not None:
        fallback_sub_model = _model_from_override(agent_cfg.get("subagent_llm"), base_model)

    if isinstance(agent_cfg, dict) and agent_cfg.get("subagent_llms") is not None:
        seq = agent_cfg.get("subagent_llms") or []
        models: List[str] = []
        if isinstance(seq, list):
            for ov in seq:
                models.append(_model_from_override(ov, base_model))
        # If the list is shorter than n_base_agents, remaining agents use the fallback subagent model
        # (subagent_llm if set, otherwise base llm).
        if n_base_agents > 0:
            models = models[:n_base_agents]
            if len(models) < n_base_agents:
                models.extend([fallback_sub_model for _ in range(n_base_agents - len(models))])
        sub_models = models
    elif isinstance(agent_cfg, dict) and agent_cfg.get("subagent_llm") is not None:
        if n_base_agents <= 0:
            n_base_agents = 1
        sub_models = [fallback_sub_model for _ in range(n_base_agents)]
    elif n_base_agents > 0:
        sub_models = [base_model for _ in range(n_base_agents)]

    parts: List[str] = []
    if orchestrator_model is not None:
        parts.append(f"orch-{_sanitize_segment(orchestrator_model)}")
    if sub_models:
        parts.append(f"sub-{_summarize_models(sub_models)}")
    if not parts:
        parts.append(_sanitize_segment(base_model))

    tag = "__".join(parts)

    # Guardrail: keep path segments short-ish.
    if len(tag) > 140:
        digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:10]
        tag = tag[:120].rstrip("_-") + f"__{digest}"

    return tag


def register_resolvers() -> None:
    # Idempotent registration.
    if OmegaConf.has_resolver("make_llm_tag"):
        return
    OmegaConf.register_new_resolver("make_llm_tag", make_llm_tag)


# Register on import so Hydra can use it for hydra.run.dir interpolation.
register_resolvers()
