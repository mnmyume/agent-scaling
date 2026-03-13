from typing import Dict, Type, TypeVar

import importlib

from .base import AgentEnvironment

T = TypeVar("T", bound=AgentEnvironment)

# Simple registry using a dictionary
_env_registry: Dict[str, Type[AgentEnvironment]] = {}
_ENV_IMPORTS: Dict[str, str] = {
    "basic": "agent_scaling.env.basic",
    "plancraft": "agent_scaling.env.plancraft",
    "web-search": "agent_scaling.env.web_search",
    "browsecomp-plus": "agent_scaling.env.browsecomp",
    "workbench": "agent_scaling.env.workbench",
}


def _ensure_env_registered(name: str) -> None:
    if name in _env_registry:
        return
    module_path = _ENV_IMPORTS.get(name)
    if module_path is None:
        return
    try:
        importlib.import_module(module_path)
    except Exception as exc:
        raise ValueError(
            f"Environment '{name}' failed to import from {module_path}: {exc}"
        ) from exc


def register_env(name: str):
    """Decorator to register an environment class."""

    def decorator(cls: Type[T]) -> Type[T]:
        _env_registry[name] = cls
        return cls

    return decorator


def list_envs() -> list[str]:
    """List all registered environment names."""
    return sorted(set(_env_registry.keys()) | set(_ENV_IMPORTS.keys()))


def is_env_registered(name: str) -> bool:
    """Check if an environment is registered."""
    _ensure_env_registered(name)
    return name in _env_registry


def get_env(name: str, **kwargs) -> AgentEnvironment:
    """Create an environment instance using the factory pattern."""
    _ensure_env_registered(name)
    if name not in _env_registry:
        raise ValueError(f"Environment '{name}' not found in registry")
    return _env_registry[name](**kwargs)


def get_env_cls(name: str) -> Type[AgentEnvironment]:
    """Get an environment class by name."""
    _ensure_env_registered(name)
    if name not in _env_registry:
        raise ValueError(f"Environment '{name}' not found in registry")
    return _env_registry[name]
