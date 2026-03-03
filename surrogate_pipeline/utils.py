"""Shared utilities for the surrogate pipeline."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

# Base output directory
SURROGATE_OUTPUTS_DIR = Path("surrogate_outputs")

# Sub-directory names
CONFIGS_DIR = "configs"
BENCHMARKS_DIR = "benchmarks"
RAW_RUNS_DIR = "raw_runs"
MODEL_DIR = "model"
EVALUATION_DIR = "evaluation"


def get_run_dir(run_name: str | None = None) -> Path:
    """Get or create the output directory for a pipeline run.

    If run_name is None, auto-generates a timestamped name.
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = SURROGATE_OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    """Configure a logger that writes to both console and file."""
    logger = logging.getLogger("surrogate_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(run_dir / "pipeline_log.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_model_pool(path: str) -> list[str]:
    """Load model pool from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    models = data.get("models", [])
    if not models:
        raise ValueError(f"No models found in {path}")
    return models


def save_json(data: dict | list, path: Path, indent: int = 2) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Path) -> dict | list:
    """Load data from JSON."""
    with open(path) as f:
        return json.load(f)


# Architecture → role structure mapping
ARCH_ROLES = {
    "multi-agent-centralized": lambda n: ["orchestrator"] + [f"worker_{i}" for i in range(n)],
    "multi-agent-decentralized": lambda n: [f"agent_{i}" for i in range(n)],
    "multi-agent-hybrid": lambda n: ["orchestrator"] + [f"worker_{i}" for i in range(n)],
    "multi-agent-independent": lambda n: [f"agent_{i}" for i in range(n)],
}
