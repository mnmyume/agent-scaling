#!/usr/bin/env python3
"""Step 1: Generate heterogeneous MAS configurations.

Produces train/test config sets for a single architecture, with models
randomly assigned to roles from a model pool file.

Usage:
    uv run python surrogate_pipeline/config_generator.py \
        --run_name my_experiment \
        --architecture multi-agent-centralized \
        --model_pool_file surrogate_pipeline/model_pool.yaml \
        --n_base_agents 3 \
        --num_train_configs 20 \
        --num_test_configs 5
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from surrogate_pipeline.utils import (
    ARCH_ROLES,
    CONFIGS_DIR,
    get_run_dir,
    load_model_pool,
    save_json,
    setup_logging,
)


def _config_hash(config: dict) -> str:
    """Deterministic hash for deduplication."""
    canonical = json.dumps(config["role_models"], sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()


def generate_config(
    architecture: str,
    model_pool: list[str],
    n_base_agents: int,
) -> dict:
    """Generate a single random heterogeneous config."""
    role_fn = ARCH_ROLES.get(architecture)
    if role_fn is None:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {list(ARCH_ROLES.keys())}"
        )
    roles = role_fn(n_base_agents)
    role_models = {role: random.choice(model_pool) for role in roles}
    return {
        "architecture": architecture,
        "n_base_agents": n_base_agents,
        "role_models": role_models,
    }


def generate_configs(
    architecture: str,
    model_pool: list[str],
    n_base_agents: int,
    num_configs: int,
    exclude_hashes: set[str] | None = None,
    max_attempts: int = 10000,
) -> list[dict]:
    """Generate unique configs, avoiding duplicates."""
    if exclude_hashes is None:
        exclude_hashes = set()
    configs = []
    seen = set(exclude_hashes)
    attempts = 0
    while len(configs) < num_configs and attempts < max_attempts:
        cfg = generate_config(architecture, model_pool, n_base_agents)
        h = _config_hash(cfg)
        if h not in seen:
            cfg["config_id"] = f"config_{len(seen):04d}"
            configs.append(cfg)
            seen.add(h)
        attempts += 1
    if len(configs) < num_configs:
        raise RuntimeError(
            f"Could only generate {len(configs)}/{num_configs} unique configs "
            f"after {max_attempts} attempts. Try a larger model pool or fewer configs."
        )
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Generate heterogeneous MAS configurations"
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Name for this pipeline run (auto-generated if omitted)",
    )
    parser.add_argument(
        "--architecture", type=str, required=True,
        choices=list(ARCH_ROLES.keys()),
        help="MAS architecture type",
    )
    parser.add_argument(
        "--model_pool_file", type=str,
        default="surrogate_pipeline/model_pool.yaml",
        help="Path to YAML file listing available models",
    )
    parser.add_argument(
        "--n_base_agents", type=int, default=3,
        help="Number of base agents (default: 3)",
    )
    parser.add_argument(
        "--num_train_configs", type=int, default=20,
        help="Number of training configs to generate",
    )
    parser.add_argument(
        "--num_test_configs", type=int, default=5,
        help="Number of test configs to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Setup
    run_dir = get_run_dir(args.run_name)
    logger = setup_logging(run_dir)
    configs_dir = run_dir / CONFIGS_DIR
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Load model pool
    model_pool = load_model_pool(args.model_pool_file)
    logger.info(f"Loaded {len(model_pool)} models from {args.model_pool_file}")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Agents per config: {args.n_base_agents}")

    # Generate train configs
    logger.info(f"Generating {args.num_train_configs} training configs...")
    train_configs = generate_configs(
        architecture=args.architecture,
        model_pool=model_pool,
        n_base_agents=args.n_base_agents,
        num_configs=args.num_train_configs,
    )

    # Generate test configs (no overlap with train)
    train_hashes = {_config_hash(c) for c in train_configs}
    logger.info(f"Generating {args.num_test_configs} test configs...")
    test_configs = generate_configs(
        architecture=args.architecture,
        model_pool=model_pool,
        n_base_agents=args.n_base_agents,
        num_configs=args.num_test_configs,
        exclude_hashes=train_hashes,
    )

    # Save
    save_json(train_configs, configs_dir / "train_configs.json")
    save_json(test_configs, configs_dir / "test_configs.json")

    metadata = {
        "architecture": args.architecture,
        "model_pool": model_pool,
        "model_pool_file": args.model_pool_file,
        "n_base_agents": args.n_base_agents,
        "num_train_configs": len(train_configs),
        "num_test_configs": len(test_configs),
        "seed": args.seed,
    }
    save_json(metadata, configs_dir / "generation_metadata.json")

    logger.info(f"Saved {len(train_configs)} train + {len(test_configs)} test configs")
    logger.info(f"Output: {configs_dir}")


if __name__ == "__main__":
    main()
