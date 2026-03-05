#!/usr/bin/env python3
"""Step 2: Run benchmarks for generated configurations.

Reads configs from surrogate_outputs/<run_name>/configs/ and executes
real benchmarks via `uv run python run_scripts/run_experiment.py`.

Usage:
    uv run python surrogate_pipeline/run_benchmarks.py \
        --run_name my_experiment \
        --dataset plancraft-test \
        --split both \
        --max_instances 5 \
        --num_workers 1
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from surrogate_pipeline.utils import (
    BENCHMARKS_DIR,
    CONFIGS_DIR,
    RAW_RUNS_DIR,
    get_run_dir,
    load_json,
    save_json,
    setup_logging,
)


def build_hydra_cmd(
    config: dict,
    dataset: str,
    max_instances: int | None,
    num_workers: int,
    output_dir: str,
) -> list[str]:
    """Build the uv run command for a single benchmark config."""
    arch = config["architecture"]
    role_models = config["role_models"]

    # Use the first model in role_models as the global model (orchestrator or agent_0)
    primary_role = next(iter(role_models))
    global_model = role_models[primary_role]

    cmd = [
        "uv", "run", "python", "run_scripts/run_experiment.py",
        f"agent={arch}",
        f"dataset={dataset}",
        f"llm.model={global_model}",
        f"num_workers={num_workers}",
        "log_langfuse=false",
        "use_disk_cache=false",
        f"hydra.run.dir={output_dir}",
    ]

    if max_instances is not None:
        cmd.append(f"max_instances={max_instances}")

    # Pass per-role model overrides via Hydra CLI
    for role, model in role_models.items():
        cmd.append(f"+llm.role_models.{role}={model}")

    if "n_base_agents" in config:
        cmd.append(f"agent.n_base_agents={config['n_base_agents']}")

    return cmd


def _stream_pipe(pipe, log_file, prefix, logger, is_stderr=False):
    """Read from a pipe line-by-line and write to log file + selective terminal output."""
    with open(log_file, "w") as f:
        for line in iter(pipe.readline, ""):
            f.write(line)
            f.flush()
            stripped = line.strip()
            if not stripped:
                continue
            if is_stderr:
                # Show tqdm progress bars and errors/warnings
                if any(kw in stripped for kw in ["%|", "it/s", "Error", "error", "Traceback", "Exception"]):
                    logger.info(f"{prefix} {stripped}")
            else:
                # Show instance-level progress from exp_runner
                if any(kw in stripped for kw in ["Instance", "instance", "avg_success", "Success", "success", "metrics", "Completed", "completed"]):
                    logger.info(f"{prefix} {stripped}")
    pipe.close()


def run_single_benchmark(
    config: dict,
    dataset: str,
    max_instances: int | None,
    num_workers: int,
    raw_runs_dir: Path,
    logger,
) -> dict | None:
    """Execute a single benchmark and return results."""
    config_id = config["config_id"]
    output_dir = raw_runs_dir / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    metrics_file = output_dir / "dataset_eval_metrics.json"
    if metrics_file.exists():
        metrics = load_json(metrics_file)
        logger.info(f"[{config_id}] ⏭️  Skipping (already completed): avg_success={metrics.get('avg_success', 'N/A')}")
        return {
            "config_id": config_id,
            "config": config,
            "dataset": dataset,
            "metrics": metrics,
        }

    cmd = build_hydra_cmd(
        config=config,
        dataset=dataset,
        max_instances=max_instances,
        num_workers=num_workers,
        output_dir=str(output_dir),
    )

    logger.info(f"[{config_id}] Running: {' '.join(cmd)}")

    timeout = 10800  # 3 hour timeout per config

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Stream stdout and stderr in background threads
        stdout_log = output_dir / "stdout.log"
        stderr_log = output_dir / "stderr.log"

        stdout_thread = threading.Thread(
            target=_stream_pipe,
            args=(proc.stdout, stdout_log, f"[{config_id}]", logger),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_pipe,
            args=(proc.stderr, stderr_log, f"[{config_id}][stderr]", logger, True),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        # Wait for process with timeout
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"[{config_id}] Timed out after {timeout}s, killing process")
            proc.kill()
            proc.wait()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            return None

        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)

        if returncode != 0:
            logger.error(f"[{config_id}] Failed (exit={returncode})")
            return None

        # Parse metrics
        metrics_file = output_dir / "dataset_eval_metrics.json"
        if metrics_file.exists():
            metrics = load_json(metrics_file)
            logger.info(f"[{config_id}] ✅ Success: avg_success={metrics.get('avg_success', 'N/A')}")
            return {
                "config_id": config_id,
                "config": config,
                "dataset": dataset,
                "metrics": metrics,
            }
        else:
            logger.warning(f"[{config_id}] No metrics file found")
            return None

    except Exception as e:
        logger.error(f"[{config_id}] Exception: {e}")
        return None


def run_split(
    split_name: str,
    configs: list[dict],
    dataset: str,
    max_instances: int | None,
    num_workers: int,
    benchmarks_dir: Path,
    logger,
) -> list[dict]:
    """Run all configs for a given split."""
    raw_runs_dir = benchmarks_dir / RAW_RUNS_DIR / split_name
    raw_runs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, config in enumerate(configs):
        logger.info(f"--- [{split_name}] Config {i+1}/{len(configs)} ---")
        result = run_single_benchmark(
            config=config,
            dataset=dataset,
            max_instances=max_instances,
            num_workers=num_workers,
            raw_runs_dir=raw_runs_dir,
            logger=logger,
        )
        if result is not None:
            results.append(result)

    output_file = benchmarks_dir / f"{split_name}_results.json"
    save_json(results, output_file)
    logger.info(f"[{split_name}] Completed {len(results)}/{len(configs)} configs")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for generated configs")
    parser.add_argument(
        "--run_name", type=str, required=True,
        help="Pipeline run name (must match config_generator output)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset/benchmark to evaluate on (e.g. plancraft-test)",
    )
    parser.add_argument(
        "--split", type=str, default="both",
        choices=["train", "test", "both"],
        help="Which config split to run (default: both)",
    )
    parser.add_argument(
        "--max_instances", type=int, default=None,
        help="Max instances per benchmark run",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Parallel workers per benchmark run",
    )
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_name)
    logger = setup_logging(run_dir)
    configs_dir = run_dir / CONFIGS_DIR
    benchmarks_dir = run_dir / BENCHMARKS_DIR
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")

    splits_to_run = []
    if args.split in ("train", "both"):
        splits_to_run.append("train")
    if args.split in ("test", "both"):
        splits_to_run.append("test")

    for split_name in splits_to_run:
        config_file = configs_dir / f"{split_name}_configs.json"
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            sys.exit(1)

        configs = load_json(config_file)
        logger.info(f"Loaded {len(configs)} {split_name} configs from {config_file}")

        run_split(
            split_name=split_name,
            configs=configs,
            dataset=args.dataset,
            max_instances=args.max_instances,
            num_workers=args.num_workers,
            benchmarks_dir=benchmarks_dir,
            logger=logger,
        )

    logger.info("All benchmark runs complete.")


if __name__ == "__main__":
    main()
