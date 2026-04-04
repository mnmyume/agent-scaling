from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_scaling.utils import read_json, read_yaml


def compute_communication_overhead(
    turns_mas: float, turns_sas: float
) -> Optional[float]:
    """
    Paper metric O = ((T_MAS - T_SAS) / T_SAS) * 100.
    """
    if turns_sas <= 0:
        return None
    return ((turns_mas - turns_sas) / turns_sas) * 100


def compute_message_density(
    total_messages: float, total_turns: float
) -> Optional[float]:
    """
    Paper metric c = messages / turns.
    """
    if total_turns <= 0:
        return None
    return total_messages / total_turns


def compute_coordination_efficiency(
    success_rate: float, turns_mas: float, turns_sas: float
) -> Optional[float]:
    """
    Paper metric E_c = S / (T / T_SAS).
    """
    if turns_mas <= 0 or turns_sas <= 0:
        return None
    return success_rate / (turns_mas / turns_sas)


def compute_success_per_1k_tokens(
    success_count: float, total_tokens: float
) -> Optional[float]:
    """
    Paper metric success / 1K tokens = 1000 * successes / tokens.
    """
    if total_tokens <= 0:
        return None
    return 1000 * success_count / total_tokens


def compute_failure_amplification_proxy(
    success_rate_mas: float, success_rate_sas: float
) -> Optional[float]:
    """
    Honest proxy when exact error-rate labels are unavailable:
    proxy = failure_rate_MAS / failure_rate_SAS.
    """
    failure_rate_sas = 1.0 - success_rate_sas
    if failure_rate_sas <= 0:
        return None
    return (1.0 - success_rate_mas) / failure_rate_sas


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def compute_bow_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Deterministic proxy similarity using bag-of-words cosine similarity.
    """
    counts_a: Dict[str, int] = {}
    counts_b: Dict[str, int] = {}
    for token in _tokenize(text_a):
        counts_a[token] = counts_a.get(token, 0) + 1
    for token in _tokenize(text_b):
        counts_b[token] = counts_b.get(token, 0) + 1
    if not counts_a or not counts_b:
        return 0.0

    vocab = set(counts_a) | set(counts_b)
    dot = sum(counts_a.get(token, 0) * counts_b.get(token, 0) for token in vocab)
    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_text_redundancy_proxy(texts: List[str]) -> Optional[float]:
    """
    Deterministic proxy for paper redundancy R when embedding cosine is unavailable.
    Computes mean pairwise bag-of-words cosine over agent output texts.
    """
    non_empty = [text for text in texts if text and text.strip()]
    if len(non_empty) < 2:
        return None
    similarities = [
        compute_bow_cosine_similarity(text_a, text_b)
        for text_a, text_b in combinations(non_empty, 2)
    ]
    if not similarities:
        return None
    return sum(similarities) / len(similarities)


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "correct"}:
            return True
        if lowered in {"false", "no", "incorrect"}:
            return False
    return None


def _extract_agent_texts(runtime_metrics: Dict[str, Any]) -> List[str]:
    outputs = runtime_metrics.get("agent_output_log", [])
    per_agent: Dict[str, tuple[int, float, str]] = {}
    priority = {
        "final_candidate": 3,
        "finding": 2,
        "final_answer": 2,
        "synthesized_answer": 1,
    }
    for output in outputs:
        agent_id = output.get("agent_id")
        if not agent_id or agent_id in {"lead_agent", "aggregator", "single_agent"}:
            continue
        content = str(output.get("content", "")).strip()
        if not content:
            continue
        score = priority.get(str(output.get("output_type", "")), 0)
        timestamp = float(output.get("timestamp", 0.0) or 0.0)
        prev = per_agent.get(agent_id)
        if prev is None or (score, timestamp) >= (prev[0], prev[1]):
            per_agent[agent_id] = (score, timestamp, content)
    return [value[2] for value in per_agent.values()]


def aggregate_instance_runtime_metrics(
    instance_runtime_metrics: List[Dict[str, Any]],
    *,
    dataset_id: Optional[str] = None,
    architecture: Optional[str] = None,
    model: Optional[str] = None,
    token_budget: Optional[int] = None,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    available = [metrics for metrics in instance_runtime_metrics if metrics]
    summaries = [metrics.get("summary", {}) for metrics in available]
    total_instances = len(available)

    total_turns = sum(float(summary.get("total_turns", 0) or 0) for summary in summaries)
    total_messages = sum(
        float(summary.get("total_messages", 0) or 0) for summary in summaries
    )
    total_tokens = sum(float(summary.get("total_tokens", 0) or 0) for summary in summaries)
    total_llm_calls = sum(
        float(summary.get("total_llm_calls", 0) or 0) for summary in summaries
    )
    total_tool_calls = sum(
        float(summary.get("total_tool_calls", 0) or 0) for summary in summaries
    )
    total_cost_usd = sum(
        float(summary.get("total_cost_usd", 0) or 0) for summary in summaries
    )
    total_execution_time_s = sum(
        float(summary.get("execution_time_s", 0) or 0) for summary in summaries
    )
    success_labels = [
        _coerce_bool(summary.get("task_success"))
        for summary in summaries
        if _coerce_bool(summary.get("task_success")) is not None
    ]
    success_count = sum(1 for label in success_labels if label)
    success_rate = (
        success_count / len(success_labels) if success_labels else None
    )

    redundancy_proxy_values = [
        compute_text_redundancy_proxy(_extract_agent_texts(metrics))
        for metrics in available
    ]
    redundancy_proxy_values = [
        value for value in redundancy_proxy_values if value is not None
    ]

    return {
        "run_dir": run_dir,
        "dataset_id": dataset_id,
        "architecture": architecture,
        "model": model,
        "token_budget": token_budget,
        "instance_count": total_instances,
        "instance_indices": [
            metrics.get("run_metadata", {}).get("instance_idx") for metrics in available
        ],
        "total_turns": total_turns,
        "avg_turns": (total_turns / total_instances) if total_instances else 0.0,
        "total_messages": total_messages,
        "avg_messages": (total_messages / total_instances) if total_instances else 0.0,
        "message_density_c": compute_message_density(total_messages, total_turns),
        "total_tokens": total_tokens,
        "total_llm_calls": total_llm_calls,
        "total_tool_calls": total_tool_calls,
        "total_cost_usd": total_cost_usd,
        "total_execution_time_s": total_execution_time_s,
        "avg_duplicate_work_ratio": (
            sum(float(summary.get("duplicate_work_ratio", 0) or 0) for summary in summaries)
            / total_instances
            if total_instances
            else 0.0
        ),
        "success_count": success_count,
        "success_rate": success_rate,
        "success_per_1k_tokens": (
            compute_success_per_1k_tokens(success_count, total_tokens)
            if success_rate is not None
            else None
        ),
        "redundancy_proxy_bow_cosine": (
            sum(redundancy_proxy_values) / len(redundancy_proxy_values)
            if redundancy_proxy_values
            else None
        ),
    }


@dataclass
class RunArtifact:
    run_dir: str
    dataset_id: str
    architecture: str
    model: str
    token_budget: Optional[int]
    instance_indices: List[int]
    instance_runtime_metrics: List[Dict[str, Any]]
    run_summary: Dict[str, Any]


def _discover_run_dirs(paths: Iterable[str]) -> List[Path]:
    discovered: List[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        candidates = [path] if path.is_dir() else [path.parent]
        if path.is_dir():
            candidates.extend(
                candidate.parent for candidate in path.rglob("run_config.yaml")
            )
        for candidate in candidates:
            if (candidate / "run_config.yaml").exists() and candidate not in seen:
                discovered.append(candidate)
                seen.add(candidate)
    return discovered


def _load_instance_runtime_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    instance_root = run_dir / "instance_runs"
    if not instance_root.exists():
        return metrics
    for instance_dir in sorted(instance_root.iterdir()):
        metrics_path = instance_dir / "runtime_metrics.json"
        if metrics_path.exists():
            metrics.append(read_json(str(metrics_path)))
    return metrics


def _load_run_artifact(run_dir: Path) -> Optional[RunArtifact]:
    resolved_run_dir = run_dir.resolve()
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return None

    config = read_yaml(str(config_path))
    instance_runtime_metrics = _load_instance_runtime_metrics(run_dir)
    run_summary_path = run_dir / "run_runtime_metrics.json"
    run_summary = (
        read_json(str(run_summary_path))
        if run_summary_path.exists()
        else aggregate_instance_runtime_metrics(
            instance_runtime_metrics,
            dataset_id=config["dataset"]["dataset_id"],
            architecture=config["agent"]["name"],
            model=config["llm"]["model"],
            token_budget=config.get("token_budget", {}).get("total_tokens_per_instance"),
            run_dir=str(resolved_run_dir),
        )
    )
    run_summary["run_dir"] = str(resolved_run_dir)
    run_summary["dataset_id"] = config["dataset"]["dataset_id"]
    run_summary["architecture"] = config["agent"]["name"]
    run_summary["model"] = config["llm"]["model"]
    run_summary["token_budget"] = config.get("token_budget", {}).get(
        "total_tokens_per_instance"
    )
    instance_indices = [
        int(metrics.get("run_metadata", {}).get("instance_idx"))
        for metrics in instance_runtime_metrics
        if metrics.get("run_metadata", {}).get("instance_idx") is not None
    ]
    return RunArtifact(
        run_dir=str(resolved_run_dir),
        dataset_id=config["dataset"]["dataset_id"],
        architecture=config["agent"]["name"],
        model=config["llm"]["model"],
        token_budget=config.get("token_budget", {}).get("total_tokens_per_instance"),
        instance_indices=sorted(instance_indices),
        instance_runtime_metrics=instance_runtime_metrics,
        run_summary=run_summary,
    )


def _has_completed_instances(artifact: RunArtifact) -> bool:
    instance_count = int(artifact.run_summary.get("instance_count") or 0)
    return instance_count > 0 and len(artifact.instance_indices) > 0


def _group_sort_key(item: tuple[tuple[Any, ...], Dict[str, RunArtifact]]) -> tuple[Any, ...]:
    dataset_id, model, token_budget, instance_indices = item[0]
    return (
        str(dataset_id),
        str(model),
        -1 if token_budget is None else int(token_budget),
        tuple(int(idx) for idx in instance_indices),
    )


def aggregate_experiment_metrics(paths: List[str]) -> Dict[str, Any]:
    run_dirs = _discover_run_dirs(paths)
    run_artifacts = [
        artifact for artifact in (_load_run_artifact(run_dir) for run_dir in run_dirs) if artifact
    ]

    runs = [artifact.run_summary for artifact in run_artifacts]
    grouped: Dict[tuple[Any, ...], Dict[str, RunArtifact]] = {}
    for artifact in run_artifacts:
        key = (
            artifact.dataset_id,
            artifact.model,
            artifact.token_budget,
            tuple(artifact.instance_indices),
        )
        grouped.setdefault(key, {})[artifact.architecture] = artifact

    paired_results: List[Dict[str, Any]] = []
    for key, artifacts_by_architecture in sorted(grouped.items(), key=_group_sort_key):
        baseline = artifacts_by_architecture.get("single-agent")
        if baseline is None or not _has_completed_instances(baseline):
            continue

        for architecture, artifact in artifacts_by_architecture.items():
            if architecture == "single-agent" or not _has_completed_instances(artifact):
                continue

            success_rate_mas = artifact.run_summary.get("success_rate")
            success_rate_sas = baseline.run_summary.get("success_rate")
            paired_results.append(
                {
                    "dataset_id": artifact.dataset_id,
                    "model": artifact.model,
                    "architecture": architecture,
                    "token_budget": artifact.token_budget,
                    "instance_indices": artifact.instance_indices,
                    "paired_instance_count": len(artifact.instance_indices),
                    "run_dir": artifact.run_dir,
                    "baseline_run_dir": baseline.run_dir,
                    "turns_T": artifact.run_summary.get("avg_turns"),
                    "baseline_turns_T_sas": baseline.run_summary.get("avg_turns"),
                    "success_rate_S": success_rate_mas,
                    "baseline_success_rate_sas": success_rate_sas,
                    "communication_overhead_percent_O": compute_communication_overhead(
                        float(artifact.run_summary.get("avg_turns") or 0),
                        float(baseline.run_summary.get("avg_turns") or 0),
                    ),
                    "message_density_c": compute_message_density(
                        float(artifact.run_summary.get("total_messages") or 0),
                        float(artifact.run_summary.get("total_turns") or 0),
                    ),
                    "coordination_efficiency_Ec": (
                        compute_coordination_efficiency(
                            float(success_rate_mas),
                            float(artifact.run_summary.get("avg_turns") or 0),
                            float(baseline.run_summary.get("avg_turns") or 0),
                        )
                        if success_rate_mas is not None
                        else None
                    ),
                    "success_per_1k_tokens": artifact.run_summary.get("success_per_1k_tokens"),
                    "redundancy_R": None,
                    "redundancy_proxy_bow_cosine": artifact.run_summary.get(
                        "redundancy_proxy_bow_cosine"
                    ),
                    "error_amplification_A_e": None,
                    "failure_amplification_proxy": (
                        compute_failure_amplification_proxy(
                            float(success_rate_mas),
                            float(success_rate_sas),
                        )
                        if success_rate_mas is not None and success_rate_sas is not None
                        else None
                    ),
                    "support": {
                        "communication_overhead_percent_O": "exact",
                        "message_density_c": "exact",
                        "coordination_efficiency_Ec": "exact",
                        "success_per_1k_tokens": "exact",
                        "redundancy_R": "deferred",
                        "redundancy_proxy_bow_cosine": "proxy",
                        "error_amplification_A_e": "deferred",
                        "failure_amplification_proxy": "proxy",
                    },
                }
            )

    return {
        "runs": runs,
        "paired_metrics": paired_results,
    }
