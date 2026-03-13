from __future__ import annotations

import math
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agent_scaling.datasets.base import Dataset, DatasetInstanceOutput
from agent_scaling.llm import ChatLiteLLMLC
from agent_scaling.metrics.artifacts import MetricsArtifacts
from agent_scaling.metrics.token_overlap import compute_token_overlap_metrics
from agent_scaling.metrics.similarity import (
    pairwise_cosine_similarities,
    pairwise_embedding_cosine_similarities,
)
from agent_scaling.utils import read_json, read_yaml


class PaperMetricsConfig(BaseModel):
    enable: bool = True
    baseline_run_dir: Optional[str] = None
    baseline_auto_discovery: bool = True
    baseline_runs_root: Optional[str] = None
    baseline_selection_strategy: str = "best_success"
    baseline_require_same_max_instances: bool = True
    similarity_mode: str = "bert_score"
    contradiction_threshold: float = 0.3
    strict_paper_alignment: bool = True
    redundancy_mode: str = "embedding_cosine"
    redundancy_embedding_model: Optional[str] = None
    redundancy_embedding_batch_size: int = 16
    compute_information_gain: bool = False
    info_gain_samples: int = 10
    info_gain_temperature: float = 0.7
    compute_error_taxonomy: bool = False
    domain_complexity_runs_root: Optional[str] = None


SUCCESS_KEYS = ["success", "is_correct", "correct", "task_success"]
SUCCESS_RATE_KEYS = [
    "avg_success",
    "avg_accuracy",
    "accuracy",
    "success_rate",
]


def _safe_div(n: float, d: float) -> Optional[float]:
    if d == 0:
        return None
    return n / d


def extract_success_indicator(metrics: Dict[str, Any]) -> Optional[float]:
    for key in SUCCESS_KEYS:
        if key in metrics:
            return float(metrics[key])
    return None


def extract_success_rate(metrics: Dict[str, Any]) -> Optional[float]:
    for key in SUCCESS_RATE_KEYS:
        if key in metrics:
            return float(metrics[key])
    if "paper_metrics" in metrics and "success_rate" in metrics["paper_metrics"]:
        return float(metrics["paper_metrics"]["success_rate"])
    return None


def compute_redundancy(
    texts: List[str], config: PaperMetricsConfig
) -> Optional[float]:
    if len(texts) < 2:
        return 0.0

    if config.redundancy_mode == "embedding_cosine":
        if not config.redundancy_embedding_model:
            if config.strict_paper_alignment:
                raise ValueError(
                    "Strict paper alignment requires `metrics.redundancy_embedding_model` "
                    "for embedding-based redundancy."
                )
            sims = pairwise_cosine_similarities(texts)
        else:
            sims = pairwise_embedding_cosine_similarities(
                texts,
                embedding_model=config.redundancy_embedding_model,
                batch_size=config.redundancy_embedding_batch_size,
            )
    elif config.redundancy_mode == "lexical_cosine":
        if config.strict_paper_alignment:
            raise ValueError(
                "Strict paper alignment does not allow lexical redundancy mode. "
                "Set `metrics.redundancy_mode=embedding_cosine`."
            )
        sims = pairwise_cosine_similarities(texts)
    else:
        raise ValueError(
            "Unsupported `redundancy_mode`. Use `embedding_cosine` or `lexical_cosine`."
        )

    if not sims:
        return None
    return sum(sims) / len(sims)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _sum(values: Iterable[Optional[float]]) -> float:
    return float(sum(v for v in values if v is not None))


def load_baseline_metrics(baseline_run_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if not baseline_run_dir:
        return None
    metrics_path = os.path.join(baseline_run_dir, "dataset_eval_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    return read_json(metrics_path)


def _default_runs_root() -> str:
    # Resolve to repository-level exp_outputs irrespective of Hydra cwd.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exp_outputs"))


def _extract_run_timestamp(run_dir: str) -> datetime:
    normalized = run_dir.replace("\\", "/")
    parts = normalized.split("/")
    # Expected tail shape: .../<YYYY-MM-DD>/<HH-MM-SS>
    for idx in range(len(parts) - 2, -1, -1):
        date_part = parts[idx]
        time_part = parts[idx + 1] if idx + 1 < len(parts) else ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_part) and re.fullmatch(
            r"\d{2}-\d{2}-\d{2}", time_part
        ):
            try:
                return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H-%M-%S")
            except ValueError:
                break
    return datetime.fromtimestamp(os.path.getmtime(run_dir))


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def discover_single_agent_baseline_run(
    dataset_id: str,
    runs_root: Optional[str],
    selection_strategy: str = "best_success",
    allowed_models: Optional[List[str]] = None,
    require_same_max_instances: bool = False,
    required_max_instances: Optional[int] = None,
) -> Optional[str]:
    root_dir = runs_root or _default_runs_root()
    if not os.path.isdir(root_dir):
        return None

    strategy = selection_strategy.strip().lower()
    if strategy not in {"best_success", "latest"}:
        raise ValueError(
            "Unsupported `baseline_selection_strategy`. Use `best_success` or `latest`."
        )
    allowed_set = {m.strip() for m in (allowed_models or []) if m and m.strip()}

    candidates: List[Dict[str, Any]] = []
    for root, _, files in os.walk(root_dir):
        if "dataset_eval_metrics.json" not in files or "run_config.yaml" not in files:
            continue
        run_config_path = os.path.join(root, "run_config.yaml")
        try:
            run_config = read_yaml(run_config_path)
        except Exception:
            continue

        if run_config.get("dataset", {}).get("dataset_id") != dataset_id:
            continue
        if require_same_max_instances:
            candidate_max_instances = _to_optional_int(run_config.get("max_instances"))
            if candidate_max_instances != required_max_instances:
                continue
        agent_name = run_config.get("agent", {}).get("name", "")
        if not agent_name.startswith("single-agent"):
            continue
        model_name = str(run_config.get("llm", {}).get("model", "")).strip()
        if allowed_set and model_name not in allowed_set:
            continue

        metrics = read_json(os.path.join(root, "dataset_eval_metrics.json"))
        success_rate = extract_success_rate(metrics)
        if success_rate is None:
            continue

        candidates.append(
            {
                "run_dir": root,
                "success_rate": success_rate,
                "timestamp": _extract_run_timestamp(root),
                "model": model_name,
            }
        )

    if not candidates:
        return None

    if strategy == "latest":
        selected = max(candidates, key=lambda c: c["timestamp"])
    else:
        selected = max(candidates, key=lambda c: (c["success_rate"], c["timestamp"]))
    return str(selected["run_dir"])


def resolve_baseline_metrics(
    config: PaperMetricsConfig,
    dataset_id: Optional[str],
    default_runs_root: Optional[str] = None,
    allowed_models: Optional[List[str]] = None,
    current_max_instances: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    baseline_run_dir = config.baseline_run_dir
    source = "manual"
    if (
        not baseline_run_dir
        and config.baseline_auto_discovery
        and dataset_id
    ):
        baseline_run_dir = discover_single_agent_baseline_run(
            dataset_id=dataset_id,
            runs_root=config.baseline_runs_root or default_runs_root,
            selection_strategy=config.baseline_selection_strategy,
            allowed_models=allowed_models,
            require_same_max_instances=config.baseline_require_same_max_instances,
            required_max_instances=current_max_instances,
        )
        source = "auto_discovery"

    baseline = load_baseline_metrics(baseline_run_dir)
    if baseline is not None:
        baseline["_run_dir"] = baseline_run_dir
        baseline["_source"] = source
    return baseline


def compute_domain_complexity_from_runs(
    runs_root: Optional[str], dataset_id: str
) -> Optional[Dict[str, float]]:
    if not runs_root:
        return None
    performances: List[float] = []
    sas_performances: List[float] = []
    for root, _, files in os.walk(runs_root):
        if "dataset_eval_metrics.json" not in files:
            continue
        metrics_path = os.path.join(root, "dataset_eval_metrics.json")
        run_config_path = os.path.join(root, "run_config.yaml")
        if not os.path.exists(run_config_path):
            continue
        try:
            run_config = read_yaml(run_config_path)
            if run_config.get("dataset", {}).get("dataset_id") != dataset_id:
                continue
        except Exception:
            continue
        metrics = read_json(metrics_path)
        perf = extract_success_rate(metrics)
        if perf is None:
            continue
        performances.append(perf)
        agent_name = run_config.get("agent", {}).get("name")
        if agent_name and agent_name.startswith("single-agent"):
            sas_performances.append(perf)

    if not performances:
        return None

    p_max = max(performances)
    mean_perf = sum(performances) / len(performances)
    variance = sum((p - mean_perf) ** 2 for p in performances) / len(performances)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_perf if mean_perf > 0 else 0.0
    if sas_performances:
        p_best = max(sas_performances)
    else:
        p_best = p_max
    complexity = (1 - p_max + cv + 1 - p_best) / 3.0
    return {
        "performance_ceiling": 1 - p_max,
        "coefficient_of_variation": cv,
        "best_model_baseline": 1 - p_best,
        "domain_complexity": complexity,
    }


def _build_task_context(dataset: Dataset, output: DatasetInstanceOutput) -> str:
    instance = output.data_instance
    if dataset.task_shared_prompts is not None:
        try:
            prompts = dataset.task_shared_prompts.get_prompt_templates_for_instance(
                instance.get_prompt_info()
            )
            return prompts.get("task_instance", str(instance.get_prompt_info()))
        except Exception:
            return str(instance.get_prompt_info())
    return str(instance.get_prompt_info())


def _parse_probability(text: str) -> Optional[float]:
    match = re.search(r"([01](?:\\.\\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _estimate_success_probability(
    llm: ChatLiteLLMLC,
    task_context: str,
    state_text: str,
    samples: int,
    temperature: float,
) -> Optional[float]:
    if samples <= 0:
        return None
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator. Given the task context and a partial or "
                "final state, estimate the probability that the state leads to a correct "
                "final answer. Return only a number between 0 and 1."
            ),
        },
        {
            "role": "user",
            "content": f"Task:\\n{task_context}\\n\\nState:\\n{state_text}\\n\\nProbability:",
        },
    ]
    probs: List[float] = []
    for _ in range(samples):
        response = llm.invoke(messages, temperature=temperature)
        prob = _parse_probability(response.text())
        if prob is not None:
            probs.append(prob)
    if not probs:
        return None
    return sum(probs) / len(probs)


def compute_information_gain(
    llm: ChatLiteLLMLC,
    dataset: Dataset,
    output: DatasetInstanceOutput,
    artifacts: MetricsArtifacts,
    config: PaperMetricsConfig,
) -> Optional[float]:
    if artifacts.pre_state_text is None or artifacts.post_state_text is None:
        return None
    task_context = _build_task_context(dataset, output)
    p_pre = _estimate_success_probability(
        llm,
        task_context,
        artifacts.pre_state_text,
        config.info_gain_samples,
        config.info_gain_temperature,
    )
    p_post = _estimate_success_probability(
        llm,
        task_context,
        artifacts.post_state_text,
        config.info_gain_samples,
        config.info_gain_temperature,
    )
    if p_pre is None or p_post is None:
        return None
    var_pre = p_pre * (1 - p_pre)
    var_post = p_post * (1 - p_post)
    if var_pre <= 0 or var_post <= 0:
        return None
    return 0.5 * math.log(var_pre / var_post)


def classify_error_taxonomy(
    llm: ChatLiteLLMLC,
    dataset: Dataset,
    output: DatasetInstanceOutput,
) -> Optional[str]:
    instance = output.data_instance
    expected = getattr(instance, "expected_output", None)
    task_context = _build_task_context(dataset, output)
    prompt = [
        {
            "role": "system",
            "content": (
                "Classify the failure into exactly one category from: "
                "logical_contradiction, numerical_drift, context_omission, coordination_failure. "
                "Return only the category label."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\\n{task_context}\\n\\nExpected:\\n{expected}\\n\\n"
                f"Model Output:\\n{output.agent_output}\\n\\nCategory:"
            ),
        },
    ]
    response = llm.invoke(prompt)
    label = response.text().strip().lower()
    for candidate in [
        "logical_contradiction",
        "numerical_drift",
        "context_omission",
        "coordination_failure",
    ]:
        if candidate in label:
            return candidate
    return None


def compute_instance_paper_metrics(
    instance_metrics: Dict[str, Any],
    output: DatasetInstanceOutput,
    dataset: Dataset,
    config: PaperMetricsConfig,
) -> Dict[str, Any]:
    success = extract_success_indicator(instance_metrics)
    error_rate = None if success is None else 1.0 - success

    artifacts: Optional[MetricsArtifacts] = getattr(output, "metrics_artifacts", None)
    reasoning_turns = None
    inter_agent_messages = None
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    total_cost = None
    total_provider_cost = None
    total_openrouter_cost = None
    llm_call_count = None
    total_response_time_ms = None
    avg_response_time_ms = None
    redundancy = None
    token_overlap: Optional[Dict[str, float]] = None
    info_gain = None
    error_category = None

    if artifacts:
        reasoning_turns = artifacts.reasoning_turns or artifacts.tool_calls
        inter_agent_messages = artifacts.inter_agent_messages
        prompt_tokens = artifacts.total_input_tokens
        completion_tokens = artifacts.total_output_tokens
        total_tokens = artifacts.total_tokens
        total_cost = artifacts.total_cost_usd
        total_provider_cost = artifacts.total_provider_cost_usd
        total_openrouter_cost = artifacts.total_openrouter_cost_usd
        llm_call_count = artifacts.llm_call_count
        total_response_time_ms = artifacts.total_latency_ms
        avg_response_time_ms = artifacts.avg_latency_ms
        if artifacts.subagent_outputs:
            redundancy = compute_redundancy(artifacts.subagent_outputs, config)
            token_overlap = compute_token_overlap_metrics(
                artifacts.subagent_outputs,
                contradiction_threshold=config.contradiction_threshold,
                similarity_mode=config.similarity_mode,
                strict_paper_alignment=config.strict_paper_alignment,
            )
        else:
            redundancy = 0.0
        if (
            config.compute_information_gain
            and dataset.eval_llm is not None
        ):
            info_gain = compute_information_gain(
                dataset.eval_llm,
                dataset,
                output,
                artifacts,
                config,
            )
    if (
        config.compute_error_taxonomy
        and dataset.eval_llm is not None
        and success is not None
        and success < 1.0
    ):
        error_category = classify_error_taxonomy(dataset.eval_llm, dataset, output)

    message_density = None
    if reasoning_turns is not None and reasoning_turns > 0:
        message_density = (
            float(inter_agent_messages or 0) / float(reasoning_turns)
        )

    success_per_1k_tokens = None
    if success is not None and total_tokens and total_tokens > 0:
        success_per_1k_tokens = success / (total_tokens / 1000.0)

    success_per_usd = None
    if success is not None and total_cost and total_cost > 0:
        success_per_usd = success / total_cost
    success_per_provider_usd = None
    if success is not None and total_provider_cost and total_provider_cost > 0:
        success_per_provider_usd = success / total_provider_cost
    success_per_openrouter_usd = None
    if success is not None and total_openrouter_cost and total_openrouter_cost > 0:
        success_per_openrouter_usd = success / total_openrouter_cost

    return {
        "success": success,
        "error_rate": error_rate,
        "reasoning_turns": reasoning_turns,
        "inter_agent_messages": inter_agent_messages,
        "message_density": message_density,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "llm_call_count": llm_call_count,
        "total_response_time_ms": total_response_time_ms,
        "avg_response_time_ms": avg_response_time_ms,
        "total_cost_usd": total_cost,
        "total_provider_cost_usd": total_provider_cost,
        "total_openrouter_cost_usd": total_openrouter_cost,
        "success_per_1k_tokens": success_per_1k_tokens,
        "success_per_usd": success_per_usd,
        "success_per_provider_usd": success_per_provider_usd,
        "success_per_openrouter_usd": success_per_openrouter_usd,
        "redundancy": redundancy,
        "token_overlap": token_overlap,
        "information_gain": info_gain,
        "error_category": error_category,
    }


def aggregate_paper_metrics(
    instance_metrics: List[Dict[str, Any]],
    config: PaperMetricsConfig,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    dataset_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    runs_root: Optional[str] = None,
) -> Dict[str, Any]:
    success_values = [m.get("success") for m in instance_metrics]
    error_rates = [m.get("error_rate") for m in instance_metrics]
    reasoning_turns = [m.get("reasoning_turns") for m in instance_metrics]
    inter_agent_messages = [m.get("inter_agent_messages") for m in instance_metrics]
    message_density = [m.get("message_density") for m in instance_metrics]
    prompt_tokens = [m.get("prompt_tokens") for m in instance_metrics]
    completion_tokens = [m.get("completion_tokens") for m in instance_metrics]
    total_tokens = [m.get("total_tokens") for m in instance_metrics]
    llm_call_counts = [m.get("llm_call_count") for m in instance_metrics]
    total_response_times = [m.get("total_response_time_ms") for m in instance_metrics]
    total_cost = [m.get("total_cost_usd") for m in instance_metrics]
    total_provider_cost = [m.get("total_provider_cost_usd") for m in instance_metrics]
    total_openrouter_cost = [m.get("total_openrouter_cost_usd") for m in instance_metrics]
    redundancy = [m.get("redundancy") for m in instance_metrics]
    info_gain_values = [m.get("information_gain") for m in instance_metrics]

    error_categories = [
        m.get("error_category")
        for m in instance_metrics
        if m.get("error_category")
    ]

    token_overlap_values = [
        m.get("token_overlap") for m in instance_metrics if m.get("token_overlap")
    ]

    n_instances = len(instance_metrics)
    success_rate = _mean(success_values)
    avg_error_rate = _mean(error_rates)
    avg_turns = _mean(reasoning_turns)
    avg_message_density = _mean(message_density)
    avg_redundancy = _mean(redundancy)
    avg_information_gain = _mean(info_gain_values)

    total_prompt_tokens_sum = _sum(prompt_tokens)
    total_completion_tokens_sum = _sum(completion_tokens)
    total_tokens_sum = _sum(total_tokens)
    total_llm_calls = int(_sum(llm_call_counts))
    total_response_time_ms_sum = _sum(total_response_times)
    avg_response_time_ms = None
    if total_llm_calls > 0:
        avg_response_time_ms = total_response_time_ms_sum / total_llm_calls
    avg_instance_response_time_ms = _mean(total_response_times)
    total_cost_sum = _sum(total_cost)
    successes_sum = _sum(success_values)

    success_per_1k_tokens = None
    if total_tokens_sum > 0:
        success_per_1k_tokens = successes_sum / (total_tokens_sum / 1000.0)

    success_per_usd = None
    if total_cost_sum > 0:
        success_per_usd = successes_sum / total_cost_sum
    total_provider_cost_sum = _sum(total_provider_cost)
    total_openrouter_cost_sum = _sum(total_openrouter_cost)

    success_per_provider_usd = None
    if total_provider_cost_sum > 0:
        success_per_provider_usd = successes_sum / total_provider_cost_sum

    success_per_openrouter_usd = None
    if total_openrouter_cost_sum > 0:
        success_per_openrouter_usd = successes_sum / total_openrouter_cost_sum

    baseline = baseline_metrics.get("paper_metrics") if baseline_metrics else None
    baseline_turns = baseline.get("avg_reasoning_turns") if baseline else None
    baseline_error_rate = baseline.get("avg_error_rate") if baseline else None

    coordination_overhead = None
    coordination_efficiency = None
    error_amplification = None
    error_absorption = None

    if agent_name and agent_name.startswith("single-agent"):
        coordination_overhead = 0.0
        coordination_efficiency = success_rate
        error_amplification = 1.0
        error_absorption = 0.0
    elif baseline_turns and baseline_turns > 0 and avg_turns is not None:
        coordination_overhead = ((avg_turns - baseline_turns) / baseline_turns) * 100.0
        coordination_efficiency = (
            success_rate / (avg_turns / baseline_turns)
            if success_rate is not None and avg_turns > 0
            else None
        )

    if baseline_error_rate is not None and baseline_error_rate > 0:
        if avg_error_rate is not None:
            error_amplification = avg_error_rate / baseline_error_rate
            error_absorption = (baseline_error_rate - avg_error_rate) / baseline_error_rate

    token_overlap_agg = {
        "unique_token_ratio": _mean(
            [t.get("unique_token_ratio") for t in token_overlap_values]
        ),
        "shared_token_ratio": _mean(
            [t.get("shared_token_ratio") for t in token_overlap_values]
        ),
        "contradictory_token_ratio": _mean(
            [t.get("contradictory_token_ratio") for t in token_overlap_values]
        ),
        "shared_token_entropy_bits": _mean(
            [t.get("shared_token_entropy_bits") for t in token_overlap_values]
        ),
        "contradictory_mass": _mean(
            [t.get("contradictory_mass") for t in token_overlap_values]
        ),
    }

    error_taxonomy_counts: Optional[Dict[str, int]] = None
    if error_categories:
        counts: Dict[str, int] = {}
        for category in error_categories:
            counts[category] = counts.get(category, 0) + 1
        error_taxonomy_counts = counts

    domain_complexity = None
    if dataset_id and runs_root:
        domain_complexity = compute_domain_complexity_from_runs(runs_root, dataset_id)

    return {
        "n_instances": n_instances,
        "success_rate": success_rate,
        "avg_error_rate": avg_error_rate,
        "avg_reasoning_turns": avg_turns,
        "avg_inter_agent_messages": _mean(inter_agent_messages),
        "avg_message_density": avg_message_density,
        "avg_redundancy": avg_redundancy,
        "avg_information_gain": avg_information_gain,
        "success_per_1k_tokens": success_per_1k_tokens,
        "success_per_usd": success_per_usd,
        "total_prompt_tokens": total_prompt_tokens_sum,
        "total_completion_tokens": total_completion_tokens_sum,
        "total_tokens": total_tokens_sum,
        "llm_call_count": total_llm_calls,
        "total_response_time_ms": total_response_time_ms_sum,
        "avg_response_time_ms": avg_response_time_ms,
        "avg_instance_response_time_ms": avg_instance_response_time_ms,
        "total_cost_usd": total_cost_sum,
        "total_provider_cost_usd": total_provider_cost_sum,
        "total_openrouter_cost_usd": total_openrouter_cost_sum,
        "success_per_provider_usd": success_per_provider_usd,
        "success_per_openrouter_usd": success_per_openrouter_usd,
        "coordination_overhead_percent": coordination_overhead,
        "coordination_efficiency": coordination_efficiency,
        "error_amplification": error_amplification,
        "error_absorption": error_absorption,
        "token_overlap": token_overlap_agg,
        "error_taxonomy_counts": error_taxonomy_counts,
        "domain_complexity": domain_complexity,
        "baseline_run_dir": baseline_metrics.get("_run_dir") if baseline_metrics else None,
        "baseline_source": baseline_metrics.get("_source") if baseline_metrics else None,
    }
