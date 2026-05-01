#!/usr/bin/env python
"""Compare two local experiment runs using per-instance trace_events.jsonl files."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyYAML is required to read run_config.yaml and instance_save.yaml. "
            "Run this script inside the project environment."
        ) from exc
    with path.open("r") as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _instance_dirs(run_dir: Path) -> Dict[str, Path]:
    root = run_dir / "instance_runs"
    if not root.exists():
        return {}
    return {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir()
    }


def _run_label(run_dir: Path) -> str:
    cfg = _read_yaml(run_dir / "run_config.yaml") or {}
    dataset = (cfg.get("dataset") or {}).get("dataset_id", "?")
    agent = (cfg.get("agent") or {}).get("name", "?")
    model = (cfg.get("llm") or {}).get("model", "?")
    return f"{run_dir} | dataset={dataset} agent={agent} model={model}"


def _instance_metrics(instance_dir: Path) -> Dict[str, Any]:
    data = _read_yaml(instance_dir / "instance_save.yaml") or {}
    metrics = data.get("metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def _metric_value(metrics: Dict[str, Any]) -> Any:
    for key in ("success", "resolved", "accuracy", "correct"):
        if key in metrics:
            return metrics[key]
    return metrics


def _shorten(value: Any, limit: int = 220) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _event_signature(event: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    event_type = event.get("event_type")
    if event_type == "llm_input":
        return None
    if event_type == "tool_call":
        return (
            event_type,
            event.get("agent_id"),
            event.get("tool_name"),
            json.dumps(event.get("tool_args", {}), sort_keys=True),
        )
    if event_type == "tool_observation":
        return (
            event_type,
            event.get("agent_id"),
            event.get("tool_name"),
            _shorten(event.get("observation", ""), 500),
        )
    if event_type in {
        "llm_response",
        "coordination_message",
        "error",
        "final_answer",
        "agent_message",
    }:
        metadata = event.get("metadata") or {}
        tool_calls = metadata.get("tool_calls", [])
        return (
            event_type,
            event.get("agent_id"),
            event.get("role"),
            _shorten(event.get("content", ""), 500),
            json.dumps(tool_calls, sort_keys=True),
        )
    return (
        event_type,
        event.get("agent_id"),
        _shorten(event.get("content", event), 500),
    )


def _semantic_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event for event in events if _event_signature(event) is not None]


def _describe_event(event: Optional[Dict[str, Any]]) -> str:
    if event is None:
        return "<missing event>"
    event_type = event.get("event_type")
    prefix = (
        f"{event_type} agent={event.get('agent_id')}"
        f" step={event.get('step')} round={event.get('round')}"
        f" iter={event.get('iteration')}"
    )
    if event_type == "tool_call":
        return (
            f"{prefix} tool={event.get('tool_name')} "
            f"args={_shorten(event.get('tool_args', {}))}"
        )
    if event_type == "tool_observation":
        return (
            f"{prefix} tool={event.get('tool_name')} "
            f"observation={_shorten(event.get('observation', ''))}"
        )
    metadata = event.get("metadata") or {}
    tool_calls = metadata.get("tool_calls")
    tool_text = f" tool_calls={_shorten(tool_calls)}" if tool_calls else ""
    return f"{prefix}{tool_text} content={_shorten(event.get('content', ''))}"


def _first_divergence(
    events_a: List[Dict[str, Any]],
    events_b: List[Dict[str, Any]],
) -> Optional[Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    semantic_a = _semantic_events(events_a)
    semantic_b = _semantic_events(events_b)
    max_len = max(len(semantic_a), len(semantic_b))
    for idx in range(max_len):
        event_a = semantic_a[idx] if idx < len(semantic_a) else None
        event_b = semantic_b[idx] if idx < len(semantic_b) else None
        if event_a is None or event_b is None:
            return idx, event_a, event_b
        if _event_signature(event_a) != _event_signature(event_b):
            return idx, event_a, event_b
    return None


def _changed_instances(
    run_a: Path,
    run_b: Path,
    only_instance: Optional[str],
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    dirs_a = _instance_dirs(run_a)
    dirs_b = _instance_dirs(run_b)
    instance_ids = sorted(set(dirs_a) | set(dirs_b))
    if only_instance is not None:
        instance_ids = [only_instance]

    changed = []
    for instance_id in instance_ids:
        metrics_a = _instance_metrics(dirs_a[instance_id]) if instance_id in dirs_a else {}
        metrics_b = _instance_metrics(dirs_b[instance_id]) if instance_id in dirs_b else {}
        if _metric_value(metrics_a) != _metric_value(metrics_b):
            changed.append((instance_id, metrics_a, metrics_b))
    return changed


def compare_runs(
    run_a: Path,
    run_b: Path,
    *,
    instance: Optional[str],
    limit: int,
) -> None:
    print("Run A:", _run_label(run_a))
    print("Run B:", _run_label(run_b))
    print()

    metrics_a = _read_json(run_a / "dataset_eval_metrics.json")
    metrics_b = _read_json(run_b / "dataset_eval_metrics.json")
    print("Aggregate metrics A:", metrics_a)
    print("Aggregate metrics B:", metrics_b)
    print()

    changed = _changed_instances(run_a, run_b, instance)
    if changed:
        print(f"Instances with changed primary metric: {len(changed)}")
        for instance_id, metrics_a, metrics_b in changed[:limit]:
            print(f"- {instance_id}: A={metrics_a} B={metrics_b}")
    else:
        print("No changed primary metrics found.")

    dirs_a = _instance_dirs(run_a)
    dirs_b = _instance_dirs(run_b)
    inspect_ids = [instance] if instance else [row[0] for row in changed[:limit]]
    if not inspect_ids and not instance:
        inspect_ids = sorted(set(dirs_a) & set(dirs_b))[: min(limit, 5)]

    print()
    print("First trace divergences:")
    for instance_id in inspect_ids:
        if instance_id not in dirs_a or instance_id not in dirs_b:
            print(f"- {instance_id}: missing from one run")
            continue
        events_a = _read_jsonl(dirs_a[instance_id] / "trace_events.jsonl")
        events_b = _read_jsonl(dirs_b[instance_id] / "trace_events.jsonl")
        if not events_a or not events_b:
            print(f"- {instance_id}: trace_events.jsonl missing from one run")
            continue
        divergence = _first_divergence(events_a, events_b)
        if divergence is None:
            print(f"- {instance_id}: no semantic divergence in trace events")
            continue
        idx, event_a, event_b = divergence
        print(f"- {instance_id}: first divergence at semantic event {idx}")
        print(f"  A: {_describe_event(event_a)}")
        print(f"  B: {_describe_event(event_b)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path, help="First experiment output directory")
    parser.add_argument("run_b", type=Path, help="Second experiment output directory")
    parser.add_argument(
        "--instance",
        help="Inspect one instance id, e.g. 0007. Defaults to changed instances.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum changed instances/divergences to print.",
    )
    args = parser.parse_args()

    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    for run_dir in (run_a, run_b):
        if not run_dir.exists():
            raise SystemExit(f"Run directory does not exist: {run_dir}")
        if not (run_dir / "instance_runs").exists():
            raise SystemExit(f"Run directory has no instance_runs/: {run_dir}")

    compare_runs(run_a, run_b, instance=args.instance, limit=args.limit)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
