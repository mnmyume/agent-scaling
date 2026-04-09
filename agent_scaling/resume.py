from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_scaling.datasets import Dataset, DatasetInstance
from agent_scaling.utils import read_yaml

_VOLATILE_RUN_FIELDS = {"resume", "run_name", "save_dir"}


def get_run_instances(
    dataset: Dataset,
    *,
    debug: bool,
    max_instances: Optional[int],
    dataset_filter: Optional[str],
) -> List[DatasetInstance]:
    limit = 10 if debug else len(dataset.instances)
    limit = min(limit, max_instances) if max_instances is not None else limit

    if dataset_filter is not None:
        instances = [
            instance
            for i, instance in enumerate(dataset.instances)
            if eval(dataset_filter, {}, {"x": instance, "i": i})
        ]
    else:
        instances = dataset.instances

    return instances[:limit]


def normalize_run_metadata_for_resume(metadata: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(metadata)
    for field_name in _VOLATILE_RUN_FIELDS:
        normalized.pop(field_name, None)
    return normalized


def get_completed_instance_indices(run_dir: str | Path) -> List[int]:
    instance_root = Path(run_dir) / "instance_runs"
    if not instance_root.exists():
        return []

    completed: List[int] = []
    for instance_dir in sorted(instance_root.iterdir()):
        if not instance_dir.is_dir():
            continue
        if not (instance_dir / "instance_save.yaml").exists():
            continue
        try:
            completed.append(int(instance_dir.name))
        except ValueError:
            continue
    return completed


def _has_aggregate_outputs(run_dir: Path) -> bool:
    return (run_dir / "dataset_eval_metrics.json").exists() and (
        run_dir / "run_runtime_metrics.json"
    ).exists()


def _run_dir_recency_key(run_dir: Path) -> tuple[int, str, str, str]:
    path = run_dir.resolve()
    parts = path.parts
    if len(parts) >= 2:
        date_part = parts[-2]
        time_part = parts[-1]
        if len(date_part) == 10 and len(time_part) == 8:
            return (1, date_part, time_part, str(path))
    return (0, "", "", str(path))


def find_latest_matching_incomplete_run(
    *,
    output_dir: str,
    run_metadata: Dict[str, Any],
    expected_instance_count: int,
) -> Optional[str]:
    current_run_dir = Path(output_dir).resolve()
    model_root = current_run_dir.parents[1]
    target_metadata = normalize_run_metadata_for_resume(run_metadata)
    candidates: List[Path] = []

    for config_path in model_root.rglob("run_config.yaml"):
        run_dir = config_path.parent.resolve()
        if run_dir == current_run_dir:
            continue

        existing_metadata = normalize_run_metadata_for_resume(read_yaml(str(config_path)))
        if existing_metadata != target_metadata:
            continue

        completed_indices = get_completed_instance_indices(run_dir)
        if len(completed_indices) >= expected_instance_count and _has_aggregate_outputs(
            run_dir
        ):
            continue

        candidates.append(run_dir)

    if not candidates:
        return None

    return str(max(candidates, key=_run_dir_recency_key))
