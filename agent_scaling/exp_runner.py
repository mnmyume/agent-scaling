import concurrent.futures
import contextlib
import os
import traceback
from typing import Any, Dict, List, Optional, Union

from langfuse._client.span import LangfuseSpan
from pydantic import BaseModel
from tqdm import tqdm

from agent_scaling.agents import AgentSystem
from agent_scaling.config.run import RunConfig
from agent_scaling.datasets import Dataset, DatasetInstance
from agent_scaling.logger import logger
from agent_scaling.metrics import aggregate_instance_runtime_metrics
from agent_scaling.resume import get_run_instances
from agent_scaling.utils import read_json, read_yaml, write_json, write_yaml
from agent_scaling.utils.token_budget import TokenBudgetManager


class InstanceSave(BaseModel):
    inp: Dict[str, Any]
    output: Dict[str, Any]
    metrics: Dict[str, Union[int, float, str]]
    expected_output: Optional[Any] = None


class ProcessedInstanceResult(BaseModel):
    eval_metrics: Any
    runtime_metrics: Optional[Dict[str, Any]] = None


class ExperimentRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.log_langfuse = config.log_langfuse
        self.output_dir = config.save_dir
        self.dataset: Dataset = self.config.dataset.dataset
        self.lf_dataset = self.config.dataset.langfuse_dataset
        self.agent: AgentSystem = self.config.get_agent()
        logger.debug(f"Logging to langfuse: {self.log_langfuse}")

    def _get_context_manager(self, index: int) -> contextlib.AbstractContextManager:
        if self.log_langfuse:
            assert self.lf_dataset is not None, (
                "Langfuse dataset must be set for logging"
            )
            return self.lf_dataset.items[index].run(
                run_name=self.config.run_name,
                run_description=self.config.llm.model,
                run_metadata=self.config.get_run_metadata(),
            )
        return contextlib.nullcontext()

    def _get_instances(self) -> List[DatasetInstance]:
        return get_run_instances(
            self.dataset,
            debug=self.config.debug,
            max_instances=self.config.max_instances,
            dataset_filter=self.config.dataset.dataset_filter,
        )

    def _get_instance_dir(self, instance_idx: int) -> Optional[str]:
        if self.output_dir is None:
            return None
        return os.path.join(self.output_dir, "instance_runs", f"{instance_idx:04d}")

    def _load_saved_instance_result(
        self, instance_dir: Optional[str]
    ) -> Optional[ProcessedInstanceResult]:
        if instance_dir is None:
            return None

        save_path = os.path.join(instance_dir, "instance_save.yaml")
        if not os.path.exists(save_path):
            return None

        saved = read_yaml(save_path) or {}
        metrics = saved.get("metrics")
        if not isinstance(metrics, dict):
            return None

        runtime_metrics = None
        runtime_metrics_path = os.path.join(instance_dir, "runtime_metrics.json")
        if os.path.exists(runtime_metrics_path):
            runtime_metrics = read_json(runtime_metrics_path)

        return ProcessedInstanceResult(
            eval_metrics=metrics,
            runtime_metrics=runtime_metrics,
        )

    def _prepare_resume_state(
        self, instances: List[DatasetInstance]
    ) -> tuple[
        List[Optional[Dict[str, Any] | str]],
        List[Optional[Dict[str, Any]]],
        List[tuple[int, DatasetInstance, Optional[str]]],
    ]:
        metrics: List[Optional[Dict[str, Any] | str]] = [None] * len(instances)
        runtime_metrics: List[Optional[Dict[str, Any]]] = [None] * len(instances)
        work_items: List[tuple[int, DatasetInstance, Optional[str]]] = []
        resumed_count = 0

        for i, instance in enumerate(instances):
            instance_dir = self._get_instance_dir(i)
            if instance_dir is not None:
                os.makedirs(instance_dir, exist_ok=True)

            saved_result = self._load_saved_instance_result(instance_dir)
            if saved_result is not None:
                metrics[i] = saved_result.eval_metrics
                runtime_metrics[i] = saved_result.runtime_metrics
                resumed_count += 1
                continue

            work_items.append((i, instance, instance_dir))

        if resumed_count:
            logger.info(
                "Resuming {} completed instance(s) from {}; {} remaining",
                resumed_count,
                self.output_dir,
                len(work_items),
            )

        return metrics, runtime_metrics, work_items

    def run(self):
        instances = self._get_instances()
        metrics, runtime_metrics, work_items = self._prepare_resume_state(instances)

        if self.log_langfuse:
            iterator = tqdm(
                work_items,
                total=len(work_items),
                desc=(
                    f"Evaluating {self.dataset.dataset_id} dataset instances"
                    + (" (resume)" if work_items and len(work_items) < len(instances) else "")
                ),
            )
        else:
            iterator = work_items

        for i, instance, instance_dir in iterator:
            if self.config.debug and i >= 10:
                break

            inst_result = self._process_single_instance(i, instance, instance_dir)
            metrics[i] = inst_result.eval_metrics
            runtime_metrics[i] = inst_result.runtime_metrics

        final_metrics = [metric for metric in metrics if metric is not None]
        final_runtime_metrics = [
            metric for metric in runtime_metrics if metric is not None
        ]

        all_metrics = self.dataset.get_metrics(final_metrics)
        if self.output_dir is not None:
            write_json(
                all_metrics,
                os.path.join(self.output_dir, "dataset_eval_metrics.json"),
                indent=True,
            )
            write_json(
                aggregate_instance_runtime_metrics(
                    final_runtime_metrics,
                    dataset_id=self.dataset.dataset_id,
                    architecture=self.config.agent.name,
                    model=self.config.llm.model,
                    token_budget=self.config.token_budget.total_tokens_per_instance,
                    run_dir=self.output_dir,
                ),
                os.path.join(self.output_dir, "run_runtime_metrics.json"),
                indent=True,
            )
        return all_metrics

    def _process_single_instance(
        self,
        i: int,
        instance,
        instance_dir: Optional[str] = None,
    ) -> ProcessedInstanceResult:
        """
        Core worker function to process a single instance.
        This can be reused for both sequential and parallel processing.

        Args:
            i: Index of the instance
            instance: Dataset instance to process
            instance_dir: Optional directory to save outputs

        Returns:
            Tuple of (index, metrics) to maintain order
        """

        context_manager = self._get_context_manager(i)
        with context_manager as span:
            agent = self.config.get_agent()
            budget_cfg = self.config.token_budget
            budget_manager = TokenBudgetManager.create(
                enabled=budget_cfg.enabled,
                total_tokens=budget_cfg.total_tokens_per_instance,
            )
            output = agent.run_agent(
                instance,
                instance_dir=instance_dir,
                llm_params=self.config.llm.params,
                instance_idx=i,
                budget_manager=budget_manager,
            )
            inst_metrics = self.dataset.get_instance_eval_metrics(output)
            inst_output = self.dataset.get_instance_eval_output(output)
            if isinstance(span, LangfuseSpan):
                span.update(
                    input=instance.get_prompt_info(),
                    output=inst_output,
                )
                for name, metric in inst_metrics.items():
                    span.score_trace(
                        name=name,
                        value=metric,
                    )

            runtime_metrics = self._finalize_runtime_metrics(output, inst_metrics, inst_output)

            if instance_dir is not None:
                output_save = InstanceSave(
                    inp=instance.get_prompt_info(),
                    output=inst_output,
                    metrics=inst_metrics,  # type: ignore
                    expected_output=instance.expected_output,
                )
                write_yaml(
                    output_save.model_dump(),
                    os.path.join(instance_dir, "instance_save.yaml"),
                    indent=True,
                )

            return ProcessedInstanceResult(
                eval_metrics=inst_metrics,
                runtime_metrics=runtime_metrics,
            )

    def _finalize_runtime_metrics(
        self,
        output,
        inst_metrics: Dict[str, Any],
        inst_output: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        runtime_metrics = output.runtime_metrics
        if runtime_metrics is None:
            return None

        runtime_metrics["evaluation"] = {
            "metrics": inst_metrics,
            "output": inst_output,
        }
        task_success = self.dataset.get_instance_success(inst_metrics, output)
        if task_success is not None:
            system_metrics = runtime_metrics.setdefault("system_metrics", {})
            summary = runtime_metrics.setdefault("summary", {})
            system_metrics["task_success"] = bool(task_success)
            system_metrics["task_success_source"] = "dataset_evaluation"
            summary["task_success"] = bool(task_success)
            summary["task_success_source"] = "dataset_evaluation"

            total_tokens = (
                system_metrics.get("total_tokens_used")
                or summary.get("total_tokens")
                or 0
            )
            success_per_1k_tokens = (
                1000 * int(task_success) / total_tokens if total_tokens else None
            )
            system_metrics["success_per_1k_tokens"] = success_per_1k_tokens
            summary["success_per_1k_tokens"] = success_per_1k_tokens

        if output.runtime_metrics_path is not None:
            write_json(runtime_metrics, output.runtime_metrics_path, indent=True)

        return runtime_metrics

    def run_parallel(self, num_workers: int = 4):
        """
        Run the experiment in parallel using multiprocessing.

        Args:
            num_workers: Maximum number of worker processes
        """
        instances = self._get_instances()
        metrics, runtime_metrics, work_items = self._prepare_resume_state(instances)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_instance,
                    *work_item,
                ): work_item[0]
                for work_item in work_items
            }
            with tqdm(
                total=len(work_items),
                desc=(
                    f"Evaluating {self.dataset.dataset_id} dataset instances (parallel)"
                    + (" (resume)" if work_items and len(work_items) < len(instances) else "")
                ),
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        metrics[i] = result.eval_metrics
                        runtime_metrics[i] = result.runtime_metrics
                    except Exception as exc:
                        tb_str = "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        )
                        logger.error(
                            f"Error generating output: {exc}\nTraceback:\n{tb_str}"
                        )
                        metrics[i] = (
                            f"Failed with exception: {exc}\nTraceback:\n{tb_str}"
                        )

                    pbar.update(1)

        final_metrics = [metric for metric in metrics if metric is not None]
        final_runtime_metrics = [
            metric for metric in runtime_metrics if metric is not None
        ]

        all_metrics = self.dataset.get_metrics(final_metrics)
        if self.output_dir is not None:
            write_json(
                all_metrics,
                os.path.join(self.output_dir, "dataset_eval_metrics.json"),
                indent=True,
            )
            write_json(
                aggregate_instance_runtime_metrics(
                    final_runtime_metrics,
                    dataset_id=self.dataset.dataset_id,
                    architecture=self.config.agent.name,
                    model=self.config.llm.model,
                    token_budget=self.config.token_budget.total_tokens_per_instance,
                    run_dir=self.output_dir,
                ),
                os.path.join(self.output_dir, "run_runtime_metrics.json"),
                indent=True,
            )
        return all_metrics
