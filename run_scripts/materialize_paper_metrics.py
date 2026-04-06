import argparse
import json

from agent_scaling.metrics import (
    get_materialized_paper_metrics_output_path,
    materialize_paper_metrics,
)
from agent_scaling.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best completed run for each architecture and save canonical "
            "paper metrics under exp_outputs/<dataset>/paper_metrics/..."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more run directories or parent experiment directories.",
    )
    args = parser.parse_args()

    summaries = materialize_paper_metrics(args.paths)
    if not summaries:
        raise SystemExit("No experiment runs were discovered under the provided paths.")

    saved = []
    for summary in summaries:
        output_path = get_materialized_paper_metrics_output_path(summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(summary, str(output_path), indent=True)
        saved.append(
            {
                "dataset_id": summary["dataset_id"],
                "model": summary["model"],
                "token_budget": summary["token_budget"],
                "output_path": str(output_path),
                "shared_instance_count_all_selected": summary[
                    "shared_instance_count_all_selected"
                ],
                "paired_architecture_count": len(summary["paired_metrics"]),
            }
        )

    print(json.dumps({"saved_summaries": saved}, indent=2))


if __name__ == "__main__":
    main()
