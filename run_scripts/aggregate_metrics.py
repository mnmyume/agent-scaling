import argparse
import json
from pathlib import Path

from agent_scaling.metrics import aggregate_experiment_metrics
from agent_scaling.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate completed experiment folders into paper-style metrics."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more run directories or parent experiment directories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to paper_metrics_summary.json in the first directory argument.",
    )
    args = parser.parse_args()

    summary = aggregate_experiment_metrics(args.paths)
    output_path = args.output
    if output_path is None:
        default_root = Path(args.paths[0])
        output_path = str(
            (default_root if default_root.is_dir() else default_root.parent)
            / "paper_metrics_summary.json"
        )

    write_json(summary, output_path, indent=True)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved aggregated metrics to {output_path}")


if __name__ == "__main__":
    main()
