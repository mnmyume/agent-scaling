import argparse
import csv
import os
import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/olly-styles/WorkBench/main"


PROCESSED_FILES = [
    "data/processed/emails.csv",
    "data/processed/calendar_events.csv",
    "data/processed/project_tasks.csv",
    "data/processed/customer_relationship_manager_data.csv",
    "data/processed/analytics_data.csv",
]

RAW_FILES = [
    "data/raw/email_addresses.csv",
]

Q_AND_A_FILES = [
    "data/processed/queries_and_answers/email_queries_and_answers.csv",
    "data/processed/queries_and_answers/calendar_queries_and_answers.csv",
    "data/processed/queries_and_answers/analytics_queries_and_answers.csv",
    "data/processed/queries_and_answers/project_management_queries_and_answers.csv",
    "data/processed/queries_and_answers/customer_relationship_manager_queries_and_answers.csv",
    "data/processed/queries_and_answers/multi_domain_queries_and_answers.csv",
]


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "agent-scaling/workbench-fetch"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def _write_combined_queries_csv(workbench_dir: Path) -> Path:
    src_dir = workbench_dir / "processed" / "queries_and_answers"
    out_path = workbench_dir / "workbench_all.csv"

    fields = [
        "query",
        "answer",
        "domains",
        "subset",
        "base_template",
        "chosen_template",
    ]
    rows = []
    for p in sorted(src_dir.glob("*_queries_and_answers.csv")):
        subset = p.name.replace("_queries_and_answers.csv", "")
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(
                    {
                        "query": row.get("query", ""),
                        "answer": row.get("answer", ""),
                        "domains": row.get("domains", ""),
                        "subset": subset,
                        "base_template": row.get("base_template", ""),
                        "chosen_template": row.get("chosen_template", ""),
                    }
                )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WorkBench data into datasets/workbench/ and create workbench_all.csv."
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Base raw GitHub URL (default: WorkBench main branch).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("datasets", "workbench"),
        help="Output directory (default: datasets/workbench).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Writing WorkBench data to: {out_dir}")

    for rel in PROCESSED_FILES + RAW_FILES + Q_AND_A_FILES:
        url = f"{args.base_url}/{rel}"
        dest = out_dir / rel.replace("data/", "")
        print(f"Downloading: {url}")
        _download(url, dest)

    combined = _write_combined_queries_csv(out_dir)
    print(f"Wrote combined queries CSV: {combined}")


if __name__ == "__main__":
    main()

