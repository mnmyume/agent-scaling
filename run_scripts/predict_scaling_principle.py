#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_EXP_ROOT = "exp_outputs"
DEFAULT_MODEL = "minimax/MiniMax-M2.7"
DEFAULT_TOKEN_BUDGET = 4800
DEFAULT_OUTPUT_DIR = "predict_results"
SUMMARY_FILENAME = "paper_metrics_summary.json"

INTELLIGENCE_INDEX_BY_MODEL = {
    "minimax/MiniMax-M2.7": 50.0,
}
INTELLIGENCE_MEAN = 56.9

TOOL_COUNT_BY_DATASET = {
    "workbench": 16,
    "plancraft-test": 4,
}

AGENT_COUNT_BY_ARCHITECTURE = {
    "single-agent": 1,
    "multi-agent-independent": 3,
    "multi-agent-decentralized": 3,
    "multi-agent-centralized": 3,
    "multi-agent-hybrid": 3,
}

# Table 4 coefficients from Kim et al. (2025), "Towards a Science of Scaling Agent Systems".
COEFFICIENTS = {
    "intercept": 0.453,
    "intelligence_centered": 0.171,
    "intelligence_centered_sq": 0.007,
    "log1p_tool_count": 0.411,
    "log1p_agent_count": 0.052,
    "log1p_overhead_percent": 0.034,
    "message_density_c": -0.057,
    "redundancy_R": -0.007,
    "coordination_efficiency_Ec": -0.043,
    "log1p_error_amplification": -0.022,
    "baseline_success_rate_sas": 0.315,
    "baseline_success_rate_sas_x_log1p_agent_count": -0.404,
    "coordination_efficiency_Ec_x_tool_count": -0.267,
    "overhead_percent_x_tool_count": -0.162,
    "error_amplification_x_tool_count": -0.019,
    "redundancy_R_x_agent_count": 0.047,
    "intelligence_centered_x_coordination_efficiency_Ec": -0.022,
    "error_amplification_x_baseline_success_rate_sas": -0.065,
    "message_density_c_x_intelligence_centered": -0.011,
    "intelligence_centered_x_log1p_tool_count": -0.069,
}

PREDICTOR_COLUMNS = [
    "intelligence_centered",
    "intelligence_centered_sq",
    "log1p_tool_count",
    "log1p_agent_count",
    "log1p_overhead_percent",
    "message_density_c",
    "redundancy_R",
    "coordination_efficiency_Ec",
    "log1p_error_amplification",
    "baseline_success_rate_sas",
    "baseline_success_rate_sas_x_log1p_agent_count",
    "coordination_efficiency_Ec_x_tool_count",
    "overhead_percent_x_tool_count",
    "error_amplification_x_tool_count",
    "redundancy_R_x_agent_count",
    "intelligence_centered_x_coordination_efficiency_Ec",
    "error_amplification_x_baseline_success_rate_sas",
    "message_density_c_x_intelligence_centered",
    "intelligence_centered_x_log1p_tool_count",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def discover_summary_paths(
    exp_root: str,
    model: str,
    token_budget: int,
) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(exp_root):
        if SUMMARY_FILENAME not in files:
            continue
        path = os.path.join(root, SUMMARY_FILENAME)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("model") != model:
            continue
        if int(data.get("token_budget", -1)) != int(token_budget):
            continue
        if data.get("dataset_id") not in TOOL_COUNT_BY_DATASET:
            continue
        matches.append(path)
    return sorted(matches)


def _selected_run_by_architecture(selected_runs: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    mapping: Dict[str, Dict[str, object]] = {}
    for run in selected_runs:
        architecture = str(run.get("architecture"))
        run_summary = run.get("run_summary", {})
        if isinstance(run_summary, dict):
            mapping[architecture] = run_summary
    return mapping


def _build_paired_row(summary: Dict[str, object], paired_row: Dict[str, object]) -> Dict[str, object]:
    dataset_id = str(summary["dataset_id"])
    architecture = str(paired_row["architecture"])
    model = str(summary["model"])
    intelligence_index = INTELLIGENCE_INDEX_BY_MODEL[model]
    tool_count = TOOL_COUNT_BY_DATASET[dataset_id]
    agent_count = AGENT_COUNT_BY_ARCHITECTURE[architecture]

    return {
        "dataset_id": dataset_id,
        "architecture": architecture,
        "model": model,
        "token_budget": int(summary["token_budget"]),
        "row_source": "paired_metrics",
        "actual_success_rate": _safe_float(paired_row.get("success_rate_S")),
        "baseline_success_rate_sas": _safe_float(
            paired_row.get("baseline_success_rate_sas")
        ),
        "message_density_c": _safe_float(paired_row.get("message_density_c")),
        "coordination_efficiency_Ec": _safe_float(
            paired_row.get("coordination_efficiency_Ec")
        ),
        "communication_overhead_percent_O": _safe_float(
            paired_row.get("communication_overhead_percent_O")
        ),
        "failure_amplification_proxy": _safe_float(
            paired_row.get("failure_amplification_proxy")
        ),
        "redundancy_proxy_bow_cosine": _safe_float(
            paired_row.get("redundancy_proxy_bow_cosine"),
            default=0.0,
        ),
        "intelligence_index": intelligence_index,
        "tool_count": float(tool_count),
        "agent_count": float(agent_count),
    }


def _build_single_agent_row(summary: Dict[str, object]) -> Dict[str, object]:
    dataset_id = str(summary["dataset_id"])
    model = str(summary["model"])
    intelligence_index = INTELLIGENCE_INDEX_BY_MODEL[model]
    selected_runs = _selected_run_by_architecture(summary.get("selected_runs", []))
    single_agent_run = selected_runs.get("single-agent")
    if single_agent_run is None:
        raise ValueError(
            f"Missing single-agent selected run for dataset '{dataset_id}'."
        )

    actual_success_rate = _safe_float(single_agent_run.get("success_rate"))
    return {
        "dataset_id": dataset_id,
        "architecture": "single-agent",
        "model": model,
        "token_budget": int(summary["token_budget"]),
        "row_source": "selected_runs_baseline",
        "actual_success_rate": actual_success_rate,
        "baseline_success_rate_sas": actual_success_rate,
        "message_density_c": _safe_float(single_agent_run.get("message_density_c")),
        # By definition, T / T_SA = 1 for the baseline, so E_c reduces to success rate.
        "coordination_efficiency_Ec": actual_success_rate,
        "communication_overhead_percent_O": 0.0,
        # Single-agent compared to itself yields a failure-amplification proxy of 1.
        "failure_amplification_proxy": 1.0,
        "redundancy_proxy_bow_cosine": 0.0,
        "intelligence_index": intelligence_index,
        "tool_count": float(TOOL_COUNT_BY_DATASET[dataset_id]),
        "agent_count": float(AGENT_COUNT_BY_ARCHITECTURE["single-agent"]),
    }


def load_prediction_rows(summary_paths: Iterable[str], include_single_agent: bool) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in summary_paths:
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

        paired_metrics = summary.get("paired_metrics", [])
        if not isinstance(paired_metrics, list):
            raise ValueError(f"'paired_metrics' is not a list in {path}")

        for paired_row in paired_metrics:
            if not isinstance(paired_row, dict):
                continue
            rows.append(_build_paired_row(summary, paired_row))

        if include_single_agent:
            rows.append(_build_single_agent_row(summary))

    if not rows:
        raise ValueError("No prediction rows were loaded from the provided summaries.")

    return pd.DataFrame(rows)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    feature_df["intelligence_centered"] = (
        feature_df["intelligence_index"] - INTELLIGENCE_MEAN
    )
    feature_df["intelligence_centered_sq"] = (
        feature_df["intelligence_centered"] ** 2
    )

    feature_df["log1p_tool_count"] = np.log1p(feature_df["tool_count"])
    feature_df["log1p_agent_count"] = np.log1p(feature_df["agent_count"])
    feature_df["log1p_overhead_percent"] = np.log1p(
        feature_df["communication_overhead_percent_O"]
    )
    feature_df["log1p_error_amplification"] = np.log1p(
        feature_df["failure_amplification_proxy"]
    )
    feature_df["redundancy_R"] = feature_df["redundancy_proxy_bow_cosine"].fillna(0.0)

    # Interaction terms follow Table 4 exactly. Note that several interactions keep
    # raw T or raw n_a even though their main effects use log1p-transformed versions.
    feature_df["baseline_success_rate_sas_x_log1p_agent_count"] = (
        feature_df["baseline_success_rate_sas"] * feature_df["log1p_agent_count"]
    )
    feature_df["coordination_efficiency_Ec_x_tool_count"] = (
        feature_df["coordination_efficiency_Ec"] * feature_df["tool_count"]
    )
    feature_df["overhead_percent_x_tool_count"] = (
        feature_df["communication_overhead_percent_O"] * feature_df["tool_count"]
    )
    feature_df["error_amplification_x_tool_count"] = (
        feature_df["failure_amplification_proxy"] * feature_df["tool_count"]
    )
    feature_df["redundancy_R_x_agent_count"] = (
        feature_df["redundancy_R"] * feature_df["agent_count"]
    )
    feature_df["intelligence_centered_x_coordination_efficiency_Ec"] = (
        feature_df["intelligence_centered"] * feature_df["coordination_efficiency_Ec"]
    )
    feature_df["error_amplification_x_baseline_success_rate_sas"] = (
        feature_df["failure_amplification_proxy"]
        * feature_df["baseline_success_rate_sas"]
    )
    feature_df["message_density_c_x_intelligence_centered"] = (
        feature_df["message_density_c"] * feature_df["intelligence_centered"]
    )
    feature_df["intelligence_centered_x_log1p_tool_count"] = (
        feature_df["intelligence_centered"] * feature_df["log1p_tool_count"]
    )

    return feature_df


def zscore_standardize(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    standardized = pd.DataFrame(index=df.index)
    for column in columns:
        values = df[column].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0.0:
            standardized[column] = 0.0
        else:
            standardized[column] = (values - mean) / std
    return standardized


def predict_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = add_engineered_features(df)
    standardized_predictors = zscore_standardize(feature_df, PREDICTOR_COLUMNS)

    prediction = np.full(len(feature_df), COEFFICIENTS["intercept"], dtype=float)
    for column in PREDICTOR_COLUMNS:
        prediction += standardized_predictors[column].to_numpy() * COEFFICIENTS[column]

    result_df = feature_df.copy()
    # Equation 1 is a linear mixed-effects model on raw success rate, so the raw
    # linear score can fall outside [0, 1]. We keep that paper-faithful value for
    # diagnostics, then clip to the feasible success-rate range for reporting.
    result_df["predicted_success_rate_raw"] = prediction
    result_df["predicted_success_rate"] = np.clip(prediction, 0.0, 1.0)
    result_df["prediction_was_clipped"] = (
        result_df["predicted_success_rate_raw"] != result_df["predicted_success_rate"]
    )
    result_df["absolute_error"] = (
        result_df["predicted_success_rate"] - result_df["actual_success_rate"]
    ).abs()
    result_df["predicted_rank_within_dataset"] = (
        result_df.groupby("dataset_id")["predicted_success_rate"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    result_df["actual_rank_within_dataset"] = (
        result_df.groupby("dataset_id")["actual_success_rate"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    return result_df


def build_output_table(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "architecture",
        "row_source",
        "predicted_success_rate_raw",
        "predicted_success_rate",
        "prediction_was_clipped",
        "actual_success_rate",
        "absolute_error",
        "predicted_rank_within_dataset",
        "actual_rank_within_dataset",
    ]
    output = predictions.loc[:, columns].copy()
    return output.sort_values(
        by=["dataset_id", "predicted_success_rate", "actual_success_rate"],
        ascending=[True, False, False],
        ignore_index=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the Kim et al. (2025) scaling-principle regression to "
            "MiniMax paper-metric summaries."
        )
    )
    parser.add_argument(
        "--exp-root",
        default=DEFAULT_EXP_ROOT,
        help="Root directory to search for paper_metrics_summary.json files.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id to filter summary files by.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Token budget to filter summary files by.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where prediction outputs will be written.",
    )
    parser.add_argument(
        "--include-single-agent",
        action="store_true",
        help="Also include the synthetic single-agent baseline rows from selected_runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = discover_summary_paths(
        exp_root=args.exp_root,
        model=args.model,
        token_budget=args.token_budget,
    )

    if not summary_paths:
        raise FileNotFoundError(
            "No matching paper_metrics_summary.json files were found for "
            f"model={args.model!r} and token_budget={args.token_budget}."
        )

    input_df = load_prediction_rows(
        summary_paths=summary_paths,
        include_single_agent=args.include_single_agent,
    )
    predictions = predict_success_rates(input_df)
    output_df = build_output_table(predictions)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "prediction_comparison.csv")
    output_df.to_csv(output_path, index=False)

    mae = float(output_df["absolute_error"].mean())
    pd.set_option("display.float_format", lambda value: f"{value:.6f}")

    print("Loaded summaries:")
    for path in summary_paths:
        print(f"  - {path}")
    print()
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Saved comparison CSV to: {output_path}")
    print()
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()
