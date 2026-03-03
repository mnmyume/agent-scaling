#!/usr/bin/env python3
"""Step 4: Evaluate the surrogate model on test benchmark results.

Loads the trained model and test results, predicts avg_success, and
computes R², MAE, MRE, MSE, and Mean Tree σ.

Usage:
    uv run python surrogate_pipeline/evaluate_surrogate.py \
        --run_name my_experiment
"""

import argparse
import csv

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from surrogate_pipeline.train_surrogate import encode_features
from surrogate_pipeline.utils import (
    BENCHMARKS_DIR,
    EVALUATION_DIR,
    MODEL_DIR,
    get_run_dir,
    load_json,
    save_json,
    setup_logging,
)


def compute_mean_tree_sigma(model, X: np.ndarray) -> float:
    """Compute mean standard deviation across individual tree predictions.

    For each sample, compute σ across all trees, then average across samples.
    """
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    # tree_preds shape: (n_trees, n_samples)
    per_sample_std = tree_preds.std(axis=0)  # std across trees for each sample
    return float(per_sample_std.mean())


def compute_mre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Relative Error, skipping zero-valued targets."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def main():
    parser = argparse.ArgumentParser(description="Evaluate surrogate model")
    parser.add_argument(
        "--run_name", type=str, required=True,
        help="Pipeline run name",
    )
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_name)
    logger = setup_logging(run_dir)
    eval_dir = run_dir / EVALUATION_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = run_dir / MODEL_DIR / "surrogate_model.joblib"
    meta_path = run_dir / MODEL_DIR / "feature_metadata.json"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)
    feature_metadata = load_json(meta_path)
    logger.info(f"Loaded model from {model_path}")

    # Reconstruct label encoders from metadata
    label_encoders = {}
    for name, classes in feature_metadata["label_encoders"].items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        label_encoders[name] = le

    # Load test results
    results_file = run_dir / BENCHMARKS_DIR / "test_results.json"
    if not results_file.exists():
        logger.error(f"Test results not found: {results_file}")
        raise FileNotFoundError(results_file)

    results = load_json(results_file)
    logger.info(f"Loaded {len(results)} test results")

    # Encode features using saved encoders
    X, y_true, _, feature_names = encode_features(
        results, label_encoders=label_encoders, fit=False
    )

    # Predict
    y_pred = model.predict(X)

    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mre = compute_mre(y_true, y_pred)
    mean_tree_sigma = compute_mean_tree_sigma(model, X)

    metrics = {
        "r2": r2,
        "mae": mae,
        "mre": mre,
        "mse": mse,
        "mean_tree_sigma": mean_tree_sigma,
        "num_test_samples": len(results),
    }

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  R²:             {r2:.4f}")
    logger.info(f"  MAE:            {mae:.4f}")
    logger.info(f"  MRE:            {mre:.4f}")
    logger.info(f"  MSE:            {mse:.6f}")
    logger.info(f"  Mean Tree σ:    {mean_tree_sigma:.4f}")
    logger.info(f"  Test samples:   {len(results)}")
    logger.info("=" * 50)

    # Save evaluation report
    save_json(metrics, eval_dir / "evaluation_report.json")

    # Save per-config predictions
    predictions_path = eval_dir / "predictions.csv"
    with open(predictions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config_id", "actual", "predicted", "abs_error", "rel_error"])
        for r, actual, predicted in zip(results, y_true, y_pred):
            abs_err = abs(actual - predicted)
            rel_err = abs_err / actual if actual != 0 else float("nan")
            writer.writerow([
                r["config_id"],
                f"{actual:.4f}",
                f"{predicted:.4f}",
                f"{abs_err:.4f}",
                f"{rel_err:.4f}",
            ])

    logger.info(f"Report saved: {eval_dir / 'evaluation_report.json'}")
    logger.info(f"Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
