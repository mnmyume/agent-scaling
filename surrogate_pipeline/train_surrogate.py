#!/usr/bin/env python3
"""Step 3: Train a surrogate model on benchmark results.

Reads train results from surrogate_outputs/<run_name>/benchmarks/,
trains a RandomForestRegressor to predict avg_success, and saves
the model artifact.

Usage:
    uv run python surrogate_pipeline/train_surrogate.py \
        --run_name my_experiment \
        --n_estimators 100
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from surrogate_pipeline.utils import (
    BENCHMARKS_DIR,
    MODEL_DIR,
    get_run_dir,
    load_json,
    save_json,
    setup_logging,
)


def encode_features(
    results: list[dict],
    label_encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, LabelEncoder], list[str]]:
    """Encode configuration features into numeric arrays.

    Features: architecture (label), each role's model (label).
    Target: avg_success.

    Returns: (X, y, label_encoders, feature_names)
    """
    if not results:
        raise ValueError("No results to encode")

    # Collect all role names across configs
    all_roles = set()
    for r in results:
        all_roles.update(r["config"]["role_models"].keys())
    all_roles = sorted(all_roles)

    feature_names = ["architecture"] + [f"model_{role}" for role in all_roles]

    if label_encoders is None:
        label_encoders = {}

    rows_X = []
    rows_y = []

    for r in results:
        config = r["config"]
        metrics = r["metrics"]

        row = [config["architecture"]]
        for role in all_roles:
            row.append(config["role_models"].get(role, "__none__"))

        rows_X.append(row)
        rows_y.append(metrics.get("avg_success", 0.0))

    # Encode each feature column
    X_encoded = np.zeros((len(rows_X), len(feature_names)), dtype=float)

    for col_idx, feat_name in enumerate(feature_names):
        col_values = [row[col_idx] for row in rows_X]
        if feat_name not in label_encoders:
            if fit:
                le = LabelEncoder()
                le.fit(col_values)
                label_encoders[feat_name] = le
            else:
                raise ValueError(f"No encoder for feature {feat_name}")
        else:
            le = label_encoders[feat_name]
            # Handle unseen labels at inference time
            if not fit:
                unseen = set(col_values) - set(le.classes_)
                if unseen:
                    le.classes_ = np.append(le.classes_, list(unseen))

        X_encoded[:, col_idx] = label_encoders[feat_name].transform(col_values)

    y = np.array(rows_y, dtype=float)
    return X_encoded, y, label_encoders, feature_names


def main():
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument(
        "--run_name", type=str, required=True,
        help="Pipeline run name",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100,
        help="Number of trees in RandomForest (default: 100)",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random state for reproducibility",
    )
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_name)
    logger = setup_logging(run_dir)
    model_dir = run_dir / MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load train results
    results_file = run_dir / BENCHMARKS_DIR / "train_results.json"
    if not results_file.exists():
        logger.error(f"Train results not found: {results_file}")
        raise FileNotFoundError(results_file)

    results = load_json(results_file)
    logger.info(f"Loaded {len(results)} training results")

    # Encode features
    X, y, label_encoders, feature_names = encode_features(results, fit=True)
    logger.info(f"Feature matrix: {X.shape}, target: {y.shape}")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}")

    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Report in-sample score
    train_score = model.score(X, y)
    logger.info(f"In-sample R²: {train_score:.4f}")

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_))
    logger.info(f"Feature importances: {json.dumps(importances, indent=2)}")

    # Save model and metadata
    joblib.dump(model, model_dir / "surrogate_model.joblib")

    # Save encoders and feature info for inference
    encoder_data = {}
    for name, le in label_encoders.items():
        encoder_data[name] = le.classes_.tolist()

    feature_metadata = {
        "feature_names": feature_names,
        "label_encoders": encoder_data,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "train_samples": len(results),
        "train_r2": train_score,
        "feature_importances": importances,
    }
    save_json(feature_metadata, model_dir / "feature_metadata.json")

    logger.info(f"Model saved to {model_dir / 'surrogate_model.joblib'}")
    logger.info(f"Metadata saved to {model_dir / 'feature_metadata.json'}")


if __name__ == "__main__":
    main()
