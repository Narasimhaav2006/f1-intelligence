"""
Final model: retrain on 2014-2023, evaluate ONCE on the sealed 2024-2025 test
seasons, save the model, and record it in the model_registry collection.

Refuses to run again if the model file exists (use --force only if a previous
run crashed before showing results). Re-running to improve test scores would
turn the test set into a second validation set.

Run from the project root with the virtual environment active:
    python ml/07_final_model.py
"""

import importlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import ALL_FEATURE_COLUMNS, LABEL_COLUMN

baseline = importlib.import_module("04_baseline")
tune = importlib.import_module("06_tune_and_compare")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "podium_rf_v1"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / f"{MODEL_VERSION}.joblib"
METRICS_PATH = MODEL_DIR / f"{MODEL_VERSION}_metrics.json"

FINAL_TRAIN_SEASONS = range(2014, 2024)  # 2014 to 2023
CHOSEN_MIN_LEAF = 50
CHOSEN_MAX_DEPTH = 6

pd.set_option("display.width", 200)


def to_plain(value):
    """Make numbers JSON and MongoDB friendly: plain floats, NaN becomes None."""
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def grid_only_logistic():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def train_and_evaluate(df):
    train = df[df["season"].isin(FINAL_TRAIN_SEASONS)]
    test = df[df["season"].isin(baseline.TEST_SEASONS)]
    if train.empty or test.empty:
        raise ValueError("Train or test set is empty. Check the seasons in training_features.")

    forest = tune.make_forest(CHOSEN_MIN_LEAF, CHOSEN_MAX_DEPTH)
    forest.fit(train[ALL_FEATURE_COLUMNS], train[LABEL_COLUMN])
    p_test = forest.predict_proba(test[ALL_FEATURE_COLUMNS])[:, 1]

    logistic = grid_only_logistic()
    logistic.fit(train[["grid_effective"]], train[LABEL_COLUMN])
    p_logistic = logistic.predict_proba(test[["grid_effective"]])[:, 1]

    score_rule = -test["grid_effective"].to_numpy()
    zeros = np.zeros(len(test), dtype=int)
    rows = [
        baseline.evaluate("always no podium", test, zeros, zeros, rank=False),
        baseline.evaluate(
            "rule: grid <= 3", test, (test["grid_effective"] <= 3).astype(int), score_rule
        ),
        baseline.evaluate(
            "logistic: grid only", test, (p_logistic >= 0.5).astype(int), p_logistic, p_logistic
        ),
        baseline.evaluate(
            "random forest (0.5)", test, (p_test >= 0.5).astype(int), p_test, p_test
        ),
        baseline.evaluate(
            "random forest (top 3 per race)", test, importlib.import_module("05_random_forest").top3_flags(test, p_test), p_test, p_test
        ),
    ]
    metrics = pd.DataFrame(rows).set_index("model")

    auc_ci, hit_ci = tune.race_bootstrap(test, p_test, score_rule)

    bins = pd.cut(p_test, [0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0], include_lowest=True)
    calibration = (
        pd.DataFrame({"predicted": p_test, "actual": test[LABEL_COLUMN].to_numpy(), "bin": bins})
        .groupby("bin", observed=True)
        .agg(rows=("actual", "size"), mean_predicted=("predicted", "mean"), actual_rate=("actual", "mean"))
    )

    info = {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_races": test[["season", "round"]].drop_duplicates().shape[0],
        "auc_gain_vs_rule_95ci": [float(auc_ci[0]), float(auc_ci[1])],
        "top3_gain_vs_rule_95ci": [float(hit_ci[0]), float(hit_ci[1])],
    }
    return forest, metrics, calibration, info


def main():
    force = "--force" in sys.argv
    if MODEL_PATH.exists() and not force:
        sys.exit(
            f"{MODEL_PATH.name} already exists, so the sealed test was already used.\n"
            "Running again to chase better scores would leak the test set. Use --force "
            "only if the previous run crashed before printing results."
        )

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
    if not uri or uri.startswith("replace-with"):
        sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[db_name]
        df = baseline.load_features(db)
        if df.empty:
            sys.exit("No rows in training_features. Run ml/03_build_training_features.py first.")

        forest, metrics, calibration, info = train_and_evaluate(df)

        print(f"Final model trained on {info['train_rows']} rows (2014-2023).")
        print(f"Sealed test: {info['test_rows']} rows, {info['test_races']} races (2024-2025).\n")
        print(metrics.round(3).to_string())
        print("\nForest minus rule, 95% interval over resampled test races:")
        print(f"  ROC-AUC gain:           [{info['auc_gain_vs_rule_95ci'][0]:+.3f}, {info['auc_gain_vs_rule_95ci'][1]:+.3f}]")
        print(f"  Correct top-3 per race: [{info['top3_gain_vs_rule_95ci'][0]:+.3f}, {info['top3_gain_vs_rule_95ci'][1]:+.3f}]")
        print("\nCalibration on the test seasons:")
        print(calibration.round(3).to_string())

        params = forest[-1].get_params()
        record = to_plain(
            {
                "model_version": MODEL_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_version": baseline.DATASET_VERSION,
                "features": ALL_FEATURE_COLUMNS,
                "train_seasons": [FINAL_TRAIN_SEASONS.start, FINAL_TRAIN_SEASONS.stop - 1],
                "test_seasons": [baseline.TEST_SEASONS.start, baseline.TEST_SEASONS.stop - 1],
                "data_cutoff": str(df["season"].max()) + " season (last race in training_features)",
                "hyperparameters": {
                    key: params[key]
                    for key in ("n_estimators", "min_samples_leaf", "max_depth", "max_features", "random_state")
                },
                "sklearn_version": sklearn.__version__,
                "artifact_path": f"ml/models/{MODEL_PATH.name}",
                "test_metrics": metrics.to_dict(orient="index"),
                "test_gain_vs_rule": {
                    "roc_auc_95ci": info["auc_gain_vs_rule_95ci"],
                    "top3_per_race_95ci": info["top3_gain_vs_rule_95ci"],
                },
                "calibration": calibration.reset_index(drop=True).to_dict(orient="records"),
            }
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(forest, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"\nSaved {MODEL_PATH.name} and {METRICS_PATH.name} in ml/models/")

        registry = db["model_registry"]
        registry.create_index("model_version", unique=True, name="uniq_model_version")
        registry.update_one({"model_version": MODEL_VERSION}, {"$set": record}, upsert=True)
        print(f"Recorded {MODEL_VERSION} in the model_registry collection.")

    except PyMongoError as error:
        print(f"Database error: {type(error).__name__}")
        print(str(error)[:400])
        print("If the model files were already saved, only the registry step failed.")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()