"""
Random Forest for the podium prediction, compared with the baselines.
Fits on the TRAIN seasons and evaluates on the VALIDATION seasons only.
The sealed test seasons are not used here.

Run from the project root with the virtual environment active:
    python ml/05_random_forest.py
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from features import ALL_FEATURE_COLUMNS, LABEL_COLUMN

# The baseline file name starts with a digit, so a normal "import" cannot be used.
baseline = importlib.import_module("04_baseline")

pd.set_option("display.width", 200)


def top3_flags(df, score):
    """1 for the three highest-scored drivers of each race, else 0."""
    tmp = df[["season", "round"]].copy()
    tmp["score"] = np.asarray(score)
    rank = tmp.groupby(["season", "round"])["score"].rank(method="first", ascending=False)
    return (rank <= 3).astype(int)


def build_forest():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),  # learned from training data only
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_forest(df):
    train, valid, _sealed_test = baseline.split_by_season(df)
    _, base_metrics = baseline.run_baselines(df)

    forest = build_forest()
    forest.fit(train[ALL_FEATURE_COLUMNS], train[LABEL_COLUMN])
    p_valid = forest.predict_proba(valid[ALL_FEATURE_COLUMNS])[:, 1]
    p_train = forest.predict_proba(train[ALL_FEATURE_COLUMNS])[:, 1]

    rows = [
        baseline.evaluate(
            "random forest (0.5)", valid, (p_valid >= 0.5).astype(int), p_valid, p_valid
        ),
        baseline.evaluate(
            "random forest (top 3 per race)", valid, top3_flags(valid, p_valid), p_valid, p_valid
        ),
        baseline.evaluate(
            "random forest on TRAIN data (0.5)", train, (p_train >= 0.5).astype(int), p_train, p_train
        ),
    ]
    metrics = pd.concat([base_metrics, pd.DataFrame(rows).set_index("model")])

    importance = pd.Series(
        forest[-1].feature_importances_, index=ALL_FEATURE_COLUMNS
    ).sort_values(ascending=False)

    bins = pd.cut(p_valid, [0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0], include_lowest=True)
    calibration = (
        pd.DataFrame(
            {"predicted": p_valid, "actual": valid[LABEL_COLUMN].to_numpy(), "bin": bins}
        )
        .groupby("bin", observed=True)
        .agg(rows=("actual", "size"), mean_predicted=("predicted", "mean"), actual_rate=("actual", "mean"))
    )
    return metrics, importance, calibration


def main():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
    if not uri or uri.startswith("replace-with"):
        sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        df = baseline.load_features(client[db_name])
    except PyMongoError as error:
        print(f"Database error: {type(error).__name__}")
        print(str(error)[:400])
        sys.exit(1)
    finally:
        client.close()

    if df.empty:
        sys.exit("No rows in training_features. Run ml/03_build_training_features.py first.")

    metrics, importance, calibration = run_forest(df)
    print("Validation results (2022-2023) unless a row says TRAIN:")
    print(metrics.round(3).to_string())
    print("\nCalibration on validation (does a predicted 30% really happen about 30% of the time?):")
    print(calibration.round(3).to_string())
    print("\nFeature importance (rough guide only):")
    print(importance.round(3).to_string())


if __name__ == "__main__":
    main()