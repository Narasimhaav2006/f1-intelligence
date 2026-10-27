"""
Baseline models for the podium prediction.
Evaluated on the VALIDATION seasons only. The test seasons stay untouched
until the final model has been chosen.

Run from the project root with the virtual environment active:
    python ml/04_baseline.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import ALL_FEATURE_COLUMNS, LABEL_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATASET_VERSION = "v1"
TRAIN_SEASONS = range(2014, 2022)       # 2014 to 2021
VALIDATION_SEASONS = range(2022, 2024)  # 2022 to 2023
TEST_SEASONS = range(2024, 2026)        # 2024 to 2025: do not look until the end

pd.set_option("display.width", 200)


def load_features(db):
    fields = ["season", "round", "driver_id"] + ALL_FEATURE_COLUMNS + [LABEL_COLUMN]
    projection = {field: 1 for field in fields}
    projection["_id"] = 0
    rows = list(db["training_features"].find({"dataset_version": DATASET_VERSION}, projection))
    df = pd.DataFrame(rows, columns=fields)
    df[ALL_FEATURE_COLUMNS] = df[ALL_FEATURE_COLUMNS].astype(float)
    return df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)


def split_by_season(df):
    """Chronological split: earlier seasons train, later seasons evaluate."""
    train = df[df["season"].isin(TRAIN_SEASONS)]
    valid = df[df["season"].isin(VALIDATION_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    return train, valid, test


def race_top3_hits(df, score):
    """Average number of real podium finishers among the 3 highest-scored
    drivers of each race (3.0 would be perfect)."""
    tmp = df[["season", "round", LABEL_COLUMN]].copy()
    tmp["score"] = np.asarray(score)
    top3 = tmp.sort_values("score", ascending=False, kind="stable").groupby(
        ["season", "round"]
    ).head(3)
    return top3.groupby(["season", "round"])[LABEL_COLUMN].sum().mean()


def evaluate(name, df, predicted, score, probability=None, rank=True):
    y = df[LABEL_COLUMN]
    return {
        "model": name,
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "avg_precision": average_precision_score(y, score),
        "brier": brier_score_loss(y, probability) if probability is not None else np.nan,
        "top3_hits": race_top3_hits(df, score) if rank else np.nan,
    }


def run_baselines(df):
    train, valid, test = split_by_season(df)

    sizes = pd.DataFrame(
        {
            "rows": [len(train), len(valid), len(test)],
            "races": [t[["season", "round"]].drop_duplicates().shape[0] for t in (train, valid, test)],
            "podium_rate": [t[LABEL_COLUMN].mean() for t in (train, valid, test)],
        },
        index=["train 2014-2021", "validation 2022-2023", "test 2024-2025 (sealed)"],
    )

    rows = []
    # 1. Always predict "no podium": shows why accuracy alone misleads.
    zeros = np.zeros(len(valid), dtype=int)
    rows.append(evaluate("always no podium", valid, zeros, zeros, rank=False))

    # 2. Rule: podium if the driver starts in the top 3. Nothing is learned.
    rule = (valid["grid_effective"] <= 3).astype(int)
    rows.append(evaluate("rule: grid <= 3", valid, rule, -valid["grid_effective"]))

    # 3. Logistic regression on the starting grid only (the first trained model).
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train[["grid_effective"]], train[LABEL_COLUMN])
    probability = model.predict_proba(valid[["grid_effective"]])[:, 1]
    rows.append(
        evaluate("logistic: grid only", valid, (probability >= 0.5).astype(int), probability, probability)
    )

    return sizes, pd.DataFrame(rows).set_index("model")


def main():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
    if not uri or uri.startswith("replace-with"):
        sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        df = load_features(client[db_name])
    except PyMongoError as error:
        print(f"Database error: {type(error).__name__}")
        print(str(error)[:400])
        sys.exit(1)
    finally:
        client.close()

    if df.empty:
        sys.exit("No rows in training_features. Run ml/03_build_training_features.py first.")

    sizes, metrics = run_baselines(df)
    print("Chronological split:")
    print(sizes.round(3).to_string())
    print("\nValidation results (2022-2023), threshold 0.5 for the logistic model:")
    print(metrics.round(3).to_string())


if __name__ == "__main__":
    main()