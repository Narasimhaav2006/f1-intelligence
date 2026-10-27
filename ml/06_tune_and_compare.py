"""
Small, honest model comparison on the VALIDATION seasons.
Tries a few Random Forest settings and measures, with a race-level bootstrap,
how much each one beats the simple rule "podium if grid <= 3".
The sealed test seasons are not used here.

Run from the project root with the virtual environment active:
    python ml/06_tune_and_compare.py
"""

import importlib
import itertools
import os
import sys

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from features import ALL_FEATURE_COLUMNS, LABEL_COLUMN

baseline = importlib.import_module("04_baseline")

LEAF_SIZES = [10, 25, 50]
MAX_DEPTHS = [None, 6]
N_BOOTSTRAP = 1000

pd.set_option("display.width", 250)


def make_forest(min_samples_leaf, max_depth):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def hits_per_race(df, score):
    """Real podium finishers among the 3 highest-scored drivers, for each race."""
    tmp = df[["season", "round", LABEL_COLUMN]].copy()
    tmp["score"] = np.asarray(score)
    tmp["rank"] = tmp.groupby(["season", "round"])["score"].rank(method="first", ascending=False)
    return tmp[tmp["rank"] <= 3].groupby(["season", "round"])[LABEL_COLUMN].sum()


def race_bootstrap(valid, score_model, score_rule, n=N_BOOTSTRAP, seed=0):
    """Resample whole races with replacement. Returns 95% intervals for the
    difference (model minus rule) in ROC-AUC and in correct picks per race."""
    rng = np.random.default_rng(seed)
    y = valid[LABEL_COLUMN].to_numpy()
    groups = valid.groupby(["season", "round"]).indices
    keys = list(groups.keys())
    races = [groups[k] for k in keys]

    hits_model = hits_per_race(valid, score_model)
    hits_rule = hits_per_race(valid, score_rule)
    hit_diff = np.array([hits_model.loc[k] - hits_rule.loc[k] for k in keys])

    auc_diffs, hit_diffs = [], []
    for _ in range(n):
        pick = rng.integers(0, len(races), len(races))
        idx = np.concatenate([races[i] for i in pick])
        auc_diffs.append(
            roc_auc_score(y[idx], score_model[idx]) - roc_auc_score(y[idx], score_rule[idx])
        )
        hit_diffs.append(hit_diff[pick].mean())
    return np.percentile(auc_diffs, [2.5, 97.5]), np.percentile(hit_diffs, [2.5, 97.5])


def interval(values):
    return f"[{values[0]:+.3f}, {values[1]:+.3f}]"


def run_comparison(df):
    train, valid, _sealed_test = baseline.split_by_season(df)
    y_train = train[LABEL_COLUMN]
    y_valid = valid[LABEL_COLUMN]
    score_rule = -valid["grid_effective"].to_numpy()

    rule_line = (
        f"Rule 'grid <= 3': ROC-AUC {roc_auc_score(y_valid, score_rule):.3f}, "
        f"correct picks per race {hits_per_race(valid, score_rule).mean():.3f}"
    )

    rows = []
    for leaf, depth in itertools.product(LEAF_SIZES, MAX_DEPTHS):
        model = make_forest(leaf, depth)
        model.fit(train[ALL_FEATURE_COLUMNS], y_train)
        p_valid = model.predict_proba(valid[ALL_FEATURE_COLUMNS])[:, 1]
        p_train = model.predict_proba(train[ALL_FEATURE_COLUMNS])[:, 1]
        auc_ci, hit_ci = race_bootstrap(valid, p_valid, score_rule)
        rows.append(
            {
                "min_leaf": leaf,
                "max_depth": "none" if depth is None else depth,
                "val_auc": roc_auc_score(y_valid, p_valid),
                "val_avg_prec": average_precision_score(y_valid, p_valid),
                "val_brier": brier_score_loss(y_valid, p_valid),
                "val_top3": hits_per_race(valid, p_valid).mean(),
                "train_avg_prec": average_precision_score(y_train, p_train),
                "auc_gain_95ci": interval(auc_ci),
                "top3_gain_95ci": interval(hit_ci),
            }
        )
    return rule_line, pd.DataFrame(rows)


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

    rule_line, table = run_comparison(df)
    print(rule_line)
    print("\nRandom Forest settings on validation (2022-2023):")
    print(table.round(3).to_string(index=False))
    print("\nGain columns = forest minus rule, 95% interval over resampled races.")
    print("If an interval includes 0, the forest is not clearly better than the rule.")


if __name__ == "__main__":
    main()