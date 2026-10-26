"""
Pre-race feature building for the podium model.

These are pure functions: they take DataFrames and return DataFrames, and they
never touch MongoDB or the network. That makes them easy to test.
"""

import numpy as np
import pandas as pd

LABEL_COLUMN = "podium"

IDENTIFIER_COLUMNS = [
    "season", "round", "date", "circuit_id", "driver_id", "constructor_id",
]

# Columns the model may use. Everything here is known before the race starts.
FEATURE_COLUMNS = [
    "grid_effective",
    "grid_is_zero",
    "qual_position",
    "has_qualifying",
    "last_quali_session",
    "quali_gap_pct",
]

# Known only after the race. These must never appear in FEATURE_COLUMNS.
POST_RACE_COLUMNS = [
    "position", "points", "laps", "status",
    "finish_time_millis", "fastest_lap_rank", LABEL_COLUMN,
]

RACE_KEYS = ["season", "round"]
DRIVER_KEYS = RACE_KEYS + ["driver_id"]


def qualifying_summary(quals):
    """One row per driver per race: qualifying position, last session reached,
    and the gap (in percent) to the fastest time of that same session."""
    q = quals[DRIVER_KEYS + ["position", "q1_seconds", "q2_seconds", "q3_seconds"]].copy()
    q = q.rename(columns={"position": "qual_position"})
    q["last_quali_session"] = 0
    q["quali_gap_pct"] = np.nan

    # Go from Q1 to Q3 so a later session overwrites an earlier one.
    for session in (1, 2, 3):
        column = f"q{session}_seconds"
        fastest = q.groupby(RACE_KEYS)[column].transform("min")
        has_time = q[column].notna()
        gap = (q[column] - fastest) / fastest * 100
        q.loc[has_time, "last_quali_session"] = session
        q.loc[has_time, "quali_gap_pct"] = gap[has_time]

    return q[DRIVER_KEYS + ["qual_position", "last_quali_session", "quali_gap_pct"]]


def build_base_features(results, quals, races):
    """Build one row per driver who started a race, with pre-race features
    and the podium label. Rolling form features are added in a later step."""
    df = results[results["started"]].copy()

    df = df.merge(
        qualifying_summary(quals), on=DRIVER_KEYS, how="left", validate="one_to_one"
    )
    df = df.merge(
        races[RACE_KEYS + ["date", "circuit_id"]],
        on=RACE_KEYS, how="left", validate="many_to_one",
    )
    if df["date"].isna().any():
        raise ValueError("Some results have no matching race, so they have no date.")
    df["date"] = pd.to_datetime(df["date"])

    df["has_qualifying"] = df["qual_position"].notna().astype(int)
    df["last_quali_session"] = df["last_quali_session"].fillna(0).astype(int)

    # Grid 0 is treated as a start behind the last car that has a real grid slot.
    gridded = df["grid"].where(df["grid"] >= 1)
    last_gridded = gridded.groupby([df["season"], df["round"]]).transform("max")
    df["grid_is_zero"] = (df["grid"] == 0).astype(int)
    df["grid_effective"] = np.where(df["grid"] >= 1, df["grid"], last_gridded + 1)

    df[LABEL_COLUMN] = df["position"].isin([1, 2, 3]).astype(int)

    df = df.sort_values(["date", "driver_id"]).reset_index(drop=True)
    return df[IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN]]