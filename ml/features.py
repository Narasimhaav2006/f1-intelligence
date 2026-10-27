"""
Pre-race feature building for the podium model.

These are pure functions: they take DataFrames and return DataFrames, and they
never touch MongoDB or the network. That makes them easy to test.

Two groups of features:
  1. Base features from the starting grid and qualifying (build_base_features).
  2. Form features from earlier races (add_form_features). Every form feature
     uses only races that happened strictly BEFORE the race it describes.
"""

import numpy as np
import pandas as pd

LABEL_COLUMN = "podium"
FORM_WINDOW = 5  # how many previous races count as "recent form"

IDENTIFIER_COLUMNS = [
    "season", "round", "date", "circuit_id", "driver_id", "constructor_id",
]

# Base features. Everything here is known before the race starts.
FEATURE_COLUMNS = [
    "grid_effective",
    "grid_is_zero",
    "qual_position",
    "has_qualifying",
    "last_quali_session",
    "quali_gap_pct",
]

# Form features. Each is built only from earlier races.
FORM_FEATURE_COLUMNS = [
    "driver_podium_rate_5",
    "driver_avg_finish_5",
    "team_podium_rate_5",
    "driver_circuit_starts",
    "driver_circuit_podium_rate",
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + FORM_FEATURE_COLUMNS

# Known only after the race. These must never appear in the feature lists.
POST_RACE_COLUMNS = [
    "position", "points", "laps", "status",
    "finish_time_millis", "fastest_lap_rank", "finish_position", LABEL_COLUMN,
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
    and the podium label. Form features are added by add_form_features."""
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


def _previous_races_mean(series):
    """Mean of the previous FORM_WINDOW values, NOT including the current one.
    shift(1) moves every value down one row, so a row only sees earlier rows."""
    return series.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()


def add_form_features(df):
    """Add form features. The input needs these columns: date, season, round,
    driver_id, constructor_id, circuit_id, podium, finish_position.
    The result is sorted oldest first. Row order of the input does not matter."""
    df = df.sort_values(["date", "driver_id"]).reset_index(drop=True)

    # Driver form: the driver's previous starts, across seasons.
    by_driver = df.groupby("driver_id")
    df["driver_podium_rate_5"] = by_driver[LABEL_COLUMN].transform(_previous_races_mean)
    df["driver_avg_finish_5"] = by_driver["finish_position"].transform(_previous_races_mean)

    # Circuit history: the driver's earlier starts at this same circuit.
    by_circuit = df.groupby(["driver_id", "circuit_id"])[LABEL_COLUMN]
    df["driver_circuit_starts"] = (
        by_circuit.transform(lambda s: s.shift(1).expanding().count()).astype(int)
    )
    df["driver_circuit_podium_rate"] = by_circuit.transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Team form: first one number per team per race, then the previous races.
    team_race = (
        df.groupby(["constructor_id", "season", "round", "date"], as_index=False)[LABEL_COLUMN]
        .mean()
        .rename(columns={LABEL_COLUMN: "team_podium_share"})
        .sort_values(["constructor_id", "date"])
        .reset_index(drop=True)
    )
    team_race["team_podium_rate_5"] = team_race.groupby("constructor_id")[
        "team_podium_share"
    ].transform(_previous_races_mean)
    df = df.merge(
        team_race[["constructor_id", "season", "round", "team_podium_rate_5"]],
        on=["constructor_id", "season", "round"],
        how="left",
        validate="many_to_one",
    )
    return df


def build_features(results, quals, races):
    """The full feature table: identifiers, all features, and the label."""
    base = build_base_features(results, quals, races)
    finish = results[DRIVER_KEYS + ["position"]].rename(
        columns={"position": "finish_position"}
    )
    df = base.merge(finish, on=DRIVER_KEYS, how="left", validate="one_to_one")
    df = add_form_features(df)
    # finish_position was only needed to build the history. It is dropped here.
    return df[IDENTIFIER_COLUMNS + ALL_FEATURE_COLUMNS + [LABEL_COLUMN]]