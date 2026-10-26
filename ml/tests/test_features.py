"""
Tests for ml/features.py using tiny hand-made data.
Run from the project root:  python -m pytest ml/tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the ml folder importable, so "import features" works from this test file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import (  # noqa: E402
    FEATURE_COLUMNS,
    POST_RACE_COLUMNS,
    build_base_features,
)


def make_inputs():
    """One made-up race. Driver e did not start. Driver d starts from the pit
    lane (grid 0) and has no qualifying row. Driver c was out after Q2."""
    results = pd.DataFrame(
        {
            "season": 2024,
            "round": 1,
            "driver_id": ["a", "b", "c", "d", "e"],
            "constructor_id": ["x", "x", "y", "y", "z"],
            "grid": [1, 2, 3, 0, 4],
            "position": [1, 2, 3, 4, 5],
            "started": [True, True, True, True, False],
        }
    )
    quals = pd.DataFrame(
        {
            "season": 2024,
            "round": 1,
            "driver_id": ["a", "b", "c"],
            "position": [1, 2, 3],
            "q1_seconds": [90.0, 90.5, 91.0],
            "q2_seconds": [89.0, 89.5, 89.89],
            "q3_seconds": [88.0, 88.44, np.nan],
        }
    )
    races = pd.DataFrame(
        {"season": [2024], "round": [1], "date": ["2024-03-02"], "circuit_id": ["bahrain"]}
    )
    return results, quals, races


def build():
    return build_base_features(*make_inputs()).set_index("driver_id")


def test_only_starters_are_kept():
    out = build()
    assert len(out) == 4
    assert "e" not in out.index


def test_podium_label():
    out = build()
    assert out["podium"].to_dict() == {"a": 1, "b": 1, "c": 1, "d": 0}


def test_pit_lane_start_goes_behind_the_last_gridded_car():
    out = build()
    assert out.loc["d", "grid_is_zero"] == 1
    assert out.loc["d", "grid_effective"] == 4
    assert out.loc["a", "grid_is_zero"] == 0
    assert out.loc["a", "grid_effective"] == 1


def test_missing_qualifying_stays_missing():
    out = build()
    assert out.loc["d", "has_qualifying"] == 0
    assert pd.isna(out.loc["d", "qual_position"])
    assert out.loc["d", "last_quali_session"] == 0
    assert pd.isna(out.loc["d", "quali_gap_pct"])


def test_gap_is_measured_in_the_last_session_reached():
    out = build()
    assert out.loc["a", "quali_gap_pct"] == pytest.approx(0.0, abs=1e-6)
    assert out.loc["b", "quali_gap_pct"] == pytest.approx(0.5, abs=1e-6)
    assert out.loc["c", "quali_gap_pct"] == pytest.approx(1.0, abs=1e-6)
    assert out["last_quali_session"].to_dict() == {"a": 3, "b": 3, "c": 2, "d": 0}


def test_duplicate_qualifying_rows_are_rejected():
    results, quals, races = make_inputs()
    quals = pd.concat([quals, quals.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        build_base_features(results, quals, races)


def test_no_post_race_column_is_a_feature():
    assert not set(FEATURE_COLUMNS) & set(POST_RACE_COLUMNS)