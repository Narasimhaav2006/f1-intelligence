"""
Tests for the form features in ml/features.py, including leakage checks.
Run from the project root:  python -m pytest ml/tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    FORM_FEATURE_COLUMNS,
    POST_RACE_COLUMNS,
    add_form_features,
    build_features,
)


def make_history():
    """Two drivers (a, b) of one team (x) over four races, oldest first."""
    rows = []
    a_finish = [1, 5, 3, 2]
    b_finish = [4, 6, 7, 8]
    circuits = ["c1", "c2", "c2", "c1"]
    for i in range(4):
        for driver, finish in (("a", a_finish[i]), ("b", b_finish[i])):
            rows.append(
                {
                    "season": 2024,
                    "round": i + 1,
                    "date": pd.Timestamp(f"2024-0{i + 1}-10"),
                    "circuit_id": circuits[i],
                    "driver_id": driver,
                    "constructor_id": "x",
                    "finish_position": finish,
                    "podium": int(finish <= 3),
                }
            )
    return pd.DataFrame(rows)


def row(df, driver, round_number):
    return df[(df["driver_id"] == driver) & (df["round"] == round_number)].iloc[0]


def test_driver_form_uses_only_earlier_races():
    out = add_form_features(make_history())
    assert pd.isna(row(out, "a", 1)["driver_podium_rate_5"])
    assert row(out, "a", 2)["driver_podium_rate_5"] == pytest.approx(1.0, abs=1e-6)
    assert row(out, "a", 3)["driver_podium_rate_5"] == pytest.approx(0.5, abs=1e-6)
    assert row(out, "a", 4)["driver_podium_rate_5"] == pytest.approx(2 / 3, abs=1e-6)
    assert pd.isna(row(out, "a", 1)["driver_avg_finish_5"])
    assert row(out, "a", 3)["driver_avg_finish_5"] == pytest.approx(3.0, abs=1e-6)
    assert row(out, "a", 4)["driver_avg_finish_5"] == pytest.approx(3.0, abs=1e-6)


def test_circuit_history_counts_only_earlier_visits():
    out = add_form_features(make_history())
    assert row(out, "a", 1)["driver_circuit_starts"] == 0
    assert pd.isna(row(out, "a", 1)["driver_circuit_podium_rate"])
    assert row(out, "a", 3)["driver_circuit_starts"] == 1
    assert row(out, "a", 3)["driver_circuit_podium_rate"] == pytest.approx(0.0, abs=1e-6)
    assert row(out, "a", 4)["driver_circuit_starts"] == 1
    assert row(out, "a", 4)["driver_circuit_podium_rate"] == pytest.approx(1.0, abs=1e-6)


def test_team_form_does_not_use_the_current_race_or_the_teammate_in_it():
    out = add_form_features(make_history())
    assert pd.isna(row(out, "a", 1)["team_podium_rate_5"])
    assert row(out, "a", 2)["team_podium_rate_5"] == pytest.approx(0.5, abs=1e-6)
    assert row(out, "b", 2)["team_podium_rate_5"] == pytest.approx(0.5, abs=1e-6)
    assert row(out, "a", 3)["team_podium_rate_5"] == pytest.approx(0.25, abs=1e-6)
    assert row(out, "a", 4)["team_podium_rate_5"] == pytest.approx(1 / 3, abs=1e-6)


def test_only_the_previous_five_races_count():
    finishes = [1, 1, 1, 1, 1, 9, 9]  # podium, podium, podium, podium, podium, no, no
    rows = [
        {
            "season": 2024, "round": i + 1, "date": pd.Timestamp(f"2024-0{i + 1}-10"),
            "circuit_id": "c1", "driver_id": "z", "constructor_id": "t",
            "finish_position": f, "podium": int(f <= 3),
        }
        for i, f in enumerate(finishes)
    ]
    out = add_form_features(pd.DataFrame(rows))
    # Race 7 looks at races 2 to 6: four podiums and one non-podium.
    assert row(out, "z", 7)["driver_podium_rate_5"] == pytest.approx(0.8, abs=1e-6)


def test_changing_a_races_own_result_does_not_change_its_own_features():
    original = add_form_features(make_history())
    changed_input = make_history()
    last_race_of_a = (changed_input["driver_id"] == "a") & (changed_input["round"] == 4)
    changed_input.loc[last_race_of_a, ["podium", "finish_position"]] = [0, 20]
    changed = add_form_features(changed_input)
    pd.testing.assert_frame_equal(original[FORM_FEATURE_COLUMNS], changed[FORM_FEATURE_COLUMNS])


def test_earlier_results_do_change_later_features():
    changed_input = make_history()
    first_race_of_a = (changed_input["driver_id"] == "a") & (changed_input["round"] == 1)
    changed_input.loc[first_race_of_a, "podium"] = 0
    changed = add_form_features(changed_input)
    assert row(changed, "a", 2)["driver_podium_rate_5"] == pytest.approx(0.0, abs=1e-6)


def test_input_row_order_does_not_matter():
    ordered = add_form_features(make_history())
    shuffled = add_form_features(make_history().sample(frac=1, random_state=0))
    pd.testing.assert_frame_equal(ordered, shuffled)


def test_build_features_returns_the_expected_columns():
    results = pd.DataFrame(
        {
            "season": 2024,
            "round": [1, 1, 2, 2],
            "driver_id": ["a", "b", "a", "b"],
            "constructor_id": ["x", "y", "x", "y"],
            "grid": [1, 2, 2, 1],
            "position": [1, 2, 3, 1],
            "started": True,
        }
    )
    quals = pd.DataFrame(
        {
            "season": 2024,
            "round": [1, 1, 2, 2],
            "driver_id": ["a", "b", "a", "b"],
            "position": [1, 2, 2, 1],
            "q1_seconds": [90.0, 90.5, 90.4, 90.0],
            "q2_seconds": np.nan,
            "q3_seconds": np.nan,
        }
    )
    races = pd.DataFrame(
        {
            "season": 2024,
            "round": [1, 2],
            "date": ["2024-03-02", "2024-03-09"],
            "circuit_id": ["c1", "c2"],
        }
    )
    out = build_features(results, quals, races)
    assert set(ALL_FEATURE_COLUMNS) <= set(out.columns)
    assert "finish_position" not in out.columns
    assert out[out["round"] == 1]["driver_podium_rate_5"].isna().all()
    assert (out[out["round"] == 2]["driver_podium_rate_5"] == 1.0).all()


def test_no_post_race_column_is_a_feature():
    assert not set(ALL_FEATURE_COLUMNS) & set(POST_RACE_COLUMNS)