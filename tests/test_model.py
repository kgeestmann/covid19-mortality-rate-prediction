import numpy as np
import pandas as pd
import pytest

from src.model import (
    COLUMNS_EXCLUDED_FROM_FEATURES,
    _evaluate,
    _split_features_and_target,
    _temporal_train_test_split,
)


def _build_cleaned_frame(days=10):
    """Build a frame shaped like clean_data's output."""
    dates = pd.date_range("2020-03-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Country/Region": "Testland",
            "Province/State": "unknown",
            "Lat": 10.0,
            "Long": 20.0,
            "Date": dates,
            "Confirmed": np.arange(days) * 100 + 100,
            "Deaths": np.arange(days) * 2,
            "Recovered": np.arange(days) * 50,
            "New_Confirmed": 100.0,
            "New_Deaths": 2.0,
            "New_Confirmed_Lag_14": 90.0,
            "Days_Since_First_Case": np.arange(days),
            "Day_of_Week": dates.dayofweek,
            "Month": dates.month,
            "Day_of_Year": dates.dayofyear,
            "Mortality_Rate": np.linspace(1.0, 2.0, days),
        }
    )


def test_split_features_and_target_excludes_same_day_counts():
    df = _build_cleaned_frame()

    X, y = _split_features_and_target(df)

    # Nothing that encodes the target's numerator or denominator survives into X.
    assert not set(COLUMNS_EXCLUDED_FROM_FEATURES) & set(X.columns)
    assert "New_Confirmed_Lag_14" in X.columns
    assert "Days_Since_First_Case" in X.columns
    assert y.tolist() == df["Mortality_Rate"].tolist()


def test_temporal_split_puts_later_dates_in_test():
    df = _build_cleaned_frame(days=10)

    train_df, test_df = _temporal_train_test_split(df, test_size=0.2)

    assert train_df["Date"].max() < test_df["Date"].min()


def test_temporal_split_shares_no_dates_between_sets():
    df = _build_cleaned_frame(days=10)

    train_df, test_df = _temporal_train_test_split(df, test_size=0.2)

    assert not set(train_df["Date"]) & set(test_df["Date"])
    assert len(train_df) + len(test_df) == len(df)


def test_evaluate_computes_expected_metrics():
    y_test = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    metrics = _evaluate(y_test, y_pred)

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["mse"] == pytest.approx(1 / 3)
    assert set(metrics.keys()) == {"mae", "mse", "r2"}
