import numpy as np
import pandas as pd
import pytest

from src.data_preparation import (
    LONG_LAG_DAYS,
    SHORT_LAG_DAYS,
    _add_daily_counts,
    _add_days_since_first_case,
    _add_lagged_case_features,
    _compute_mortality_rate,
    _drop_rows_with_excessive_missing_values,
    _enforce_logical_consistency,
    _extract_date_features,
    _fill_missing_values,
    _filter_low_case_counts,
    _remove_duplicates,
    _remove_outliers_iqr,
    _sort_by_location_and_date,
    _split_numerical_and_categorical_columns,
    _standardize_country_names,
    clean_data,
)


def _build_location_series(days, country="Testland", province="unknown", start_confirmed=0):
    """Build a single location's contiguous daily series with steadily growing counts."""
    dates = pd.date_range("2020-03-01", periods=days, freq="D")
    confirmed = np.arange(days) * 50 + start_confirmed
    return pd.DataFrame(
        {
            "Province/State": province,
            "Country/Region": country,
            "Lat": 10.0,
            "Long": 20.0,
            "Date": dates,
            "Confirmed": confirmed,
            "Deaths": (confirmed * 0.02).astype(int),
            "Recovered": (confirmed * 0.5).astype(int),
        }
    )


def test_remove_duplicates_drops_exact_repeats():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    result = _remove_duplicates(df)

    assert len(result) == 2


def test_drop_rows_with_excessive_missing_values():
    df = pd.DataFrame(
        {
            "a": [1, np.nan, np.nan],
            "b": [1, np.nan, 2],
            "c": [1, np.nan, 3],
        }
    )
    # Row 0: 0% missing (kept), row 1: 100% missing, row 2: 1/3 = 33% missing (both exceed 30%)

    result = _drop_rows_with_excessive_missing_values(df, max_missing_ratio=0.3)

    assert list(result.index) == [0]


def test_split_numerical_and_categorical_columns():
    df = pd.DataFrame({"num": [1, 2], "flt": [1.0, 2.0], "cat": ["a", "b"]})

    numerical, categorical = _split_numerical_and_categorical_columns(df)

    assert set(numerical) == {"num", "flt"}
    assert categorical == ["cat"]


def test_fill_missing_values_uses_zero_and_unknown():
    df = pd.DataFrame({"num": [1.0, np.nan], "cat": ["a", np.nan]})

    result = _fill_missing_values(df, numerical_columns=["num"], categorical_columns=["cat"])

    assert result["num"].tolist() == [1.0, 0.0]
    assert result["cat"].tolist() == ["a", "unknown"]


def test_remove_outliers_iqr_drops_extreme_value():
    df = pd.DataFrame({"value": [10, 11, 9, 10, 12, 1000]})

    result = _remove_outliers_iqr(df, numerical_columns=["value"])

    assert 1000 not in result["value"].tolist()


def test_enforce_logical_consistency_drops_deaths_above_confirmed():
    df = pd.DataFrame({"Confirmed": [100, 50], "Deaths": [10, 60]})

    result = _enforce_logical_consistency(df)

    assert result["Confirmed"].tolist() == [100]


def test_standardize_country_names_normalizes_aliases():
    df = pd.DataFrame({"Country/Region": ["US", "UK", "U.S.", "France"]})

    result = _standardize_country_names(df)

    assert result["Country/Region"].tolist() == [
        "United States",
        "United Kingdom",
        "United States",
        "France",
    ]


def test_extract_date_features_derives_expected_values():
    # 2020-03-15 is a Sunday: dayofweek=6, month=3, dayofyear=75
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-03-15"])})

    result = _extract_date_features(df)

    assert result["Day_of_Week"].tolist() == [6]
    assert result["Month"].tolist() == [3]
    assert result["Day_of_Year"].tolist() == [75]


def test_compute_mortality_rate_is_a_percentage():
    df = pd.DataFrame({"Deaths": [10, 0], "Confirmed": [100, 0]})

    result = _compute_mortality_rate(df)

    # The zero-confirmed row produces inf -> dropped by the NaN filter
    assert result["Mortality_Rate"].tolist() == [10.0]


def test_sort_by_location_and_date_orders_each_series():
    df = pd.DataFrame(
        {
            "Country/Region": ["B", "A", "A"],
            "Province/State": ["unknown", "unknown", "unknown"],
            "Date": pd.to_datetime(["2020-03-01", "2020-03-02", "2020-03-01"]),
        }
    )

    result = _sort_by_location_and_date(df)

    assert result["Country/Region"].tolist() == ["A", "A", "B"]
    assert result["Date"].tolist() == pd.to_datetime(
        ["2020-03-01", "2020-03-02", "2020-03-01"]
    ).tolist()


def test_add_daily_counts_differences_the_cumulative_series():
    df = _build_location_series(days=4)

    result = _add_daily_counts(df)

    # Cumulative confirmed grows by 50/day; the first day falls back to its own total.
    assert result["New_Confirmed"].tolist() == [0.0, 50.0, 50.0, 50.0]


def test_add_daily_counts_clips_downward_revisions_to_zero():
    df = _build_location_series(days=3)
    df.loc[2, "Confirmed"] = 10  # simulate a downward data revision

    result = _add_daily_counts(df)

    assert (result["New_Confirmed"] >= 0).all()


def test_add_daily_counts_does_not_bleed_across_locations():
    df = pd.concat(
        [_build_location_series(days=3, country="A"), _build_location_series(days=3, country="B")],
        ignore_index=True,
    )

    result = _add_daily_counts(df)

    # Each location's first day restarts the series rather than differencing the previous one.
    first_days = result.groupby("Country/Region").head(1)
    assert first_days["New_Confirmed"].tolist() == [0.0, 0.0]


def test_add_days_since_first_case_counts_from_first_nonzero_confirmed():
    df = _build_location_series(days=4)  # confirmed: 0, 50, 100, 150 -> first case on day 1

    result = _add_days_since_first_case(df)

    assert result["Days_Since_First_Case"].tolist() == [-1, 0, 1, 2]


def test_add_lagged_case_features_shifts_by_expected_offsets():
    df = _build_location_series(days=30)
    df = _add_daily_counts(df)

    result = _add_lagged_case_features(df)

    short_lag = f"New_Confirmed_Lag_{SHORT_LAG_DAYS}"
    long_lag = f"New_Confirmed_Lag_{LONG_LAG_DAYS}"

    # Lagged columns are undefined until enough history exists.
    assert result[short_lag].head(SHORT_LAG_DAYS).isna().all()
    assert result[long_lag].head(LONG_LAG_DAYS).isna().all()
    assert result[short_lag].iloc[SHORT_LAG_DAYS] == result["New_Confirmed"].iloc[0]
    assert result[long_lag].iloc[LONG_LAG_DAYS] == result["New_Confirmed"].iloc[0]
    assert f"Smoothed_New_Confirmed_Lag_{LONG_LAG_DAYS}" in result.columns


def test_filter_low_case_counts_drops_small_denominators():
    df = pd.DataFrame({"Confirmed": [5, 99, 100, 5000]})

    result = _filter_low_case_counts(df, min_confirmed=100)

    assert result["Confirmed"].tolist() == [100, 5000]


def test_clean_data_end_to_end_produces_leak_free_features():
    df = pd.concat(
        [
            _build_location_series(days=60, country="US"),
            _build_location_series(days=60, country="France"),
        ],
        ignore_index=True,
    )
    # Duplicate a row to exercise deduplication
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_data(df)

    assert not cleaned.isnull().any().any()
    assert (cleaned["Deaths"] <= cleaned["Confirmed"]).all()
    assert (cleaned["Confirmed"] >= 100).all()
    assert "United States" in cleaned["Country/Region"].tolist()
    assert "US" not in cleaned["Country/Region"].tolist()
    assert {
        "Day_of_Week",
        "Month",
        "Day_of_Year",
        "Mortality_Rate",
        "New_Confirmed",
        "Days_Since_First_Case",
        f"New_Confirmed_Lag_{LONG_LAG_DAYS}",
    }.issubset(cleaned.columns)
