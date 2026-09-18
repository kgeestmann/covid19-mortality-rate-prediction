import numpy as np
import pandas as pd

JHU_TIME_SERIES_BASE_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)
URL_CONFIRMED = JHU_TIME_SERIES_BASE_URL + "time_series_covid19_confirmed_global.csv"
URL_DEATHS = JHU_TIME_SERIES_BASE_URL + "time_series_covid19_deaths_global.csv"
URL_RECOVERED = JHU_TIME_SERIES_BASE_URL + "time_series_covid19_recovered_global.csv"

LOCATION_COLUMNS = ["Province/State", "Country/Region", "Lat", "Long"]
MERGE_KEYS = LOCATION_COLUMNS + ["Date"]

# Identifies a single reporting unit, i.e. one continuous time series.
LOCATION_KEY = ["Country/Region", "Province/State"]

MAX_MISSING_RATIO_PER_ROW = 0.3
COUNTRY_NAME_ALIASES = {"US": "United States", "UK": "United Kingdom", "U.S.": "United States"}

# Below this many cumulative cases the mortality rate is dominated by small-denominator
# noise (1 case / 1 death reads as 100% mortality).
MIN_CONFIRMED_CASES = 100

# Deaths lag infections by roughly 2-3 weeks, so case counts from that far back are the
# epidemiologically meaningful predictors of today's mortality rate.
SHORT_LAG_DAYS = 7
LONG_LAG_DAYS = 14
SMOOTHING_WINDOW_DAYS = 7

# IQR trimming is applied only to the target. Applying it to Lat/Long would delete whole
# countries, and applying it to case counts would delete every major outbreak.
OUTLIER_COLUMNS = ["Mortality_Rate"]


def _load_time_series(url, value_name):
    """Load one JHU wide-format time series CSV and reshape it to long format."""
    df = pd.read_csv(url)
    df = df.melt(id_vars=LOCATION_COLUMNS, var_name="Date", value_name=value_name)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    return df


def load_covid_data():
    """
    Loads and processes global COVID-19 data for confirmed cases, deaths, and recoveries.

    The function retrieves time-series data from the Johns Hopkins University repository,
    reshapes the data into a long format, and merges the datasets into a single DataFrame.

    Returns:
        pd.DataFrame: A merged DataFrame containing COVID-19 confirmed cases, deaths,
        and recoveries along with data such as location and date.
    """
    df_confirmed = _load_time_series(URL_CONFIRMED, "Confirmed")
    df_deaths = _load_time_series(URL_DEATHS, "Deaths")
    df_recovered = _load_time_series(URL_RECOVERED, "Recovered")

    df_merged = pd.merge(df_confirmed, df_deaths, on=MERGE_KEYS)
    df_merged = pd.merge(df_merged, df_recovered, on=MERGE_KEYS)

    return df_merged


def _remove_duplicates(df):
    """Drop exact duplicate rows."""
    return df.drop_duplicates(keep="first")


def _drop_rows_with_excessive_missing_values(df, max_missing_ratio=MAX_MISSING_RATIO_PER_ROW):
    """Drop rows whose share of missing values exceeds the given threshold."""
    row_missing_ratio = df.isnull().mean(axis=1)
    return df[row_missing_ratio <= max_missing_ratio].copy()


def _split_numerical_and_categorical_columns(df):
    """Return (numerical_columns, categorical_columns) based on dtype."""
    numerical_columns = [col for col in df.columns if df[col].dtype in ("float64", "int64")]
    categorical_columns = [col for col in df.columns if col not in numerical_columns]
    return numerical_columns, categorical_columns


def _fill_missing_values(df, numerical_columns, categorical_columns):
    """Fill missing numerical values with 0 and missing categorical values with 'unknown'."""
    df[numerical_columns] = df[numerical_columns].fillna(0)
    df[categorical_columns] = df[categorical_columns].fillna("unknown")
    return df


def _standardize_country_names(df):
    """Normalize inconsistent country name spellings/abbreviations."""
    df["Country/Region"] = df["Country/Region"].replace(COUNTRY_NAME_ALIASES)
    return df


def _sort_by_location_and_date(df):
    """Order rows so that each location forms a contiguous, chronological series."""
    return df.sort_values(LOCATION_KEY + ["Date"]).reset_index(drop=True)


def _add_daily_counts(df):
    """
    Convert the cumulative counts into per-day counts.

    Cumulative series are occasionally revised downward, which produces negative
    differences; those are clipped to zero.
    """
    grouped = df.groupby(LOCATION_KEY)

    df["New_Confirmed"] = grouped["Confirmed"].diff().fillna(df["Confirmed"]).clip(lower=0)
    df["New_Deaths"] = grouped["Deaths"].diff().fillna(df["Deaths"]).clip(lower=0)

    return df


def _add_days_since_first_case(df):
    """Add the number of days elapsed since a location's first confirmed case."""
    first_case_dates = (
        df[df["Confirmed"] > 0]
        .groupby(LOCATION_KEY)["Date"]
        .min()
        .rename("First_Case_Date")
    )

    df = df.merge(first_case_dates, on=LOCATION_KEY, how="left")
    df["Days_Since_First_Case"] = (df["Date"] - df["First_Case_Date"]).dt.days

    return df.drop(columns=["First_Case_Date"])


def _add_lagged_case_features(df):
    """
    Add past case counts as features.

    Only lagged values are used: a same-day count would reveal the denominator of the
    mortality rate being predicted.
    """
    new_confirmed_by_location = df.groupby(LOCATION_KEY)["New_Confirmed"]

    df[f"New_Confirmed_Lag_{SHORT_LAG_DAYS}"] = new_confirmed_by_location.shift(SHORT_LAG_DAYS)
    df[f"New_Confirmed_Lag_{LONG_LAG_DAYS}"] = new_confirmed_by_location.shift(LONG_LAG_DAYS)

    smoothed = new_confirmed_by_location.transform(
        lambda series: series.rolling(SMOOTHING_WINDOW_DAYS, min_periods=1).mean()
    )
    df["Smoothed_New_Confirmed"] = smoothed
    df[f"Smoothed_New_Confirmed_Lag_{LONG_LAG_DAYS}"] = df.groupby(LOCATION_KEY)[
        "Smoothed_New_Confirmed"
    ].shift(LONG_LAG_DAYS)

    return df.drop(columns=["Smoothed_New_Confirmed"])


def _enforce_logical_consistency(df):
    """Drop rows where reported deaths exceed reported confirmed cases."""
    return df[df["Deaths"] <= df["Confirmed"]].copy()


def _filter_low_case_counts(df, min_confirmed=MIN_CONFIRMED_CASES):
    """Drop rows whose case count is too small for the mortality rate to be meaningful."""
    return df[df["Confirmed"] >= min_confirmed].copy()


def _extract_date_features(df):
    """Derive day-of-week, month, and day-of-year features from the Date column."""
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    return df


def _compute_mortality_rate(df):
    """Compute mortality rate as a percentage of deaths over confirmed cases."""
    df["Mortality_Rate"] = (df["Deaths"] / df["Confirmed"]) * 100
    df["Mortality_Rate"] = df["Mortality_Rate"].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=["Mortality_Rate"])


def _remove_outliers_iqr(df, numerical_columns):
    """Remove rows containing outliers in any of the given columns, using the IQR method."""
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]
    return df.copy()


def clean_data(df_merged):
    """
    Cleans the merged COVID-19 dataset and derives the features used to model mortality rate.

    The cumulative JHU series is converted into per-day counts and lagged case features,
    so that the model predicts a location's mortality rate from epidemiological history
    rather than from same-day counts (which would leak the target).

    Steps:
        - Remove duplicates, rows with excessive missing data, and fill remaining gaps.
        - Standardize country names, then order each location's series chronologically.
        - Derive daily counts, days since first case, and lagged/smoothed case features.
        - Enforce logical consistency (deaths <= confirmed) and drop low-case-count rows.
        - Compute the mortality rate and trim its outliers.

    Args:
        df_merged (pd.DataFrame): The merged COVID-19 dataset.

    Returns:
        pd.DataFrame: The cleaned dataset ready for model training.
    """
    df = _remove_duplicates(df_merged)
    df = _drop_rows_with_excessive_missing_values(df)

    numerical_columns, categorical_columns = _split_numerical_and_categorical_columns(df)
    df = _fill_missing_values(df, numerical_columns, categorical_columns)
    df = _standardize_country_names(df)

    # The lag/diff features below require each location's series to be contiguous and in
    # order, so all row filtering happens after they are computed.
    df = _sort_by_location_and_date(df)
    df = _add_daily_counts(df)
    df = _add_days_since_first_case(df)
    df = _add_lagged_case_features(df)

    df = _enforce_logical_consistency(df)
    df = _filter_low_case_counts(df)
    df = _extract_date_features(df)
    df = _compute_mortality_rate(df)
    df = _remove_outliers_iqr(df, OUTLIER_COLUMNS)

    df_cleaned = df.dropna()

    return df_cleaned
