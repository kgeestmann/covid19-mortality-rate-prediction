import numpy as np
import pandas as pd

def loadCovidData():
    """
    Loads and processes global COVID-19 data for confirmed cases, deaths, and recoveries.

    The function retrieves time-series data from the Johns Hopkins University repository,
    reshapes the data into a long format, and merges the datasets into a single DataFrame.

    Returns:
        pd.DataFrame: A merged DataFrame containing COVID-19 confirmed cases, deaths,
        and recoveries along with data such as location and date.
    """
    # URLs for COVID-19 data
    url_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    url_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    # Load CSV files into DataFrames
    df_confirmed = pd.read_csv(url_confirmed)
    df_deaths = pd.read_csv(url_deaths)
    df_recovered = pd.read_csv(url_recovered)

    # Transform DataFrames into long format
    df_confirmed = df_confirmed.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"],
                                     var_name="Date",
                                     value_name="Confirmed")

    df_deaths = df_deaths.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"],
                               var_name="Date",
                               value_name="Deaths")

    df_recovered = df_recovered.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"],
                                     var_name="Date",
                                     value_name="Recovered")

    # Convert the Date column to datetime format
    df_confirmed["Date"] = pd.to_datetime(df_confirmed["Date"], format="%m/%d/%y", errors="coerce")
    df_deaths["Date"] = pd.to_datetime(df_deaths["Date"], format="%m/%d/%y", errors="coerce")
    df_recovered["Date"] = pd.to_datetime(df_recovered["Date"], format="%m/%d/%y", errors="coerce")

    # Merge the datasets into a single DataFrame
    df_merged = pd.merge(df_confirmed, df_deaths, on=["Province/State", "Country/Region", "Lat", "Long", "Date"])
    df_merged = pd.merge(df_merged, df_recovered, on=["Province/State", "Country/Region", "Lat", "Long", "Date"])

    return df_merged

def cleanData(df_merged):
    """
    Cleans and processes the merged COVID-19 dataset by handling duplicates, missing values, outliers,
    logical inconsistencies and reshapes data to prepare the dataset for training the model.

    Steps:
        - Remove duplicate rows.
        - Separate numerical and categorical columns for targeted cleaning.
        - Fill missing values in numerical and categorical columns.
        - Remove rows with excessive missing data.
        - Handle outliers using the IQR method.
        - Check and enforce logical consistency (e.g., deaths <= confirmed cases).
        - Standardize country names and extract features from the date column.
        - Compute the mortality rate (deaths/confirmed cases).

    Args:
        df_merged (pd.DataFrame): The merged COVID-19 dataset.

    Returns:
        pd.DataFrame: The cleaned dataset ready for model training.
    """
    # Remove duplicate rows
    df_merged = df_merged.drop_duplicates(keep='first')

    # Remove rows with more than 30% missing values
    threshold = 0.3
    row_missing_percentage = df_merged.isnull().mean(axis=1)
    df_merged = df_merged[row_missing_percentage <= threshold]

    # Separate columns into numerical and categorical
    numerical = []
    categorical = []

    for col in df_merged.columns:
        if df_merged[col].dtypes == 'float64' or df_merged[col].dtypes == 'int64':
            numerical.append(col)
        else:
            categorical.append(col)

    # Fill missing values
    df_merged[numerical] = df_merged[numerical].fillna(0)
    df_merged[categorical] = df_merged[categorical].fillna("unknown")

    # Handle outliers using the IQR method
    for col in numerical:
        Q1 = df_merged[col].quantile(0.25)
        Q3 = df_merged[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df_merged = df_merged[(df_merged[col] >= lower_limit) & (df_merged[col] <= upper_limit)]

    # Enforce logical consistency: deaths cannot exceed confirmed cases
    df_merged = df_merged[df_merged['Deaths'] <= df_merged['Confirmed']]

    # Standardize country names
    df_merged["Country/Region"] = df_merged["Country/Region"].replace({
        "US": "United States", "UK": "United Kingdom", "U.S.": "United States"
    })

    # Extract features from the Date column
    df_merged['Day_of_Week'] = df_merged['Date'].dt.dayofweek
    df_merged['Month'] = df_merged['Date'].dt.month
    df_merged['Day_of_Year'] = df_merged['Date'].dt.dayofyear

    # Calculate mortality rate
    df_merged['Mortality_Rate'] = (df_merged['Deaths'] / df_merged['Confirmed']) * 100

    # Handle infinite values and NaNs in Mortality_Rate
    df_merged['Mortality_Rate'] = df_merged['Mortality_Rate'].replace([np.inf, -np.inf], np.nan)
    df_merged.dropna(subset=['Mortality_Rate'], inplace=True)

    # Drop any remaining rows with NaN values
    df_cleaned = df_merged.dropna()

    return df_cleaned