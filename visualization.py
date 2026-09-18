import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import zscore

def dataExploration(df):
    """
    Performs exploratory data analysis (EDA) on the given DataFrame.

    - Prints descriptive statistics
    - Generates a heatmap of the correlation matrix for numerical variables
    - Displays histograms of important variables

    Args:
        df (pd.DataFrame): The merged DataFrame containing COVID-19 data
    """
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Correlation matrix and heatmap
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_cols].corr()

    print("\nCorrelation Matrix (Numerical Columns Only):")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
    plt.title("Correlation Matrix of Numerical Variables")
    plt.show()

    # Distribution of important variables
    important_columns = ['Deaths', 'Confirmed', 'Recovered']
    for col in important_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

def cumulativeDeaths(df):
    """
    Visualizes global cumulative deaths over time

    Args:
        df (pd.DataFrame): The DataFrame containing COVID-19 data
    """
    df_global = df.groupby("Date").agg({"Deaths": "sum"}).reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(df_global["Date"], df_global["Deaths"], label="Global Deaths")
    plt.title("Global COVID-19 Deaths")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Deaths")
    plt.legend()
    plt.grid()
    plt.show()

def topCountries(df):
    """
    Visualizes the top 10 countries with the highest number of deaths

    Args:
        df (pd.DataFrame): The DataFrame containing COVID-19 data
    """
    df_country_deaths = df.groupby("Country/Region").agg({"Deaths": "sum"}).reset_index()
    df_country_deaths = df_country_deaths.sort_values(by="Deaths", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(df_country_deaths["Country/Region"][:10], df_country_deaths["Deaths"][:10], color='orange')
    plt.title("Top 10 Countries with the Most COVID-19 Deaths")
    plt.xlabel("Deaths")
    plt.ylabel("Country")
    plt.show()

def cumulativeByCountry(df, countries):
    """
    Visualizes cumulative deaths for selected countries over time

    Args:
        df (pd.DataFrame): The DataFrame containing COVID-19 data
        countries (list): List of country names to visualize
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df_selected = df[df["Country/Region"].isin(countries)]
    df_cumulative = df_selected.groupby(["Date", "Country/Region"]).agg({"Deaths": "sum"}).reset_index()

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_cumulative, x="Date", y="Deaths", hue="Country/Region", marker="o")
    plt.title("Cumulative COVID-19 Deaths in Selected Countries")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    plt.legend(title="Country", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def mortalityRateComparison(df):
    """
    Visualizes the mortality rates of the top 10 countries

    Args:
        df (pd.DataFrame): The DataFrame containing COVID-19 data
    """
    df["Mortality_Rate"] = df["Deaths"] / df["Confirmed"]
    df_country_mortality = df.groupby("Country/Region").agg({"Mortality_Rate": "mean"}).reset_index()
    df_country_mortality = df_country_mortality.sort_values(by="Mortality_Rate", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Mortality_Rate", y="Country/Region", data=df_country_mortality[:10], palette="Reds_r",hue="Country/Region", legend=False)
    plt.title("Mortality Rates of the Top 10 Countries")
    plt.xlabel("Mortality Rate")
    plt.ylabel("Country")
    plt.show()