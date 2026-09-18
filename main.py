from src.data_preparation import load_covid_data, clean_data
from src.visualization import (
    explore_data,
    plot_cumulative_deaths,
    plot_top_countries,
    plot_cumulative_deaths_by_country,
    plot_mortality_rate_comparison,
)
from src.model import train_and_evaluate_model


def main():
    """
    Loads the COVID-19 data, prepares it, then runs exploration, visualization, and modeling.
    """
    # Load and prepare data
    print("Loading and preparing data...\n")
    df = load_covid_data()
    print(df.head())
    print("\nShape of the data:", df.shape)
    print("\nColumn names:", df.columns)
    print(f"\nNumber of records: {df.shape[0]}")
    print("\nMissing values in the columns:", df.isnull().sum())

    # Clean the data
    print("\nCleaning and transforming the data...\n")
    df_cleaned = clean_data(df)
    print(df_cleaned.head())
    print("\nColumn names after transforming:", df_cleaned.columns)
    print(f"\nNumber of records after cleaning: {df_cleaned.shape[0]}")
    print("\nMissing values in the columns after cleaning:", df_cleaned.isnull().sum())

    # Data exploration
    print("\nData exploration...")
    explore_data(df_cleaned)

    # Visualization of cumulative deaths, top countries, cumulative by country, and mortality rate
    plot_cumulative_deaths(df_cleaned)
    plot_top_countries(df_cleaned)
    plot_cumulative_deaths_by_country(df_cleaned, countries=df_cleaned["Country/Region"].unique())
    plot_mortality_rate_comparison(df_cleaned)

    # Build the model and evaluate metrics
    print("\nTraining the model...")
    train_and_evaluate_model(df_cleaned)


if __name__ == "__main__":
    main()
