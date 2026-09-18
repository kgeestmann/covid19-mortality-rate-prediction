from dataPreparation import *
from visualization import *
from model import *

def main():
    """
    Main function that loads the data, prepares it, and calls visualization and modeling functions.

    This function handles the entire process by:
        - Loading the COVID-19 data (loadCovidData())
        - Cleaning the data. (cleanData(df))
        - Performing data exploration (dataExploration(df_cleaned))
        - Visualizing key insights like cumulative deaths,
          top countries, and mortality rate comparisons.
          (cumulativeDeaths(df_cleaned), topCountries(df_cleaned), mortalityRateComparison(df_cleaned))
        - Training the model and evaluating its performance (modelBuilder(df_cleaned))
    """
    # Load and prepare data
    print("Loading and preparing data...\n")
    df = loadCovidData()
    print(df.head())
    print("\nShape of the data:", df.shape)
    print("\nColumn names:", df.columns)
    print(f"\nNumber of records: {df.shape[0]}")
    print("\nMissing values in the columns:", df.isnull().sum())

    # Clean the data
    print("\nCleaning and transforming the data...\n")
    df_cleaned = cleanData(df)
    print(df_cleaned.head())
    print("\nColumn names after transforming:", df_cleaned.columns)
    print(f"\nNumber of records after cleaning: {df_cleaned.shape[0]}")
    print("\nMissing values in the columns after cleaning:", df_cleaned.isnull().sum())

    # Data exploration
    print("\nData exploration...")
    dataExploration(df_cleaned)

    # Visualization of cumulative deaths, top countries, cumulative by country, and mortality rate comparison
    cumulativeDeaths(df_cleaned)
    topCountries(df_cleaned)
    cumulativeByCountry(df_cleaned, countries=df_cleaned['Country/Region'].unique())
    mortalityRateComparison(df_cleaned)

    # Build the model and evaluate metrics
    print("\nTraining the model...")
    modelBuilder(df_cleaned)

if __name__ == "__main__":
    main()