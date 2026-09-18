from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

TARGET_COLUMN = "Mortality_Rate"
DATE_COLUMN = "Date"

# Mortality rate is Deaths/Confirmed, so any same-day count reveals the target directly.
# Only lagged case history, location, and calendar features are left as predictors.
COLUMNS_EXCLUDED_FROM_FEATURES = [
    "Confirmed",
    "Deaths",
    "Recovered",
    "New_Confirmed",
    "New_Deaths",
    "Mortality_Rate",
    "Date",
]
CATEGORICAL_COLUMNS = ["Country/Region", "Province/State"]

TEST_SIZE = 0.2
GRADIENT_BOOSTING_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 7,
    "n_estimators": 300,
}


def _temporal_train_test_split(df, test_size=TEST_SIZE):
    """
    Split chronologically, holding out the most recent dates as the test set.

    A random split would scatter near-identical consecutive days of the same cumulative
    series across both sets, letting the model recall answers instead of predicting them.
    """
    df_sorted = df.sort_values(DATE_COLUMN)
    split_position = int(len(df_sorted) * (1 - test_size))
    split_date = df_sorted.iloc[split_position][DATE_COLUMN]

    train_df = df_sorted[df_sorted[DATE_COLUMN] < split_date]
    test_df = df_sorted[df_sorted[DATE_COLUMN] >= split_date]

    return train_df, test_df


def _split_features_and_target(df):
    """Split the cleaned dataset into feature matrix X and target vector y."""
    X = df.drop(columns=COLUMNS_EXCLUDED_FROM_FEATURES)
    y = df[TARGET_COLUMN]
    return X, y


def _build_pipeline(numerical_columns, categorical_columns):
    """Build the preprocessing + model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS)),
        ]
    )


def _evaluate(y_test, y_pred):
    """Compute regression evaluation metrics."""
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def _plot_actual_vs_predicted(y_test, y_pred):
    """Visualize actual vs predicted mortality rates."""
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Mortality Rate")
    plt.ylabel("Predicted Mortality Rate")
    plt.title("Actual vs Predicted Mortality Rate")
    plt.show()


def train_and_evaluate_model(df):
    """
    Trains a gradient boosting model to predict COVID-19 mortality rate and evaluates it.

    Args:
        df (pd.DataFrame): The cleaned COVID-19 dataset.

    Returns:
        tuple[Pipeline, dict]: The fitted pipeline and a dict of evaluation metrics
        (mae, mse, r2).
    """
    train_df, test_df = _temporal_train_test_split(df)
    X_train, y_train = _split_features_and_target(train_df)
    X_test, y_test = _split_features_and_target(test_df)

    numerical_columns = [col for col in X_train.columns if col not in CATEGORICAL_COLUMNS]

    print(f"\nTraining on {len(X_train)} rows up to {train_df[DATE_COLUMN].max().date()}")
    print(f"Testing on {len(X_test)} rows from {test_df[DATE_COLUMN].min().date()} onward")
    print("\nFeatures used:", list(X_train.columns))

    pipeline = _build_pipeline(numerical_columns, CATEGORICAL_COLUMNS)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = _evaluate(y_test, y_pred)
    print(f"\nMAE: {metrics['mae']}, MSE: {metrics['mse']}, R²: {metrics['r2']}")

    _plot_actual_vs_predicted(y_test, y_pred)

    return pipeline, metrics
