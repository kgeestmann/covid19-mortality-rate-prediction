from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def modelBuilder(df):

    # Features and target variable
    X = df.drop(columns=['Confirmed', 'Deaths', 'Recovered', 'Mortality_Rate', 'Date'])
    y = df['Mortality_Rate']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print out basic overview of train and test sets
    print("\nTrain Set Overview:")
    print(X_train.head())
    print("\nTrain Set Description:")
    print(X_train.describe())

    print("\nTest Set Overview:")
    print(X_test.head())
    print("\nTest Set Description:")
    print(X_test.describe())

    # Columns to encode
    categorical_columns = ['Country/Region', 'Province/State']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # One-hot encode categorical features
        ]
    )

    # Full pipeline including the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(learning_rate=0.05, max_depth=7, n_estimators=300))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMAE: {mae}, MSE: {mse}, R²: {r2}")

    # Visualizing actual vs predicted mortality rates
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Mortality Rate')
    plt.ylabel('Predicted Mortality Rate')
    plt.title('Actual vs Predicted Mortality Rate')
    plt.show()