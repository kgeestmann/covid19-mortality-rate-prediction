# Analysis and Prediction of COVID-19 Mortality Rate

Predicting a region's COVID-19 mortality rate (deaths as a percentage of confirmed cases) from
Johns Hopkins University time-series data, using a Gradient Boosting model.

The central challenge in this project is **avoiding data leakage**. The target is derived from
death and case counts, so naive feature selection or a naive train/test split lets the model see
the answer it is supposed to predict. Most of the design decisions below exist to prevent that.

**Results on held-out future dates:**

| Metric | Model | Baseline (predict training mean) |
|---|---|---|
| MAE | **0.268** | 0.888 |
| MSE | **0.149** | 1.019 |
| R² | **0.816** | −0.259 |

---

## Project structure

```
covid19-mortality-rate-prediction/
├── src/
│   ├── data_preparation.py     Loading, cleaning, and feature engineering
│   ├── model.py                Train/test split, pipeline, training, evaluation
│   └── visualization.py        Exploratory plots
├── tests/
│   ├── test_data_preparation.py
│   └── test_model.py
├── images/                     Generated plots (not tracked in git)
├── main.py                     Entry point — runs the full pipeline
├── requirements.txt
└── pytest.ini
```

## Setup and usage

Requires Python 3.9+.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows (PowerShell)
source .venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Run the tests
pytest
```

`main.py` downloads the data, cleans it, prints summaries, displays several plots, then trains and
evaluates the model. **Each plot window blocks execution until it is closed.** Training the
Gradient Boosting model takes a couple of minutes, as its 300 trees are built sequentially.

---

## Data retrieval and preparation

Handled by `load_covid_data()`, which returns a `pd.DataFrame`.

Data is retrieved from the [Johns Hopkins University CSSE repository](https://github.com/CSSEGISandData/COVID-19)
as three separate time series: confirmed cases, deaths, and recoveries.

**Reshaping:** Each file arrives in wide format, with one column per date. It is melted into long
format so each row is a single observation (one location, one date, one count), which makes the
three datasets mergeable and easier to analyse.

**Date conversion:** The date column arrives as strings (`"1/22/20"`) and is parsed into proper
datetime values, so it can be sorted and used for time-based features.

**Merging:** The three datasets are joined on province/state, country/region, latitude, longitude,
and date into a single DataFrame.

```console
Shape of the data: (306324, 8)

Column names: Index(['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'Confirmed',
       'Deaths', 'Recovered'], dtype='object')

Missing values in the columns:
Province/State    222885
Country/Region         0
Lat                 1143
Long                1143
Date                   0
Confirmed              0
Deaths                 0
Recovered              0
```

---

## Data cleaning and feature engineering

Handled by `clean_data(df)`, which expects the merged DataFrame and returns a cleaned one.

The JHU series is **cumulative** — each row is a running total, not a daily figure. Consecutive
days for the same location are therefore nearly identical. This single fact drives most of the
decisions below.

### Order of operations

Row filtering happens *after* the lag and difference features are computed. Those features require
each location's series to be contiguous and chronological; removing rows first would silently
corrupt them by creating gaps.

### Cleaning steps

**Removing duplicates** — exact duplicate rows are dropped.

**Handling missing values** — columns are split into numerical and categorical. Numerical gaps are
filled with 0, categorical gaps with `"unknown"` (most `Province/State` values are missing, since
most countries report at national level). Rows missing more than 30% of their values are dropped.

**Standardizing country names** — `"US"` and `"U.S."` become `"United States"`, `"UK"` becomes
`"United Kingdom"`. This runs *before* grouping, so each country forms one series rather than
several fragments.

**Enforcing logical consistency** — rows where deaths exceed confirmed cases are removed, as they
represent reporting errors.

**Filtering small denominators** — rows with fewer than 100 cumulative confirmed cases are dropped.
A region with 1 case and 1 death produces a mortality rate of 100%, which is noise rather than
signal, and such rows otherwise dominate the target's upper range.

**Outlier removal** — the IQR method is applied **to the mortality rate only**. This is deliberate:

- Applying it to `Lat`/`Long` would delete entire countries at extreme latitudes. Coordinates are
  identifiers, not measurements, so they have no meaningful outliers.
- Applying it to case counts would delete every major outbreak. In an exponential epidemic the
  large values *are* the signal.

### Engineered features

**Daily counts** (`New_Confirmed`, `New_Deaths`) — the cumulative totals are differenced within
each location to recover per-day figures. Cumulative series are occasionally revised downward,
producing negative differences, which are clipped to zero.

**Days since first case** (`Days_Since_First_Case`) — days elapsed since a location's first
confirmed case, capturing how far into its own outbreak a region is, independent of the calendar.

**Lagged case counts** (`New_Confirmed_Lag_7`, `New_Confirmed_Lag_14`,
`Smoothed_New_Confirmed_Lag_14`) — case counts from 7 and 14 days earlier. Deaths lag infections by
roughly two to three weeks, so past case load is the epidemiologically meaningful predictor of
today's mortality rate. The smoothed variant is a 7-day rolling mean, which absorbs the weekly
reporting artefacts (weekend dips) present in the raw data.

**Calendar features** (`Day_of_Week`, `Month`, `Day_of_Year`) — extracted from the date.

**Mortality rate** (`Mortality_Rate`) — the target: deaths as a percentage of confirmed cases.
Infinite values (from zero-case rows) are removed.

### Resulting dataset

```console
Number of records after cleaning: 243586
Columns: 18
Date range: 2020-02-05 to 2023-03-09
```

79.5% of the original rows are retained. The target is now well behaved, with no small-denominator
extremes:

```console
Mortality_Rate
mean      1.327
std       1.042
min       0.000
25%       0.498
50%       1.119
75%       1.966
max       4.660
```

---

## Avoiding data leakage

Two distinct leaks had to be closed. Both are easy to miss, and both inflate scores while producing
a model that would fail in practice.

### 1. Feature leakage

`Mortality_Rate` is `Deaths / Confirmed`. Any same-day count therefore reveals part of the answer
directly. The following columns are excluded from the feature matrix:

`Confirmed`, `Deaths`, `Recovered`, `New_Confirmed`, `New_Deaths`, `Mortality_Rate`, `Date`

Only **lagged** case history survives into the model — information that would genuinely have been
available before the prediction date.

### 2. Split leakage

Because the source data is cumulative, day 200 and day 201 for the same province are near-duplicate
rows with near-identical targets. A random `train_test_split` scatters those near-duplicates across
both sets, so the model can recall answers from training rather than generalise to new ones.

The split is therefore **chronological**: the model trains on the earliest 80% of dates and is
tested on the most recent 20%, which it has never seen.

```console
Training on 194671 rows up to 2022-08-29
Testing on 48915 rows from 2022-08-30 onward
```

This is a strictly harder evaluation than a random split, and it is the only one that reflects the
real task: predicting forward in time.

---

## Data exploration and visualization

Handled by `explore_data()`, `plot_cumulative_deaths()`, `plot_top_countries()`,
`plot_cumulative_deaths_by_country()`, and `plot_mortality_rate_comparison()`. Each expects a
cleaned DataFrame and renders one or more plots.

`explore_data(df)` — prints descriptive statistics, computes the correlation matrix, renders a
correlation heatmap, and plots distributions for `Deaths`, `Confirmed`, and `Recovered`.

Correlation of each numerical feature with the target:

```console
Mortality_Rate                   1.000000
Deaths                           0.099188
New_Deaths                       0.087337
Recovered                        0.076391
New_Confirmed                    0.001898
New_Confirmed_Lag_7             -0.001744
New_Confirmed_Lag_14            -0.004346
Smoothed_New_Confirmed_Lag_14   -0.005511
Confirmed                       -0.010665
Lat                             -0.070477
Days_Since_First_Case           -0.233079
Long                            -0.244717
```

Every linear correlation is weak. The strongest are `Long` (−0.24) and `Days_Since_First_Case`
(−0.23). This is informative in itself: any predictive power the model has must come from
non-linear structure and from the categorical location features, not from simple linear
relationships — which is consistent with the feature importances reported below.

`plot_cumulative_deaths(df)` — global cumulative deaths over time, grouped by date.

`plot_top_countries(df)` — horizontal bar chart of the 10 countries with the highest total deaths.

`plot_cumulative_deaths_by_country(df, countries)` — cumulative deaths over time, one line per
country. `main.py` passes every country in the dataset, which is more series than a single chart
can legibly display; it is included for exploratory purposes.

`plot_mortality_rate_comparison(df)` — the 10 countries with the highest average mortality rate.

---

## Model building and evaluation

Handled by `train_and_evaluate_model(df)`, which returns the fitted pipeline and a metrics
dictionary.

**Features used:**

```console
['Province/State', 'Country/Region', 'Lat', 'Long', 'Days_Since_First_Case',
 'New_Confirmed_Lag_7', 'New_Confirmed_Lag_14', 'Smoothed_New_Confirmed_Lag_14',
 'Day_of_Week', 'Month', 'Day_of_Year']
```

**Preprocessing** — applied through a `ColumnTransformer` so both transformations stay consistent:

- Numerical columns are standardized with `StandardScaler`, giving each a mean of 0 and standard
  deviation of 1. This prevents features with large ranges (such as longitude) from dominating
  those with smaller ones.
- `Country/Region` and `Province/State` are one-hot encoded. These categories have no ordinal
  relationship — "China" is not greater than "France" — so one-hot encoding is the appropriate
  representation. `handle_unknown="ignore"` ensures categories appearing only in the test period
  do not cause failures.

**Model** — `GradientBoostingRegressor`, chosen after comparing against `XGBoost` and
`LinearRegression`:

- `learning_rate=0.05` — a low rate makes each tree's correction small, which improves
  generalization at the cost of needing more trees.
- `max_depth=7` — deep enough to capture interactions between location and outbreak stage, shallow
  enough to limit overfitting.
- `n_estimators=300` — balances model capacity against training time and overfitting risk.

Preprocessing and model are combined into a single `Pipeline`, so the scaler and encoder are fitted
on training data only and applied identically to the test data.

### Results

```console
MAE: 0.268, MSE: 0.149, R²: 0.816
```

Measured against a naive baseline that predicts the training mean for every row:

| Metric | Model | Baseline |
|---|---|---|
| MAE | 0.268 | 0.888 |
| MSE | 0.149 | 1.019 |
| R² | 0.816 | −0.259 |

The baseline's *negative* R² is worth noting: average mortality rates fell over the course of the
pandemic as treatment and vaccination improved, so the training-period mean is actively misleading
when applied to later dates. The model handles that distribution shift, which the baseline by
definition cannot.

### What the model actually learned

Feature importances, with one-hot columns aggregated back to their source feature:

| Feature | Importance |
|---|---|
| Country/Region | 0.361 |
| Province/State | 0.221 |
| Long | 0.136 |
| Lat | 0.134 |
| Days_Since_First_Case | 0.122 |
| Smoothed_New_Confirmed_Lag_14 | 0.015 |
| Day_of_Year | 0.009 |
| New_Confirmed_Lag_7 | 0.002 |
| New_Confirmed_Lag_14 | 0.002 |
| Month | 0.001 |
| Day_of_Week | 0.000 |

**Roughly 85% of the model's predictive weight comes from location** (`Country/Region`,
`Province/State`, `Lat`, `Long`), and a further 12% from `Days_Since_First_Case`. The lagged case
features together contribute under 2%.

In other words, the model has largely learned *"where are you, and how far into your outbreak are
you?"* — which is a real and defensible signal, since mortality rates genuinely varied enormously
between countries because of healthcare capacity, population age structure, and testing regimes.
But it is not modelling epidemic dynamics, and the lagged case counts added far less than expected.

### Limitations

- **The model is largely a country-level lookup.** It would generalize poorly to a region absent
  from the training data, since location dominates its decisions.
- **No population normalization.** Absolute case counts are used rather than per-capita rates, so
  the model cannot distinguish a large outbreak from a large country.
- **Key drivers are missing from the data.** Population age structure, healthcare capacity,
  vaccination coverage, and testing rates all strongly affect mortality but are not present in the
  JHU time series.
- **Reporting quality varies by country**, and the mortality rate measures *reported* deaths over
  *reported* cases. Differences between countries partly reflect differences in testing and
  reporting rather than in real outcomes.

---

## Tests

21 unit tests cover the non-trivial logic, and can be run with `pytest`.

`tests/test_data_preparation.py` covers each cleaning and feature-engineering step in isolation:
deduplication, missing-value handling, IQR trimming, logical consistency, the small-denominator
filter, and the derived features. Several tests specifically guard the correctness of the
time-series logic — that differencing does not bleed across location boundaries, that downward data
revisions are clipped, and that lagged columns are offset by the expected number of days.

`tests/test_model.py` covers the split and evaluation logic, including two assertions that protect
against the leaks described above: that no excluded column can reach the feature matrix, and that
the train and test sets share no dates.

---

## Author

Kim Geestmann
