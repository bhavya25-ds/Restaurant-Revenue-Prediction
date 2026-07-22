# 🍽️ Restaurant Revenue Prediction

Predicting annual restaurant revenue from metadata and 37 obfuscated numeric features using seven regression models, with cross-validation, feature engineering, and GridSearchCV hyperparameter tuning on a deliberately small Kaggle dataset (137 rows).

## ✨ Features

* **Exploratory Data Analysis:** Checks for missing values and duplicates, profiles cardinality of categorical columns (`City` at 34 unique values vs `City Group` at 2 and `Type` at 3), and visualizes distributions of the obfuscated P1–P37 features and the revenue target.

* **Feature Engineering:** Parses `Open Date` into `restaurant_age_years` (measured from a 2010-01-01 snapshot), drops the high-cardinality `City` column (34 levels for 137 rows is too granular to generalize), and drops identifier columns `Id` and `Open Date`.

* **One-Hot Encoding:** Converts `City Group` (2 levels) and `Type` (3 levels) into numeric dummy columns with `drop_first=True` to avoid the dummy-variable trap, yielding a final 41-feature matrix.

* **Multi-Model Comparison:** Trains and evaluates 7 regressors — Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and KNN — ranked by 5-fold cross-validation R² to avoid overfitting conclusions from a single train/test split on 137 rows.

* **Hyperparameter Tuning:** Uses GridSearchCV over `max_depth`, `n_estimators`, and `min_samples_leaf` to find the least-overfitting Random Forest configuration on a dataset where ~40 features vs 110 training rows makes regularization critical.

* **Baseline Comparison:** Benchmarks the tuned model against the naive "predict the training mean" baseline to quantify whether the model adds real predictive value beyond a flat guess.

* **Predicted vs Actual Visualization:** Scatter plot of Random Forest predictions against held-out actuals, with a perfect-prediction reference line to visually confirm the model's tendency to regress toward the mean.

* **Feature Importance Chart:** Horizontal bar chart of the top 15 most influential features from the Random Forest, identifying which of the obfuscated P1–P37 variables carry the most signal.

* **Submission Generation:** preprocesses test data through the same pipeline, aligns columns via reindex to handle unseen categories (e.g. Type_MB), and exports predictions to submission.csv.

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning:** Scikit-Learn (RandomForestRegressor, GradientBoostingRegressor, Ridge, Lasso, LinearRegression, DecisionTreeRegressor, KNeighborsRegressor, StandardScaler, GridSearchCV, cross\_val\_score)
* **Dataset:** TFI Restaurant Revenue Prediction (Kaggle) — `train.csv`, 137 rows, 43 columns: `Id`, `Open Date`, `City`, `City Group`, `Type`, `P1`–`P37` (obfuscated numerics), and `revenue` target

## 🚀 Setup

1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn`
3. Place `train.csv` in the project root (same folder as `utils.py`).
4. Run the notebooks in order:
   - `01_EDA.ipynb` — explore and understand the raw data
   - `02_Data_Cleaning.ipynb` — feature engineer and save `train_cleaned.csv`
   - `03_Model_Building.ipynb` — train, compare, and tune all models

## 📊 Results

* **Best Model (by test R²):** Gradient Boosting — `test_r2: 0.097`, `test_rmse: ~$3.32M`
* **Best CV-Stable Model:** Random Forest (tuned) — `cv_r2: −0.024`, `test_r2: 0.042`, `test_rmse: ~$3.42M`
* **Tuned Params:** `max_depth=None`, `min_samples_leaf=5`, `n_estimators=100`
* **Baseline RMSE** (predict training mean): ~$3.52M → tuned RF achieves a **2.7% improvement**
* **Key Observation:** Revenue is only weakly predictable from these features at n=137. Linear models overfit catastrophically (negative R²) because there are ~40 features for just 110 training rows. Tree ensembles are the most stable but still barely beat the mean baseline — the obfuscated P1–P37 features carry limited interpretable signal, and with this few rows, cross-validation variance is very high. Collecting more data would have far more impact than any further model tuning.
