# 🍽️ Restaurant Revenue Prediction

Predicting annual restaurant revenue from operational metadata and 37 obfuscated numeric features on a deliberately small Kaggle dataset (137 rows). The real learning here wasn't building a high-performing model — it's understanding what happens when you have far more features than samples, and why collecting more data beats any amount of tuning.

## ✨ What I Did and Why
* **Exploratory Data Analysis:** Checked for missing values and duplicates, profiled cardinality of categorical columns (`City` at 34 unique values vs `City Group` at 2 and `Type` at 3), and visualized distributions of the obfuscated P1–P37 features and the revenue target.
* **Feature Engineering:** Parsed `Open Date` into `restaurant_age_years` (measured from a 2010-01-01 snapshot). Dropped the high-cardinality `City` column — 34 unique city values for 137 rows is too granular to generalize and would cause massive overfitting.
* **One-Hot Encoding:** Converted `City Group` and `Type` into dummy columns with `drop_first=True` to avoid the dummy variable trap, yielding a final 41-feature matrix.
* **Multi-Model Comparison:** Trained 7 regressors — Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and KNN — ranked by 5-fold cross-validation R². Single train/test splits are unreliable at n=137, so CV is the only honest evaluation method here.
* **Baseline Comparison:** Benchmarked every model against predicting the training mean. If a model can't beat a flat guess, it has no value. The tuned Random Forest beats baseline by 2.7% — barely.
* **GridSearchCV Tuning:** Tuned for `max_depth`, `n_estimators`, and `min_samples_leaf`. With ~40 features and 110 training rows, regularization through `min_samples_leaf` is critical to prevent the tree from memorizing the training set.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning:** Scikit-Learn (RandomForestRegressor, GradientBoostingRegressor, Ridge, Lasso, LinearRegression, DecisionTreeRegressor, KNeighborsRegressor, StandardScaler, GridSearchCV, cross_val_score)
* **Dataset:** TFI Restaurant Revenue Prediction (Kaggle) — `train.csv`, 137 rows, 43 columns: `Id`, `Open Date`, `City`, `City Group`, `Type`, `P1`–`P37` (obfuscated numerics), `revenue` target

## 🚀 Setup
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn`
3. Place `train.csv` in the project root.
4. Run notebooks in order:
   - `01_EDA.ipynb` — explore and understand the raw data
   - `02_Data_Cleaning.ipynb` — feature engineer and save `train_cleaned.csv`
   - `03_Model_Building.ipynb` — train, compare, and tune all models

## 📊 Results
* **Best Model (by test R²):** Gradient Boosting — `test_r2: 0.097`, `test_rmse: ~$3.32M`
* **Most CV-Stable Model:** Random Forest (tuned) — `cv_r2: −0.024`, `test_r2: 0.042`, `test_rmse: ~$3.42M`
* **Baseline RMSE** (predict training mean): ~$3.52M → tuned RF achieves a 2.7% improvement
* **Key Observation:** Revenue is barely predictable from these features at n=137. Linear models overfit catastrophically (negative R²) because 40 features vs 110 training rows is a near-impossible ratio. The obfuscated P1–P37 features carry limited interpretable signal. The main takeaway: more data would have more impact than any further model tuning. This is the honest result — and documenting it is more valuable than inflating the numbers.

## ⚠️ Limitations
* 137 rows is not enough to reliably train any model with 40+ features — cross-validation variance is very high.
* The P1–P37 features are obfuscated with no domain context, making feature engineering nearly impossible.
* A 2.7% improvement over baseline is not deployable — this project is a learning exercise in recognizing data limitations, not a production model.
