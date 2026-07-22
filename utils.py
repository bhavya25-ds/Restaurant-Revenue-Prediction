"""
Restaurant Revenue Prediction- Utility Functions
Regression on the kaggle (TFI) restaurant dataset: predict annual revenue
from restaurant metadata and 37 obfuscated (intentionally obscure)
numeric feature ranging from P1-P37.

This is a small dataset of only 137 rows, so cross validation matters more than 
any single train/test split.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor



# DATA LOADING

def load_data(filepath= "train.csv"):
    return pd.read_csv(filepath)





# FEATURE ENGINEERING

REFERENCE_DATE= pd.Timestamp("2010-01-01") #age measured from here

"""
Engineer model-ready features from the raw dataframe:
> restaurant_age_years= years between open date and reference snapshot
> drop ID (identifier) and City (34 high-cardinality level - too many unique values, i.e., granular for 137 row df)
> keep City group and Type: low cardinality categories, and P1-P37: numeric
returns a new df including the 'revenue' target (if present).
"""

def create_features(df, reference_date= REFERENCE_DATE):
    df= df.copy()
    df["Open Date"] = pd.to_datetime(df["Open Date"], format="%m/%d/%Y")
    df["restaurant_age_years"] = (reference_date- df["Open Date"]).dt.days/365.25

    drop_cols= [c for c in ["Id", "Open Date", "City"] if c in df.columns]
    df= df.drop(columns= drop_cols)
    return df






# PREPROCESSING

def preprocess_data(df, target= "revenue"):
    """
    Full pipeline: feature engineer -> one-hot encode categoricals (drop_first) 
    (Convert text columns like City Group into numbers) -> 
    split X/y. Returns X (Dataframe), y (Series), feature names.
    """

    fe= create_features(df)
    fe= pd.get_dummies(fe, columns= [c for c in ["City Group", "Type"] if c in fe.columns], drop_first= True)
    y= fe[target] if target in fe.columns else None
    X= fe.drop(columns= [target]) if target in fe.columns else fe
    return X,y, list(X.columns)





# MODELS

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=1000.0, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5)
    }





# MODEL EVALUATION 

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred= model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_test, pred))
    return{
        "r2": round(r2_score(y_test, pred), 4),
        "rmse": round(rmse, 1),
        "mae": round(mean_absolute_error(y_test, pred), 1),
    }




def compare_models(models, X_train, X_test, y_train, y_test, X_all=None, y_all= None, cv=5):
    """
    Evaluating every model based on the test data and get R², RMSE, MAE scores.
    Ranking the most appropriate models from top to bottom based on the Cross Validation Score
    """
    rows=[]
    for name, model in models.items():
        m= evaluate_model(model, X_train, X_test, y_train, y_test)
        row= {"model": name, **{f"test_{k}": v for k, v in m.items()}}
        if X_all is not None:
            scores= cross_val_score(model, X_all, y_all, cv=cv, scoring="r2")
            row["cv_r2_mean"]= round(scores.mean(), 4)
            row["cv_r2_std"]= round(scores.std(), 4)
        rows.append(row)
    out= pd.DataFrame(rows)
    sort_key= "cv_r2_mean" if "cv_r2_mean" in out.columns else "test_r2"
    return out.sort_values(sort_key, ascending=False).reset_index(drop=True)






# PLOTTING

def plot_pred_vs_actual(model, X_test, y_test, ax=None):
    if ax is None:
        _, ax= plt.subplots(figsize= (5,5))
    pred= model.predict(X_test)
    ax.scatter(y_test, pred, alpha=0.6)
    lims= [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", label= "perfect")
    ax.set_xlabel("actual revenue"); ax.set_ylabel("predicted revenue")
    return ax




def plot_feature_importance(model, feature_names, top_n=15, ax=None):
    "Bar chart of the top-n feature importances for a tree-based model"
    if ax is None:
        _, ax= plt.subplots(figsize= (7,5))
    imp= pd.Series(model.feature_importances_, index= feature_names).sort_values(ascending=False).head(top_n)
    imp[::-1].plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} feature importances"); ax.set_xlabel("importance")
    return ax
