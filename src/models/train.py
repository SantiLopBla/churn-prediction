import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

from src.data.transformation import fit_imputer, apply_imputer

def train(df: pd.DataFrame):

    # X = everything the model uses to learn (all columns except churn)
    # y = the column the model tries to predict (1 = churned, 0 = stayed)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # split the data BEFORE doing anything else — this prevents data leakage
    # stratify=y makes sure every split has the same churn percentage as the original
    # result: 80% goes to train, 20% goes to a temporary holdout
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # split the 20% holdout in half — 10% for CV, 10% for test
    # CV is used to tune and compare models, test is only touched at the very end
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # learn the median of each column using only X_train rows
    # this median will be used to fill NaNs in all three subsets
    imputer = fit_imputer(X_train)

    # fill missing values in all three subsets using the same median learned above
    # none of these calls re-learn anything — they just apply the stored values
    X_train = apply_imputer(X_train, imputer)
    X_cv    = apply_imputer(X_cv,    imputer)
    X_test  = apply_imputer(X_test,  imputer)

    # logistic regression is sensitive to feature scale (e.g. tenure vs monthly charges)
    # StandardScaler fixes this by converting every column to mean=0 and std=1
    # fit_transform: learn the mean and std from X_train, then scale it
    # transform: apply that same mean and std to CV and test — no re-learning
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_cv_lr    = scaler.transform(X_cv)
    X_test_lr  = scaler.transform(X_test)

    # logistic regression — needs scaled data (X_train_lr)
    # class_weight="balanced" tells the model to pay more attention to churners
    # because the dataset has ~3x more non-churners than churners
    lr = LogisticRegression(
        max_iter=15000, random_state=42, class_weight="balanced"
    )
    lr.fit(X_train_lr, y_train)

    # random forest — does not need scaled data, works directly on X_train
    # it makes decisions based on value thresholds, not distances
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # xgboost — also does not need scaled data
    # scale_pos_weight: tells xgboost how much to penalize missing a churner
    # calculated as: number of non-churners / number of churners
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)

    # save everything to disk — models, scaler, and imputer are all needed
    # when the pipeline runs on new data later
    joblib.dump(lr,      "models/logistic_regression.pkl")
    joblib.dump(rf,      "models/random_forest.pkl")
    joblib.dump(xgb,     "models/xgboost.pkl")
    joblib.dump(scaler,  "models/scaler.pkl")
    joblib.dump(imputer, "models/imputer.pkl")

    return lr, rf, xgb, X_train, y_train, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test