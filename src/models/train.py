import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

def train(df: pd.DataFrame):
    # X: all columns the model uses to learn
    # y: the column the model tries to predict (1=churn, 0=no churn)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # first split: 80% train, 20% temp
    # stratify=y keeps the same churn ratio in both splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # second split: divides the 20% into cv and test equally
    # final breakdown: 80% train / 10% cv / 10% test
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # scaler learns mean and std from X_train only — never from cv or test
    # fit_transform: learns AND transforms
    # transform: applies the same learned values without re-learning
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_cv_lr    = scaler.transform(X_cv)
    X_test_lr  = scaler.transform(X_test)

    # logistic regression requires scaled data — sensitive to feature magnitude
    # class_weight="balanced": compensates for churn/no-churn class imbalance
    lr = LogisticRegression(max_iter=15000, random_state=42, class_weight="balanced")
    lr.fit(X_train_lr, y_train)

    # random forest is scale-invariant — splits on thresholds, not magnitudes
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    # xgboost is also scale-invariant
    # scale_pos_weight: ratio of negatives to positives — penalizes churn errors more
    # eval_metric="logloss": internal error metric for binary classification
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)

    # serialize all models and scaler to disk for later use
    joblib.dump(lr, "models/logistic_regression.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xgb, "models/xgboost.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return lr, rf, xgb, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test