# pandas for data manipulation, scikit-learn for model training
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def train(df: pd.DataFrame):
    # separate features from target column
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # scale features for logistic regression only
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_cv_lr    = scaler.transform(X_cv)
    X_test_lr  = scaler.transform(X_test)

    # logistic regression uses scaled data
    lr = LogisticRegression(max_iter=15000, random_state=42, class_weight="balanced")
    lr.fit(X_train_lr, y_train)

    # random forest uses original unscaled data
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    # save both models and scaler to disk
    joblib.dump(lr, "models/logistic_regression.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")  

    # return both models and evaluation sets
    return lr, rf, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test