# pandas for data manipulation, scikit-learn for model training
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def train(df: pd.DataFrame):
    # separate features from target column
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # first split: reserve 20% as temporary holdout
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # second split: divide holdout evenly into cv (Cross Validation) and test (10% each)
    # cv is used to tune decisions during development
    # test is used only once to report final performance
    X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # initialize logistic regression
    # max_iter=1000 prevents convergence warnings on larger datasets
    # random_state=42 ensures reproducibility
    model = LogisticRegression(max_iter=15000, random_state=42, class_weight="balanced")

    # train the model on training data only — cv and test are never seen here
    model.fit(X_train, y_train)

    # save trained model to disk for later use in prediction
    joblib.dump(model, "models/churn_model.pkl")

    # return model and evaluation sets for use in evaluate.py and predict.py
    return model, X_cv, y_cv, X_test, y_test