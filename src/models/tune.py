from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    # candidate values to try for each hyperparameter
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }

    # base model — scale_pos_weight compensates for class imbalance
    base_model = XGBClassifier(
        random_state=42,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        eval_metric="logloss",
        verbosity=0
    )

    # GridSearchCV tries every combination in param_grid
    # cv=3: evaluates each combination with 3-fold cross-validation
    # scoring="f1": selects the best combination based on F1 score
    # n_jobs=-1: uses all available CPU cores to run in parallel
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    # print the winning combination
    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV F1:      {search.best_score_:.4f}")

    # save the best model to disk
    joblib.dump(search.best_estimator_, "models/xgboost.pkl")

    return search.best_estimator_