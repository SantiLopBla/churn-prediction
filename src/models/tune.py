from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:

    # list of values to try for each hyperparameter
    # GridSearchCV will test every possible combination of these values
    param_grid = {
        "n_estimators":  [100, 200, 300],  # number of trees to build
        "max_depth":     [3, 5, 7],        # how deep each tree can grow
        "learning_rate": [0.01, 0.1, 0.2], # how much each tree corrects the previous one
    }

    # base model with no hyperparameters set yet — GridSearchCV will fill those in
    # scale_pos_weight tells xgboost to penalize missed churners more
    # calculated as: number of non-churners / number of churners
    base_model = XGBClassifier(
        random_state=42,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        eval_metric="logloss",
        verbosity=0
    )

    # GridSearchCV trains a model for every combination in param_grid
    # with 3 values per parameter and 3 parameters, that is 27 combinations
    # each combination is evaluated 3 times (cv=3), so 81 models total
    # scoring="f1": the combination with the highest F1 on CV wins
    # n_jobs=-1: runs all combinations in parallel using every CPU core
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    # train all 81 models and find the best combination
    search.fit(X_train, y_train)

    # print the hyperparameter combination that scored highest
    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV F1:      {search.best_score_:.4f}")

    # save the best model to disk so it can be used later without retraining
    joblib.dump(search.best_estimator_, "models/xgboost.pkl")

    return search.best_estimator_