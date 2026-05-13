# orchestrates the full training flow from raw csv to evaluated model
from src.data.ingestion import load_csv
from src.data.validation import validate
from src.data.transformation import transform
from src.models.train import train
from src.models.tune import tune_xgboost
from src.models.evaluate import evaluate

# This function trains the models with the data
def run_training_pipeline(filepath: str) -> None:
    # Loads,validates and transform the Dataframe
    df = load_csv(filepath) 
    validate(df)
    df = transform(df)

    # Separete data to train. Order: Logistic Regression, Random Forest, XGBoost.
    # It is separated in training set, cross-validation set and testing set
    lr, rf, xgb, X_train, y_train, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test = train(df)

    # tune_xgb is replacing xgb "base model"
    tuned_xgb = tune_xgboost(X_train, y_train)

    # Evaluates all the models. Order: Logistic Regression, Random Forest, XGBoost.
    evaluate(lr, rf, tuned_xgb, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test)