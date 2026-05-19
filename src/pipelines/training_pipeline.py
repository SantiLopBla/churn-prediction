# runs the full pipeline from raw CSV to trained and evaluated models
from src.data.ingestion import load_csv
from src.data.validation import validate
from src.data.transformation import clean_and_encode
from src.models.train import train
from src.models.tune import tune_xgboost
from src.models.evaluate import evaluate

def run_training_pipeline(filepath: str) -> None:

    # load the CSV file into a DataFrame
    df = load_csv(filepath)

    # check that all required columns exist before doing anything else
    validate(df)

    # clean and encode the data — rename columns, fix types, encode categories
    # this step does NOT learn any statistics, so it is safe to run on the full dataset
    df = clean_and_encode(df)

    # split the data and train all three models (LR, RF, XGBoost)
    # imputation happens inside train() — after the split, not before
    lr, rf, xgb, X_train, y_train, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test = train(df)

    # search for the best hyperparameters for XGBoost using cross-validation
    # tuned_xgb replaces the base xgb model in the evaluation step
    tuned_xgb = tune_xgboost(X_train, y_train)

    # print metrics and confusion matrices for all three models
    evaluate(lr, rf, tuned_xgb, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test)