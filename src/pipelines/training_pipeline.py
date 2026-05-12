# orchestrates the full training flow from raw csv to evaluated model
from src.data.ingestion import load_csv
from src.data.validation import validate
from src.data.transformation import transform
from src.models.train import train
from src.models.tune import tune_xgboost
from src.models.evaluate import evaluate


def run_training_pipeline(filepath: str) -> None:
    # load raw data from disk
    df = load_csv(filepath)

    # ensure data meets minimum quality requirements before processing
    validate(df)

    # encode and clean features for model consumption
    df = transform(df)

    # train all models — returns X_train and y_train for tuning
    lr, rf, xgb, X_train, y_train, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test = train(df)

    # tune xgboost using the same X_train from the training split
    xgb = tune_xgboost(X_train, y_train)

    # report metrics for all three models
    evaluate(lr, rf, xgb, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test)