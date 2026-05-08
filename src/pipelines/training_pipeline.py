# orchestrates the full training flow from raw csv to evaluated model
from src.data.ingestion import load_csv
from src.data.validation import validate
from src.data.transformation import transform
from src.models.train import train
from src.models.evaluate import evaluate


def run_training_pipeline(filepath: str) -> None:
    # load raw data from disk
    df = load_csv(filepath)

    # ensure data meets minimum quality requirements before processing
    validate(df)

    # encode and clean features for model consumption
    df = transform(df)

    # train model and split data into cv and test sets
    lr, rf, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test = train(df)

    # report precision, recall and f1 to assess bias and variance
    evaluate(lr, rf, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test)