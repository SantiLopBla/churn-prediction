# orchestrates the inference flow from raw csv to predictions
from src.data.ingestion import load_csv
from src.data.transformation import transform
from src.models.predict import predict


def run_inference_pipeline(filepath: str) -> None:
    # load new data from disk
    df = load_csv(filepath)

    # encode and clean features to match training format
    df = transform(df)

    # generate churn predictions using the saved model
    predictions = predict(df)

    return predictions