# orchestrates the inference flow from raw csv to predictions
from src.data.ingestion import load_csv
from src.data.transformation import transform
from src.models.predict import predict


def run_inference_pipeline(filepath: str):
    df = load_csv(filepath)
    df = transform(df)

    # if the new dataframe has a "churn" column, it will drop it
    # this is because the model was trained without this column
    if "churn" in df.columns:
        df = df.drop(columns=["churn"])

    predictions = predict(df)
    return predictions