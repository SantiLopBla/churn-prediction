# runs the full pipeline from raw CSV to predictions on new data
from src.data.ingestion import load_csv
from src.data.transformation import clean_and_encode, apply_imputer
from src.models.predict import predict
import joblib

def run_inference_pipeline(filepath: str):

    # load the new data from a CSV file into a DataFrame
    df = load_csv(filepath)

    # apply the same cleaning and encoding used during training
    # this ensures the columns and values match what the model expects
    df = clean_and_encode(df)

    # load the imputer that was saved during training
    # it already knows the median of each column from the training data
    # using the same imputer guarantees new data is filled the same way
    imputer = joblib.load("models/imputer.pkl")
    df = apply_imputer(df, imputer)

    # remove the churn column if it exists in the new data
    # the model only takes features as input — not the answer
    if "churn" in df.columns:
        df = df.drop(columns=["churn"])

    return predict(df)