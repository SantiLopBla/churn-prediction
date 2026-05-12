import joblib
import pandas as pd
import numpy as np

def predict(df: pd.DataFrame, model_name: str = "random_forest") -> np.ndarray:
    # reject empty dataframes before loading the model
    if len(df) == 0:
        raise ValueError("Empty DataFrame: no data to predict on")

    # validate model name before attempting to load
    valid_models = ["logistic_regression", "random_forest"]
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name. Choose from: {valid_models}")

    # load the requested model from disk
    model = joblib.load(f"models/{model_name}.pkl")

    # logistic regression requires scaled input — apply the saved scaler
    if model_name == "logistic_regression":
        scaler = joblib.load("models/scaler.pkl")
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return model.predict(df)