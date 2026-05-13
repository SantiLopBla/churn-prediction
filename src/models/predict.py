import joblib
import pandas as pd
import numpy as np

# the dictionary maps if the models requires scaling or not
# It´s better because new models just needs to be added here
MODELS_CONFIG = {
    "logistic_regression": {"requires_scaling": True},
    "random_forest":       {"requires_scaling": False},
    "xgboost":             {"requires_scaling": False},
}

def predict(df: pd.DataFrame, model_name: str = "xgboost") -> np.ndarray:
    # Validates if the dataframe is empty or not
    if len(df) == 0:
        raise ValueError("Empty DataFrame: no data to predict on")

    # It validates if the model is in the dictionary MODELS_CONFIG
    if model_name not in MODELS_CONFIG:
        valid = list(MODELS_CONFIG.keys())
        raise ValueError(f"Invalid model name. Choose from: {valid}")

    # Uploads the model from disk
    model = joblib.load(f"models/{model_name}.pkl")

    # Applies scaling if the model needs it. Prior indexed in the MODELS_CONFIG dictionary
    if MODELS_CONFIG[model_name]["requires_scaling"]:
        scaler = joblib.load("models/scaler.pkl")
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return model.predict(df)