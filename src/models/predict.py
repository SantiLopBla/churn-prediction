import joblib
import pandas as pd

def predict(df: pd.DataFrame) -> list:
    # reject empty dataframes before loading the model
    if len(df) == 0:
        raise ValueError("Empty DataFrame: no data to predict on")

    # load the trained model from disk
    model = joblib.load("models/churn_model.pkl")

    # generate predictions on new data
    return model.predict(df)