import pandas as pd

# columns the model needs to run
required_columns = [
    "customerID",
    "gender",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

def validate(df: pd.DataFrame) -> None:
    # catch empty files before they cause weird errors downstream
    if len(df) == 0:
        raise ValueError("Empty Dataframe")
    
    # make sure every required column actually exists
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")