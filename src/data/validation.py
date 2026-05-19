import pandas as pd

# columns the model needs to run
required_columns = [
    "customerid",
    "gender",
    "tenure",
    "monthlycharges",
    "totalcharges",
    "churn"
]

def validate(df: pd.DataFrame) -> None:
    # catch empty files before they cause weird errors downstream
    if len(df) == 0:
        raise ValueError("Empty DataFrame")
    
    normalized_columns = [col.lower().strip() for col in df.columns]
    
    # make sure every required column actually exists
    for col in required_columns:
        if col not in normalized_columns:
            raise ValueError(f"Missing required column: '{col}'")