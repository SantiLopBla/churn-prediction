# columns the model needs to run
required_columns = [
    "customerID", "gender", "tenure",
    "MonthlyCharges", "TotalCharges", "Churn"
]

def validate(df):
    # catch empty files before they cause weird errors downstream
    if len(df) == 0:
        raise ValueError("Empty Dataframe")
    
    # make sure every required column actually exists
    for columns in required_columns:
        if columns not in df.columns:
            raise ValueError("There are missing required columns")