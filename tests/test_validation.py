import pytest as pt #Pytest to detect and run test automatically
import pandas as pd
from src.data.validation import validate
from src.data.transformation import fix_column_types

def test_validate_empty_dataframe():
    #validates if the DataFrame is empty
    df = pd.DataFrame()

    with pt.raises(ValueError, match="Empty"):
        validate(df)

def test_validate_missing_column():
    # churn is required — without it train() has no target to learn from
    df = pd.DataFrame({
        "customerid": [1],
        "gender": ["male"],
        "tenure": [12],
        "monthlycharges": [50.0],
        "totalcharges": [600.0]
    
    }) 
    with pt.raises(ValueError, match="churn"):
        validate(df)

