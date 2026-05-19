import pandas as pd
from src.data.transformation import fix_column_types, encode_binary_columns,engineer_features


def test_fix_column_types():
    # Fake DataFrame
    df = pd.DataFrame({
        "totalcharges": ["50.5", "100.0", "200.75"]
    })

    # Saves in the variable the fixed DataFrame
    result = fix_column_types(df)

    # It is waiting that "result" is a float64
    assert result["totalcharges"].dtype == "float64"

def test_encode_binary_columns():
    #Fake DataFrame
    df=pd.DataFrame({
        "gender":["male","female"]
    })

    # Saves in the variable the fixed DataFrame
    result=encode_binary_columns(df)
    
    # It is waiting that "result" is 1 or 0
    assert result ["gender"].tolist() == [1, 0]

def test_engineer_features():
    #Fake DataFrame
    df = pd.DataFrame({
        "monthlycharges": [50.0],
        "tenure": [6],
        "totalcharges": [300.0]
    })

    # Saves in the variable the fixed DataFrame
    result=engineer_features(df)

    #Verifies if the columns exists
    assert "charges_per_tenure" in result.columns
    assert "charge_to_total_ratio" in result.columns
    assert "new_customer" in result.columns
