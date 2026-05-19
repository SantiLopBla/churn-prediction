import pandas as pd
import numpy as np

# two-value columns — mapped directly to 0/1
binary_columns = [
    "gender",
    "partner",
    "dependents",
    "phoneservice",
    "paperlessbilling",
    "churn"
]
# three or more values — handled with one-hot encoding
categorical_columns = [
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paymentmethod"
]

def fix_column_types(df):
    df_copy = df.copy()

    # totalcharges comes as object in the raw data — convert to numeric
    # errors='coerce' turns unparseable values into NaN (handled later by imputation)
    if "totalcharges" in df_copy.columns:
        df_copy["totalcharges"] = pd.to_numeric(df_copy["totalcharges"], errors="coerce")

    return df_copy

def impute_missing_values(df):
    df_copy = df.copy()

    for col in df_copy.columns:
        # robust way to detect numeric columns
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            # numeric: fill with median to avoid outlier influence
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        else:
            # categorical: fill with most frequent value
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    return df_copy

def encode_categorical_columns(df):
    df_copy = df.copy()
    
    # only encode columns that actually exist in the dataframe
    cols = [col for col in categorical_columns if col in df_copy.columns]
    
    # drop_first avoids the dummy variable trap
    df_copy = pd.get_dummies(df_copy, columns=cols, drop_first=True)
    
    return df_copy


def encode_binary_columns(df):
    df_copy = df.copy()

    for col in binary_columns:
        if col in df_copy.columns:
            
            # count nulls before mapping to use as baseline
            nulls_before = df[col].isnull().sum()

            # standardized map covers both yes/no and male/female
            df_copy[col] = df_copy[col].map({"yes": 1, "no": 0, "male": 1, "female": 0})

            # count nulls after mapping to detect values that didn't match
            nulls_after = df_copy[col].isnull().sum()

            # difference reveals how many values the map couldn't handle
            new_nulls = nulls_after - nulls_before

            # warn if any values were lost during mapping
            if new_nulls > 0:
                print(f"Warning: {new_nulls} values in '{col}' could not be mapped.")

    return df_copy

def engineer_features (df):
    
    #Create new columns
    df_copy = df.copy()
    df_copy["charges_per_tenure"] = df_copy["monthlycharges"] / (df_copy["tenure"] + 1)
    df_copy["charge_to_total_ratio"] = df_copy["monthlycharges"] / (df_copy["totalcharges"] + 1)
    df_copy["new_customer"] = np.where(df_copy["tenure"] <= 12, 1, 0)

    return df_copy

def transform(df_copy):

    # normalize column names first so everything downstream is consistent
    df_copy.columns = df_copy.columns.str.lower().str.strip()

    # lowercase string values so the map() calls don't miss anything
    df_copy = df_copy.apply(
        lambda col: col.str.lower() if col.dtype == "object" else col
    )

    if "customerid" in df_copy.columns:
        df_copy = df_copy.drop(columns=["customerid"])
    else:
        print(f"Warning: 'customerid' not found. Available columns: {list(df_copy.columns)}")

    # the secuence needs to be in this exact same order for correct data cleaning
    df_copy = fix_column_types(df_copy)
    df_copy = impute_missing_values(df_copy)
    df_copy = encode_binary_columns(df_copy)
    df_copy = encode_categorical_columns(df_copy)
    
    df_copy = engineer_features(df_copy)
    
    print(f"Transformation complete. Final shape: {df_copy.shape}")

    return df_copy