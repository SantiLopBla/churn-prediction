import pandas as pd

# two-value columns — mapped directly to 0/1
binary_columns = [
    "gender",
    "discount_applied",
    "price_increase_last_3m"
]

# three or more values — handled with one-hot encoding
categorical_columns = [
    "country",
    "city", 
    "customer_segment",
    "signup_channel",
    "contract_type",
    "payment_method",
    "complaint_type",
    "survey_response"
]


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
            # standardized map covers both yes/no and male/female
            df_copy[col] = df_copy[col].map({"yes": 1, "no": 0, "male": 1, "female": 0})
    
    return df_copy


def transform(df):
    df_copy = df.copy()

    # normalize column names first so everything downstream is consistent
    df_copy.columns = df_copy.columns.str.lower().str.strip()

    # lowercase string values so the map() calls don't miss anything
    df_copy = df_copy.apply(
        lambda col: col.str.lower() if col.dtype == "object" else col
    )

    # identifier column — no predictive value
    df_copy = df_copy.drop(columns=["customer_id"])

    df_copy = encode_binary_columns(df_copy)
    df_copy = encode_categorical_columns(df_copy)

    print(f"Transformación completa. Shape final: {df_copy.shape}")

    return df_copy