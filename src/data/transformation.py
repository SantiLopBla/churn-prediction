import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

binary_columns = [
    "gender", "partner", "dependents", "phoneservice",
    "paperlessbilling", "churn"
]
categorical_columns = [
    "multiplelines", "internetservice", "onlinesecurity",
    "onlinebackup", "deviceprotection", "techsupport",
    "streamingtv", "streamingmovies", "contract", "paymentmethod"
]

def fix_column_types(df):
    df_copy = df.copy()
    if "totalcharges" in df_copy.columns:
        # errors='coerce' turns unparseable strings into NaN — handled later by imputation
        df_copy["totalcharges"] = pd.to_numeric(
            df_copy["totalcharges"], errors="coerce"
        )
    return df_copy

def encode_binary_columns(df):
    df_copy = df.copy()

    for col in binary_columns:
        if col in df_copy.columns:

            # count NaNs before mapping so we can detect if any values were lost
            nulls_before = df_copy[col].isnull().sum()

            # replace text values with numbers — "yes"/"male" become 1, "no"/"female" become 0
            # any value not in the dictionary (ex: "unknown") becomes NaN automatically
            df_copy[col] = df_copy[col].map(
                {"yes": 1, "no": 0, "male": 1, "female": 0}
            )

            # count NaNs again after mapping
            # if the number went up, some values didn't match the dictionary
            new_nulls = df_copy[col].isnull().sum() - nulls_before
            if new_nulls > 0:
                print(f"Warning: {new_nulls} values in '{col}' could not be mapped.")

    return df_copy

def encode_categorical_columns(df):
    df_copy = df.copy()
    cols = [col for col in categorical_columns if col in df_copy.columns]
    # drop_first=True avoids the dummy variable trap
    df_copy = pd.get_dummies(df_copy, columns=cols, drop_first=True)
    return df_copy

def engineer_features(df):
    df_copy = df.copy()
    df_copy["charges_per_tenure"]    = df_copy["monthlycharges"] / (df_copy["tenure"] + 1)
    df_copy["charge_to_total_ratio"] = df_copy["monthlycharges"] / (df_copy["totalcharges"] + 1)
    # flag customers in their first year — highest churn risk window
    df_copy["new_customer"]          = np.where(df_copy["tenure"] <= 12, 1, 0)
    return df_copy


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all transformations that do not require statistical fitting:
    column normalization, type casting, ID drop, encoding, feature engineering.
    Safe to apply to any subset (train, val, test) without leakage risk.
    """
    df_copy = df.copy()

    # normalize column names so everything downstream is consistent
    df_copy.columns = df_copy.columns.str.lower().str.strip()

    # lowercase string values so map() calls don't miss anything
    df_copy = df_copy.apply(
        lambda col: col.str.lower() if col.dtype == "object" else col
    )

    # drop customer ID — carries no predictive signal
    if "customerid" in df_copy.columns:
        df_copy = df_copy.drop(columns=["customerid"])
    else:
        print(f"Warning: 'customerid' not found. Available columns: {list(df_copy.columns)}")

    # order matters: cast types before encoding, encode before engineering
    df_copy = fix_column_types(df_copy)
    df_copy = encode_binary_columns(df_copy)
    df_copy = encode_categorical_columns(df_copy)
    df_copy = engineer_features(df_copy)

    return df_copy


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    """
    Learns the median of each column using only training data.
    That median is saved inside the imputer so it can be reused later
    on the CV and test sets without recalculating anything.
    """
    # strategy="median" — for each column, find the middle value in X_train
    # that value will be used to fill any NaN found in CV or test later
    imputer = SimpleImputer(strategy="median")

    # .fit() reads X_train and stores the median per column
    # it does NOT change the data — it just memorizes the values
    imputer.fit(X_train)

    return imputer


def apply_imputer(df: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    """
    Fills missing values using the medians the imputer already learned.
    Works on any subset — train, CV, or test — without learning anything new.
    """
    # .transform() fills NaNs using the stored medians — no new learning here
    # it returns a plain numpy array, so column names are lost
    imputed_array = imputer.transform(df)

    # rebuild the DataFrame so the rest of the pipeline still has column names
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)