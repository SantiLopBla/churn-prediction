import pandas as pd
from pathlib import Path


def load_csv(filepath: str) -> pd.DataFrame:
    path = Path(filepath)

    # verify the file exists before attempting to read
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # verify the file is actually a CSV
    if path.suffix != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    df = pd.read_csv(path)

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df