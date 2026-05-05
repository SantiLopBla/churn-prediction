import pandas as pd 

def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    return df