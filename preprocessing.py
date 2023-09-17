import pandas as pd

from extract import extract
from measure_utilities import measure_utilities


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    extract(df)
    measure_utilities(df)
    return df
