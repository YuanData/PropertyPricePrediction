import math

import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox


def normalize_data(df: pd.DataFrame):
    df['UnitPrice'] = np.log1p(df['UnitPrice'])  # Log Transform target value
    # Transform other numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numerical_cols].apply(lambda x: skew(x.dropna()))
    skew_threshold = 0.75
    skewed_cols = skewness[abs(skewness) > skew_threshold].index

    for col in skewed_cols:
        if col == 'X':
            continue
        min_value = df[col].min()
        if min_value < 0:  # Offset if minimum value is negative
            df[col] += abs(min_value) + 1  # Adding 1 to ensure positive values

        # Box-Cox Transform (now using the lambda value)
        df[f'{col}_transformed'], _ = boxcox(df[col] + 1)  # Adding 1 to ensure positive values


def geo_transform(x, y):
    a, b, long_0, dx = 6378137, 6356752, 121 * math.pi / 180, 250000
    e = math.sqrt(1 - (b / a) ** 2)
    x, m = x - dx, y
    mu = m / (a * (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256))
    e1 = (1 - math.sqrt(1 - e ** 2)) / (1 + math.sqrt(1 - e ** 2))
    j_values = [3 * e1 / 2 - 27 * e1 ** 3 / 32, 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32, 151 * e1 ** 3 / 96,
                1097 * e1 ** 4 / 512]
    fp = mu + sum(j * math.sin(2 * (i + 1) * mu) for i, j in enumerate(j_values))
    e2 = e ** 2 * (a / b) ** 2
    c1 = e2 * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    n1 = a / math.sqrt(1 - e ** 2 * math.sin(fp) ** 2)
    d = x / n1
    latitude_correction = n1 * math.tan(fp) / (a * (1 - e ** 2) / pow(1 - e ** 2 * math.sin(fp) ** 2, 1.5))
    lat = fp - latitude_correction * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e2) * d ** 4 / 24 + (
            61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 3 * c1 ** 2 - 252 * e2) * d ** 6 / 720)
    lon = long_0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (
            5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(fp)
    return math.degrees(lon), math.degrees(lat)


def gen_categorical_column(df: pd.DataFrame, source_col: str, bin_start: int, bin_end: int, bin_step: int,
                           num_labels: int):
    bins = range(bin_start, bin_end, bin_step)
    labels = range(num_labels)
    df[f'{source_col}_category'] = pd.cut(df[source_col], bins=bins, labels=labels, right=False)


def categorize_data(df: pd.DataFrame):
    gen_categorical_column(df, 'TransactionFloor', 0, 30, 10, 5)
    gen_categorical_column(df, 'TotalFloors', 0, 30, 10, 5)
    gen_categorical_column(df, 'BuildingAge', 0, 50, 10, 5)


def gen_label_by_group_mean(df: pd.DataFrame):
    for col in ['CountyCity', 'MainUsage', 'MainMaterial', 'ConstructionType']:
        df[f'{col}_mean_label'] = df.groupby(col)['UnitPrice'].transform('mean')


def extract(df: pd.DataFrame) -> pd.DataFrame:
    df['Lon'], df['Lat'] = zip(*df.apply(lambda row: geo_transform(row['X'], row['Y']), axis=1))
    normalize_data(df)
    categorize_data(df)
    gen_label_by_group_mean(df)
    return df
