import pandas as pd

from mine import extract
from measure_utilities import measure_utilities
from model_fitting import train_models
from price_prediction import predict_property_prices


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = extract(df)
    df = measure_utilities(df)
    return df


def main():
    df_train = pd.read_csv("input/training_data.csv")
    df_train = preprocessing(df_train)

    train_models(df_train)

    df_test = pd.read_csv("input/public_dataset.csv")
    df_test = preprocessing(df_test)

    predict_property_prices(df_test)


if __name__ == '__main__':
    main()
