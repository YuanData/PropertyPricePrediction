import pandas as pd

from model_fitting import train_models
from preprocessing import preprocessing
from price_prediction import predict_property_prices

if __name__ == '__main__':
    df_train = pd.read_csv("input/training_data.csv")
    df_train = preprocessing(df_train)

    train_models(df_train)

    df_test = pd.read_csv("input/public_dataset.csv")
    df_test = preprocessing(df_test)

    predict_property_prices(df_test)
