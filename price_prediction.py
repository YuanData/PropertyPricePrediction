import pickle

import numpy as np
import pandas as pd


def predict_property_prices(df: pd.DataFrame):
    # Load models
    with open('output/models/hybrid_model.pkl', 'rb') as file:
        hybrid_model = pickle.load(file)
    with open('output/models/XGBoost_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    # Select relevant columns for prediction
    prediction_features = [
        'CountyCity', 'Lon', 'Lat', 'LandArea', 'TotalBuildingArea',
        'TransactionFloor', 'TotalFloors', 'BuildingAge', 'MainUsage',
        'MainMaterial', 'ConstructionType', 'has_ConvenienceStore',
        'has_MetroStation', 'has_University', 'has_MedicalInstitution',
    ]
    feature_data = df.loc[:, prediction_features]  # Extracting relevant features

    # Convert DataFrame to NumPy array for model input
    feature_array = feature_data.values

    # Predict using the hybrid model
    hybrid_pred = hybrid_model.predict(feature_array)

    # Predict using the XGBoost model
    xgb_pred = xgb_model.predict(feature_array)

    # Revert log transformation on predictions
    hybrid_pred_restored = np.expm1(hybrid_pred)
    xgb_pred_restored = np.expm1(xgb_pred)

    # Combine predictions from both models
    final_pred = (hybrid_pred_restored + xgb_pred_restored) / 2

    # Prepare and save the submission file
    df_submission = pd.DataFrame(
        {'ID': df['ID'].values, 'predicted_price': final_pred})  # Creating DataFrame for submission
    df_submission.to_csv(r'output/prediction/submission.csv', index=False)  # Saving submission file
