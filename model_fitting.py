import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


class HybridEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_models_ = []
        self.meta_model_ = None

    def fit(self, X, y):
        self.base_models_ = [clone(model) for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models_):
            for train_index, holdout_index in kfold.split(X):
                model.fit(X[train_index], y[train_index])
                y_pred = model.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_
        ]).mean(axis=1)
        return self.meta_model_.predict(meta_features.reshape(-1, 1))


def calc_mape(y_true, y_pred):
    return mean_absolute_percentage_error(np.expm1(y_true), np.expm1(y_pred))


def extract_features_and_target(df_data: pd.DataFrame, used_cols: list, target_col: str):
    df_x = df_data[used_cols]
    df_y = df_data[target_col]
    return df_x.values, df_y.values


def build_model(model_name: str, params: dict, model_registry: dict):
    model_class = model_registry[model_name]
    if model_name in ['Lasso', 'KernelRidge', 'ElasticNet']:
        return make_pipeline(RobustScaler(), model_class(**params))
    else:
        return model_class(**params)


def train_and_evaluate(model, train_x, train_y, kf, scorer):
    scores = cross_val_score(model, train_x, train_y, cv=kf, scoring=scorer, n_jobs=4)
    print(f'Model: {type(model).__name__}')
    print(f'MAPE {scores=}')
    print(f'Mean MAPE: {np.mean(scores)}')
    print(f'Std Dev MAPE: {np.std(scores)}')
    model.fit(train_x, train_y)
    return model


def train_models(df: pd.DataFrame):
    used_cols = [
        'CountyCity_label', 'CountyCity鄉鎮市區_label', 'Lon_transformed', 'Lat_transformed',
        'LandArea_transformed', 'TotalBuildingArea_transformed', 'TransactionFloor_category', 'TotalFloors_category',
        'BuildingAge_category', 'MainUsage_label', 'MainMaterial_label', 'ConstructionType_label', 'has_parking',
        '車位面積_transformed',
    ]
    target_col = 'UnitPrice'
    train_x, train_y = extract_features_and_target(df, used_cols, target_col)

    kf = KFold(n_splits=8, shuffle=True, random_state=1)
    mape_scorer = make_scorer(calc_mape, greater_is_better=False)

    models_params = {
        'XGBoost': {
            'colsample_bytree': 0.6,
            'gamma': 0.001,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 2,
            'n_estimators': 350,
            'reg_alpha': 0.003,
            'reg_lambda': 0.01,
            'subsample': 0.75
        },
        'Lasso': {'alpha': 1e-9, 'max_iter': 1500},
        'KernelRidge': {'alpha': 1e-06, 'coef0': 5, 'degree': 3, 'gamma': 0.004},
        'ElasticNet': {'alpha': 1e-06, 'l1_ratio': 0.8},
        'GradientBoosting': {
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_samples_leaf': 0.0005,
            'min_samples_split': 0.0015,
            'n_estimators': 850,
            'subsample': 0.65
        }
    }

    model_registry = {
        'XGBoost': XGBRegressor,
        'Lasso': Lasso,
        'KernelRidge': KernelRidge,
        'ElasticNet': ElasticNet,
        'GradientBoosting': GradientBoostingRegressor
    }

    models = {}
    for model_name, params in models_params.items():
        model = build_model(model_name, params, model_registry)
        models[model_name] = train_and_evaluate(model, train_x, train_y, kf, mape_scorer)

        with open(f'output/models/{model_name}_model.pkl', 'wb') as file:
            pickle.dump(model, file)

    base_models = [models[m] for m in ['ElasticNet', 'GradientBoosting', 'KernelRidge']]
    meta_model = models['Lasso']
    hybrid_model = HybridEnsembleRegressor(base_models=base_models, meta_model=meta_model)
    hybrid_model = train_and_evaluate(hybrid_model, train_x, train_y, kf, mape_scorer)
    with open('output/models/hybrid_model.pkl', 'wb') as file:
        pickle.dump(hybrid_model, file)

    hybrid_pred = hybrid_model.predict(train_x)
    hybrid_mape = calc_mape(train_y, hybrid_pred)
    print(f'MAPE for {hybrid_mape=}')

    xgb_pred = models['XGBoost'].predict(train_x)
    xgb_mape = calc_mape(train_y, xgb_pred)
    print(f'MAPE for {xgb_mape=}')
