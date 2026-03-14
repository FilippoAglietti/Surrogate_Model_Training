"""
XGBoost surrogate model.
For multi-output targets, wraps XGBRegressor in sklearn's MultiOutputRegressor.
"""
import os
import numpy as np
import joblib
from models.base import SurrogateModel


class XGBoostSurrogate(SurrogateModel):
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        )
        self._model = None
        self._n_out = 1

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor

        self._n_out = y_train.shape[1] if y_train.ndim > 1 else 1
        base = XGBRegressor(
            **self.params, objective="reg:squarederror",
            eval_metric="rmse", verbosity=0, n_jobs=-1
        )

        if self._n_out == 1:
            self._model = base
            self._model.fit(
                X_train, y_train.ravel(),
                eval_set=[(X_val, y_val.ravel())],
                verbose=False
            )
        else:
            self._model = MultiOutputRegressor(base, n_jobs=1)
            self._model.fit(X_train, y_train)

        return self

    def predict(self, X) -> np.ndarray:
        out = self._model.predict(X)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self._model, os.path.join(directory, "model.pkl"))
        joblib.dump(self._n_out, os.path.join(directory, "n_out.pkl"))

    @classmethod
    def load(cls, directory: str) -> "XGBoostSurrogate":
        inst = cls()
        inst._model = joblib.load(os.path.join(directory, "model.pkl"))
        try:
            inst._n_out = joblib.load(os.path.join(directory, "n_out.pkl"))
        except Exception:
            inst._n_out = 1
        return inst

    def get_sklearn_estimator(self):
        return self._model

    @property
    def algo_name(self) -> str:
        return "XGBoost"
