"""
Random Forest surrogate model.
Provides ensemble variance as an uncertainty estimate.
"""
import os
import numpy as np
import joblib
from models.base import SurrogateModel


class RandomForestSurrogate(SurrogateModel):
    def __init__(self, n_estimators=200, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt"):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        self._model = None

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        from sklearn.ensemble import RandomForestRegressor

        n_out = y_train.shape[1] if y_train.ndim > 1 else 1
        self._model = RandomForestRegressor(**self.params, n_jobs=-1, random_state=42)
        y_tr = y_train.ravel() if n_out == 1 else y_train
        self._model.fit(X_train, y_tr)
        return self

    def predict(self, X) -> np.ndarray:
        out = self._model.predict(X)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    def predict_with_uncertainty(self, X):
        """Uncertainty from ensemble variance across trees."""
        tree_preds = np.array([t.predict(X) for t in self._model.estimators_])
        # tree_preds: (n_trees, n_samples) or (n_trees, n_samples, n_outputs)
        if tree_preds.ndim == 2:
            tree_preds = tree_preds[:, :, np.newaxis]
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        return mean, std

    def supports_uncertainty(self) -> bool:
        return True

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self._model, os.path.join(directory, "model.pkl"))

    @classmethod
    def load(cls, directory: str) -> "RandomForestSurrogate":
        inst = cls()
        inst._model = joblib.load(os.path.join(directory, "model.pkl"))
        return inst

    def get_sklearn_estimator(self):
        return self._model

    @property
    def algo_name(self) -> str:
        return "Random Forest"
