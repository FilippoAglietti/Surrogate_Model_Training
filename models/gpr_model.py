"""
Gaussian Process Regression surrogate model.
Provides predictive mean + std (calibrated uncertainty).
For multi-output: wraps in MultiOutputRegressor (one GP per output).
"""
import os
import numpy as np
import joblib
from models.base import SurrogateModel


class GPRSurrogate(SurrogateModel):
    def __init__(self, kernel="RBF", alpha=1e-6, n_restarts=3, normalize_y=True):
        self.kernel_name = kernel
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y
        self._model = None
        self._n_out = 1

    def _build_kernel(self):
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, RationalQuadratic, ConstantKernel
        )
        k_map = {
            "RBF": ConstantKernel(1.0) * RBF(1.0),
            "Matern": ConstantKernel(1.0) * Matern(nu=1.5),
            "RationalQuadratic": ConstantKernel(1.0) * RationalQuadratic(),
        }
        return k_map.get(self.kernel_name, ConstantKernel(1.0) * RBF(1.0))

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.multioutput import MultiOutputRegressor

        self._n_out = y_train.shape[1] if y_train.ndim > 1 else 1
        gpr = GaussianProcessRegressor(
            kernel=self._build_kernel(), alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts, normalize_y=self.normalize_y
        )

        if self._n_out == 1:
            self._model = gpr
            self._model.fit(X_train, y_train.ravel())
        else:
            self._model = MultiOutputRegressor(gpr, n_jobs=1)
            self._model.fit(X_train, y_train)

        return self

    def predict(self, X) -> np.ndarray:
        if self._n_out == 1:
            out = self._model.predict(X)
            return out.reshape(-1, 1)
        else:
            return self._model.predict(X)

    def predict_with_uncertainty(self, X):
        if self._n_out == 1:
            mean, std = self._model.predict(X, return_std=True)
            return mean.reshape(-1, 1), std.reshape(-1, 1)
        else:
            means, stds = [], []
            for est in self._model.estimators_:
                m, s = est.predict(X, return_std=True)
                means.append(m)
                stds.append(s)
            return np.column_stack(means), np.column_stack(stds)

    def supports_uncertainty(self) -> bool:
        return True

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self._model, os.path.join(directory, "model.pkl"))
        joblib.dump(self._n_out, os.path.join(directory, "n_out.pkl"))

    @classmethod
    def load(cls, directory: str) -> "GPRSurrogate":
        inst = cls()
        inst._model = joblib.load(os.path.join(directory, "model.pkl"))
        try:
            inst._n_out = joblib.load(os.path.join(directory, "n_out.pkl"))
        except Exception:
            inst._n_out = 1
        return inst

    def get_sklearn_estimator(self):
        return None  # KernelExplainer will be used for SHAP

    @property
    def algo_name(self) -> str:
        return "Gaussian Process"
