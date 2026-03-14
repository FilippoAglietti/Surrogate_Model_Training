"""
Abstract base class for all surrogate model algorithms.
"""
from abc import ABC, abstractmethod
import numpy as np


class SurrogateModel(ABC):
    """Common interface for all surrogate model algorithms."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        """Fit the model. Returns self."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predictions of shape (n_samples, n_outputs)."""
        pass

    def predict_with_uncertainty(self, X):
        """Return (mean, std) both of shape (n_samples, n_outputs).
        Default implementation: std = zeros."""
        mean = self.predict(X)
        return mean, np.zeros_like(mean)

    def supports_uncertainty(self) -> bool:
        """True if this model provides meaningful uncertainty estimates."""
        return False

    @abstractmethod
    def save(self, directory: str):
        """Save model artefacts into `directory`."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> "SurrogateModel":
        """Load from `directory`. Returns a new instance."""
        pass

    def get_sklearn_estimator(self):
        """Return the underlying sklearn/xgboost estimator for SHAP, or None."""
        return None

    @property
    def algo_name(self) -> str:
        return "Unknown"
