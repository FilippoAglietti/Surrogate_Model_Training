"""
Neural Network surrogate model — thin wrapper around a trained Keras model.
Training is driven by the ModelBuilderFrame UI thread; this class handles
predict/save/load so the rest of the app can treat it uniformly.
"""
import os
import numpy as np
from models.base import SurrogateModel


class NeuralNetworkSurrogate(SurrogateModel):
    def __init__(self, keras_model=None):
        self._model = keras_model

    # ── SurrogateModel interface ──────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        raise NotImplementedError(
            "NN training is driven by the ModelBuilderFrame UI thread."
        )

    def predict(self, X) -> np.ndarray:
        out = self._model.predict(X, verbose=0)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    def supports_uncertainty(self) -> bool:
        return False

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self._model.save(os.path.join(directory, "model.keras"))

    @classmethod
    def load(cls, directory: str) -> "NeuralNetworkSurrogate":
        from tensorflow.keras.models import load_model
        model = load_model(os.path.join(directory, "model.keras"))
        return cls(model)

    def get_sklearn_estimator(self):
        return None

    @property
    def algo_name(self) -> str:
        return "Neural Network"
