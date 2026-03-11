"""
Session state management utilities.
Centralized helpers for safe state access across modules.
"""
import streamlit as st
from typing import Any


def init_state(key: str, default: Any):
    """Initialize a session state key if it doesn't exist."""
    if key not in st.session_state:
        st.session_state[key] = default


def get_state(key: str, default: Any = None) -> Any:
    """Safely get a session state value."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """Set a session state value."""
    st.session_state[key] = value


def init_all_defaults():
    """Initialize all default session state values for the app."""
    defaults = {
        # Data loading
        "raw_data": None,
        "input_columns": [],
        "output_column": None,

        # Preprocessing
        "preprocessed": False,
        "X_train": None, "X_val": None, "X_test": None,
        "y_train": None, "y_val": None, "y_test": None,
        "scaler_X": None, "scaler_y": None,
        "nan_strategy": "drop",
        "normalization": "minmax",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,

        # Model builder
        "model": None,
        "model_config": None,
        "layers_config": [{"units": 64, "activation": "ReLU", "dropout": 0.0}],

        # Hyperparameter optimization
        "best_params": None,
        "optuna_study": None,

        # Training
        "trained": False,
        "train_losses": [],
        "val_losses": [],
        "best_model_state": None,
        "training_metrics": {},

        # Results
        "predictions": None,
    }

    for key, default in defaults.items():
        init_state(key, default)
