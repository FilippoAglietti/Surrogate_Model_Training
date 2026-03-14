"""
Centralized state management for the CustomTkinter application.
"""

class AppState:
    _state = {}

    @classmethod
    def get(cls, key, default=None):
        return cls._state.get(key, default)

    @classmethod
    def set(cls, key, value):
        cls._state[key] = value

    @classmethod
    def init_defaults(cls):
        defaults = {
            "nav_page": "📁  Data Loading",
            "data_loaded": False,
            "preprocessed": False,
            "model_ready": False,
            "trained": False,
            "df": None,
            "input_columns": [],
            "output_column": None,
            "X_train": None, "X_val": None, "X_test": None,
            "y_train": None, "y_val": None, "y_test": None,
            "scaler_X": None,
            "scaler_y": None,
            "pca_X": None,
            "pca_y": None,
            "applied_hpo_params": None,
            "layers_config": [{"units": 64, "activation": "ReLU", "dropout": 0.0}],
            "model_config": None,
            "model_params_count": 0,
            "model": None,           # Legacy Keras model (kept for compat)
            "surrogate_model": None, # Active SurrogateModel instance
            "selected_algo": None,   # Set by Hyperopt "Apply Best" to signal Model Builder
            "results_stale": False,  # Set True after each training to force Results rebuild
            "best_params": None,
            "optuna_study": None,
            "train_losses": [],
            "val_losses": [],
            "training_metrics": {}
        }
        for k, v in defaults.items():
            if k not in cls._state:
                cls._state[k] = v

get_state = AppState.get
set_state = AppState.set
init_all_defaults = AppState.init_defaults
