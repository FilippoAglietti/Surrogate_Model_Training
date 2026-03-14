from models.nn_model import NeuralNetworkSurrogate
from models.xgb_model import XGBoostSurrogate
from models.rf_model import RandomForestSurrogate
from models.gpr_model import GPRSurrogate

ALGORITHM_REGISTRY = {
    "Neural Network": NeuralNetworkSurrogate,
    "XGBoost": XGBoostSurrogate,
    "Random Forest": RandomForestSurrogate,
    "Gaussian Process": GPRSurrogate,
}

ALGO_NAMES = list(ALGORITHM_REGISTRY.keys())
