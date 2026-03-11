"""
Module 3 — Model Builder
Configure NN architecture: layers, activations, dropout, loss, optimizer.
"""
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LeakyReLU, ELU

from utils.theme import neon_header, terminal_block, status_badge
from utils.state import get_state, set_state


# ── Activation Map ───────────────────────────────────────────
def get_keras_activation(name: str):
    if name == "LeakyReLU":
        return LeakyReLU()
    elif name == "ELU":
        return ELU()
    # Keras strings for the rest
    mapping = {
        "ReLU": "relu",
        "SELU": "selu",
        "Tanh": "tanh",
        "Sigmoid": "sigmoid",
        "GELU": "gelu",
        "SiLU (Swish)": "swish",
    }
    return mapping.get(name, "relu")

def build_surrogate_model(input_dim: int, output_dim: int, layers_config: list) -> Sequential:
    """Dynamically-built feedforward neural network."""
    model = Sequential()
    
    # First layer needs input_shape
    first_layer_added = False
    
    for layer in layers_config:
        if not first_layer_added:
            model.add(Dense(layer["units"], input_shape=(input_dim,)))
            first_layer_added = True
        else:
            model.add(Dense(layer["units"]))
            
        # Add activation
        act = get_keras_activation(layer["activation"])
        if isinstance(act, str):
            model.add(Activation(act))
        else:
            model.add(act)
            
        # Add dropout
        if layer["dropout"] > 0:
            model.add(Dropout(layer["dropout"]))
            
    # Output layer
    model.add(Dense(output_dim))
    return model


def render():
    neon_header("MODEL BUILDER", "🏗")

    if not get_state("preprocessed"):
        terminal_block("[ BLOCKED ] Preprocess data first.\n\n  ← Go to 'Preprocessing'")
        return

    X_train = get_state("X_train")
    if X_train is None:
        return
        
    input_dim = X_train.shape[1]

    # ── Layer Configuration ──────────────────────────────────
    neon_header("ARCHITECTURE", "🧱")

    layers_config = get_state("layers_config",
                              [{"units": 64, "activation": "ReLU", "dropout": 0.0}])

    # Add / remove layer buttons
    col_add, col_rem = st.columns(2)
    with col_add:
        if st.button("＋ Add Layer", use_container_width=True):
            layers_config.append({"units": 64, "activation": "ReLU", "dropout": 0.0})
            set_state("layers_config", layers_config)
            st.rerun()
    with col_rem:
        if st.button("－ Remove Layer", use_container_width=True) and len(layers_config) > 1:
            layers_config.pop()
            set_state("layers_config", layers_config)
            st.rerun()

    st.markdown("---")

    ACTIVATION_NAMES = ["ReLU", "LeakyReLU", "ELU", "SELU", "Tanh", "Sigmoid", "GELU", "SiLU (Swish)"]

    # Configure each layer
    for i, layer in enumerate(layers_config):
        with st.expander(f"Layer {i + 1}", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                layer["units"] = st.number_input(
                    "Neurons", 4, 1024, layer["units"], 8,
                    key=f"units_{i}"
                )
            with c2:
                default_idx = ACTIVATION_NAMES.index(layer["activation"]) if layer["activation"] in ACTIVATION_NAMES else 0
                layer["activation"] = st.selectbox(
                    "Activation", ACTIVATION_NAMES, default_idx,
                    key=f"act_{i}"
                )
            with c3:
                layer["dropout"] = st.slider(
                    "Dropout", 0.0, 0.8, layer["dropout"], 0.05,
                    key=f"drop_{i}"
                )

    set_state("layers_config", layers_config)

    # ── Loss & Optimizer ─────────────────────────────────────
    st.markdown("---")
    neon_header("LOSS & OPTIMIZER", "⚙")

    col_l, col_o = st.columns(2)
    with col_l:
        loss_name = st.selectbox("Loss Function", [
            "MeanSquaredError", "MeanAbsoluteError", "Huber", "LogCosh"
        ])
    with col_o:
        optim_name = st.selectbox("Optimizer", [
            "Adam", "AdamW", "SGD", "RMSprop"
        ])

    lr_col, bs_col, ep_col = st.columns(3)
    with lr_col:
        lr = st.number_input("Learning Rate", 1e-6, 1.0, 1e-3, format="%.6f")
    with bs_col:
        batch_size = st.number_input("Batch Size", 8, 2048, 64, 8)
    with ep_col:
        epochs = st.number_input("Epochs", 10, 10000, 200, 10)

    # ── Build Model ──────────────────────────────────────────
    st.markdown("---")

    if st.button("⚡  BUILD MODEL", use_container_width=True, type="primary"):
        # We don't save the actual Keras backend model here to avoid session state serialization issues with Keras 3/TF 2.15
        # Instead, we just save the config. We will build and compile it in Training/Hyperopt.
        
        model_config = {
            "layers": layers_config,
            "loss": loss_name,
            "optimizer": optim_name,
            "lr": lr,
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "input_dim": input_dim,
        }

        # Tell downstream we have a config ready to build
        set_state("model_ready", True)
        set_state("model_config", model_config)
        set_state("trained", False)
        
        # Build temporarily just to count params
        temp_model = build_surrogate_model(input_dim, 1, layers_config)
        set_state("model_params_count", temp_model.count_params())
        
        st.rerun()

    # ── Architecture Summary ─────────────────────────────────
    config = get_state("model_config")

    if get_state("model_ready") and config is not None:
        total_params = get_state("model_params_count", 0)

        arch_text = f"""┌──────────────────────────────────────────┐
│  NEURAL NETWORK ARCHITECTURE             │
├──────────────────────────────────────────┤
│  Input dim  : {config['input_dim']:<26}│
│  Output dim : 1{' ' * 25}│
│  Loss       : {config['loss']:<26}│
│  Optimizer  : {config['optimizer']:<26}│
│  LR         : {config['lr']:<26}│
│  Batch Size : {config['batch_size']:<26}│
│  Epochs     : {config['epochs']:<26}│
├──────────────────────────────────────────┤
│  LAYERS:                                 │"""

        for i, layer in enumerate(config['layers']):
            arch_text += f"\n│   [{i+1}] Dense({layer['units']}) → {layer['activation']}"
            if layer['dropout'] > 0:
                arch_text += f" → Dropout({layer['dropout']})"

        arch_text += f"""
├──────────────────────────────────────────┤
│  Total params    : {total_params:<21}│
└──────────────────────────────────────────┘"""

        terminal_block(arch_text)
        status_badge("✓ Model configured — Proceed to Training or Hyperopt →", "ready")
