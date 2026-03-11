"""
Module 3 — Model Builder
Configure NN architecture: layers, activations, dropout, loss, optimizer.
"""
import streamlit as st
import torch
import torch.nn as nn
from utils.theme import neon_header, terminal_block, status_badge, COLORS
from utils.state import get_state, set_state


# ── Activation Map ───────────────────────────────────────────
ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "SELU": nn.SELU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "GELU": nn.GELU,
    "SiLU (Swish)": nn.SiLU,
}


class SurrogateNet(nn.Module):
    """Dynamically-built feedforward neural network."""

    def __init__(self, input_dim: int, output_dim: int, layers_config: list):
        super().__init__()
        modules = []
        in_features = input_dim

        for layer in layers_config:
            modules.append(nn.Linear(in_features, layer["units"]))
            act_cls = ACTIVATIONS.get(layer["activation"], nn.ReLU)
            modules.append(act_cls())
            if layer["dropout"] > 0:
                modules.append(nn.Dropout(layer["dropout"]))
            in_features = layer["units"]

        modules.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


def render():
    neon_header("MODEL BUILDER", "🏗")

    if not get_state("preprocessed"):
        terminal_block("[ BLOCKED ] Preprocess data first.\n\n  ← Go to 'Preprocessing'")
        return

    X_train = get_state("X_train")
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
                act_options = list(ACTIVATIONS.keys())
                default_idx = act_options.index(layer["activation"]) if layer["activation"] in act_options else 0
                layer["activation"] = st.selectbox(
                    "Activation", act_options, default_idx,
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
            "MSELoss", "L1Loss (MAE)", "HuberLoss", "SmoothL1Loss"
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
        model = SurrogateNet(input_dim, 1, layers_config)

        model_config = {
            "layers": layers_config,
            "loss": loss_name,
            "optimizer": optim_name,
            "lr": lr,
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "input_dim": input_dim,
        }

        set_state("model", model)
        set_state("model_config", model_config)
        set_state("trained", False)
        st.rerun()

    # ── Architecture Summary ─────────────────────────────────
    model = get_state("model")
    config = get_state("model_config")

    if model is not None and config is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
│  Trainable params: {trainable:<21}│
└──────────────────────────────────────────┘"""

        terminal_block(arch_text)
        status_badge("✓ Model built — Proceed to Training or Hyperopt →", "ready")
