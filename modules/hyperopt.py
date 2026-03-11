"""
Module 4 — Hyperparameter Optimization
Grid search, Random search, Optuna (TPE) with live progress.
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import itertools
import random as rand_module

from modules.model_builder import build_surrogate_model
from utils.theme import neon_header, terminal_block, status_badge, COLORS
from utils.state import get_state, set_state

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_keras_loss(name):
    losses = {
        "MeanSquaredError": tf.keras.losses.MeanSquaredError(),
        "MeanAbsoluteError": tf.keras.losses.MeanAbsoluteError(),
        "Huber": tf.keras.losses.Huber(),
        "LogCosh": tf.keras.losses.LogCosh(),
    }
    return losses.get(name, tf.keras.losses.MeanSquaredError())

def get_keras_optimizer(name, lr):
    optimizers = {
        "Adam": tf.keras.optimizers.Adam(learning_rate=lr),
        "AdamW": tf.keras.optimizers.AdamW(learning_rate=lr) if hasattr(tf.keras.optimizers, 'AdamW') else tf.keras.optimizers.Adam(learning_rate=lr),
        "SGD": tf.keras.optimizers.SGD(learning_rate=lr),
        "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=lr),
    }
    return optimizers.get(name, tf.keras.optimizers.Adam(learning_rate=lr))

def _train_eval(input_dim, layers_cfg, X_train, y_train, X_val, y_val, lr, batch_size, epochs, loss_name):
    """Quick train + eval for HPO trial. Returns best val loss."""
    model = build_surrogate_model(input_dim, 1, layers_cfg)
    
    criterion = get_keras_loss(loss_name)
    optimizer = get_keras_optimizer("Adam", lr) # Default to Adam for HPO if not specified
    
    model.compile(optimizer=optimizer, loss=criterion)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    return min(history.history['val_loss'])


def render():
    neon_header("HYPERPARAMETER OPTIMIZATION", "🔍")

    if not get_state("preprocessed"):
        terminal_block("[ BLOCKED ] Preprocess data first.\n\n  ← Go to 'Preprocessing'")
        return

    X_train = get_state("X_train")
    y_train = get_state("y_train")
    X_val = get_state("X_val")
    y_val = get_state("y_val")
    
    if X_train is None:
        return
        
    input_dim = X_train.shape[1]

    # ── Strategy Selection ───────────────────────────────────
    neon_header("STRATEGY", "🎯")

    strategy = st.selectbox("Optimization Method", [
        "Optuna (TPE)", "Random Search", "Grid Search"
    ])

    # ── Search Space ─────────────────────────────────────────
    neon_header("SEARCH SPACE", "📐")

    # Re-use activation list
    ACTIVATION_NAMES = ["ReLU", "LeakyReLU", "ELU", "SELU", "Tanh", "Sigmoid", "GELU", "SiLU (Swish)"]

    col1, col2 = st.columns(2)
    with col1:
        lr_min = st.number_input("LR min", 1e-6, 0.1, 1e-4, format="%.6f")
        layers_min = st.number_input("Min layers", 1, 10, 1)
        units_min = st.number_input("Min neurons/layer", 8, 512, 16, 8)
        dropout_min = st.slider("Dropout min", 0.0, 0.5, 0.0, 0.05)
    with col2:
        lr_max = st.number_input("LR max", 1e-5, 1.0, 1e-2, format="%.6f")
        layers_max = st.number_input("Max layers", 1, 10, 4)
        units_max = st.number_input("Max neurons/layer", 16, 1024, 256, 8)
        dropout_max = st.slider("Dropout max", 0.0, 0.8, 0.3, 0.05)

    batch_options = st.multiselect("Batch Sizes", [16, 32, 64, 128, 256, 512], default=[32, 64, 128])
    if not batch_options:
        batch_options = [64]

    loss_name = st.selectbox("Loss for HPO", ["MeanSquaredError", "MeanAbsoluteError", "Huber", "LogCosh"])
    hpo_epochs = st.number_input("Epochs per trial", 10, 500, 50, 10)
    n_trials = st.number_input("Number of trials", 3, 200, 20, 1)

    st.markdown("---")

    # ── Run Optimization ─────────────────────────────────────
    if st.button("⚡  RUN OPTIMIZATION", use_container_width=True, type="primary"):
        progress = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        all_results = []

        if strategy == "Optuna (TPE)":
            study = optuna.create_study(direction="minimize")

            def objective(trial):
                n_layers = trial.suggest_int("n_layers", int(layers_min), int(layers_max))
                layers_cfg = []
                for i in range(n_layers):
                    u = trial.suggest_int(f"units_{i}", int(units_min), int(units_max), step=8)
                    d = trial.suggest_float(f"dropout_{i}", dropout_min, dropout_max, step=0.05)
                    act = trial.suggest_categorical(f"act_{i}", ACTIVATION_NAMES)
                    layers_cfg.append({"units": u, "activation": act, "dropout": d})

                lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
                bs = trial.suggest_categorical("batch_size", batch_options)

                val_loss = _train_eval(input_dim, layers_cfg, X_train, y_train, X_val, y_val,
                                       lr, bs, int(hpo_epochs), loss_name)
                return val_loss

            for i in range(int(n_trials)):
                study.optimize(objective, n_trials=1, show_progress_bar=False)
                pct = (i + 1) / int(n_trials)
                progress.progress(pct)
                best = study.best_trial
                status_text.markdown(
                    f"<span style='color:{COLORS['cyan']}'>Trial {i+1}/{int(n_trials)} | "
                    f"Best loss: {best.value:.6f}</span>",
                    unsafe_allow_html=True
                )
                all_results.append({"trial": i + 1, "loss": study.trials[-1].value,
                                    "best_loss": best.value})

            best_p = study.best_params
            set_state("optuna_study", study)

        elif strategy == "Random Search":
            best_val_loss = float("inf")
            best_p = {}

            for i in range(int(n_trials)):
                n_layers = rand_module.randint(int(layers_min), int(layers_max))
                layers_cfg = []
                for _ in range(n_layers):
                    layers_cfg.append({
                        "units": rand_module.choice(range(int(units_min), int(units_max) + 1, 8)),
                        "activation": rand_module.choice(ACTIVATION_NAMES),
                        "dropout": round(rand_module.uniform(dropout_min, dropout_max), 2)
                    })
                lr = 10 ** rand_module.uniform(np.log10(lr_min), np.log10(lr_max))
                bs = rand_module.choice(batch_options)

                val_loss = _train_eval(input_dim, layers_cfg, X_train, y_train, X_val, y_val,
                                       lr, bs, int(hpo_epochs), loss_name)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_p = {"n_layers": n_layers, "layers": layers_cfg,
                              "lr": lr, "batch_size": bs}

                pct = (i + 1) / int(n_trials)
                progress.progress(pct)
                status_text.markdown(
                    f"<span style='color:{COLORS['cyan']}'>Trial {i+1}/{int(n_trials)} | "
                    f"Best loss: {best_val_loss:.6f}</span>",
                    unsafe_allow_html=True
                )
                all_results.append({"trial": i + 1, "loss": val_loss,
                                    "best_loss": best_val_loss})

        else:  # Grid Search
            layer_counts = list(range(int(layers_min), int(layers_max) + 1))
            unit_options = list(range(int(units_min), int(units_max) + 1, 32))
            lr_options = np.logspace(np.log10(lr_min), np.log10(lr_max), 4).tolist()

            grid = list(itertools.product(layer_counts, unit_options[:4], lr_options, batch_options))
            if len(grid) > int(n_trials):
                grid = rand_module.sample(grid, int(n_trials))

            best_val_loss = float("inf")
            best_p = {}

            for i, (nl, nu, lr_v, bs) in enumerate(grid):
                layers_cfg = [{"units": nu, "activation": "ReLU", "dropout": 0.1}] * nl
                val_loss = _train_eval(input_dim, layers_cfg, X_train, y_train, X_val, y_val,
                                       lr_v, bs, int(hpo_epochs), loss_name)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_p = {"n_layers": nl, "units": nu, "lr": lr_v, "batch_size": bs}

                pct = (i + 1) / len(grid)
                progress.progress(pct)
                status_text.markdown(
                    f"<span style='color:{COLORS['cyan']}'>Trial {i+1}/{len(grid)} | "
                    f"Best loss: {best_val_loss:.6f}</span>",
                    unsafe_allow_html=True
                )
                all_results.append({"trial": i + 1, "loss": val_loss,
                                    "best_loss": best_val_loss})

        set_state("best_params", best_p)
        progress.progress(1.0)
        status_text.markdown(
            f"<span style='color:{COLORS['green']}'>✓ Optimization complete!</span>",
            unsafe_allow_html=True
        )

        # Show results
        with results_container:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r["trial"] for r in all_results],
                y=[r["loss"] for r in all_results],
                mode="markers", name="Trial Loss",
                marker=dict(color=COLORS['cyan'], size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[r["trial"] for r in all_results],
                y=[r["best_loss"] for r in all_results],
                mode="lines", name="Best So Far",
                line=dict(color=COLORS['green'], width=2)
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS['bg'],
                plot_bgcolor=COLORS['bg_card'],
                title="Optimization Progress",
                xaxis_title="Trial", yaxis_title="Validation Loss",
                height=350,
                font=dict(family="JetBrains Mono, monospace", color=COLORS['text'])
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Show Best Params ─────────────────────────────────────
    bp = get_state("best_params")
    if bp:
        lines = "┌──────────────────────────────────────────┐\n"
        lines += "│  BEST HYPERPARAMETERS                    │\n"
        lines += "├──────────────────────────────────────────┤\n"
        for k, v in bp.items():
            if isinstance(v, float):
                val = f"{v:.6f}"
            else:
                val = str(v)
            lines += f"│  {k:<16}: {val:<22}│\n"
        lines += "└──────────────────────────────────────────┘"
        terminal_block(lines)
        status_badge("✓ Best params found — Apply in Model Builder or Train directly →", "ready")
