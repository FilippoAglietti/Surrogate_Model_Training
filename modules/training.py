"""
Module 5 — Training Dashboard
Training loop with real-time loss curves, metrics, early stopping.
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
import plotly.graph_objects as go
import time

from modules.model_builder import build_surrogate_model
from modules.hyperopt import get_keras_loss, get_keras_optimizer
from utils.theme import neon_header, terminal_block, status_badge, COLORS
from utils.state import get_state, set_state


class StreamlitUpdateCallback(Callback):
    """Custom Keras callback to update Streamlit UI during training."""
    def __init__(self, epochs, X_val, y_val, ui_elements):
        super().__init__()
        self.epochs = epochs
        self.X_val = X_val
        self.y_val = y_val
        self.epoch_metric = ui_elements['epoch']
        self.tloss_metric = ui_elements['tloss']
        self.vloss_metric = ui_elements['vloss']
        self.status_metric = ui_elements['status']
        self.progress = ui_elements['progress']
        self.chart_placeholder = ui_elements['chart']
        self.log_placeholder = ui_elements['log']
        
        self.train_losses = []
        self.val_losses = []
        self.log_lines = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', 0.0)
        train_loss = logs.get('loss', 0.0)
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Compute R² manually for Streamlit update
        val_pred = self.model.predict(self.X_val, verbose=0).flatten()
        y_val_flat = self.y_val.flatten()
        ss_res = np.sum((y_val_flat - val_pred) ** 2)
        ss_tot = np.sum((y_val_flat - np.mean(y_val_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        pct = (epoch + 1) / self.epochs
        self.progress.progress(pct)
        self.epoch_metric.metric("Epoch", f"{epoch + 1}/{self.epochs}")
        self.tloss_metric.metric("Train Loss", f"{train_loss:.6f}")
        self.vloss_metric.metric("Val Loss", f"{val_loss:.6f}")
        self.status_metric.metric("R²", f"{r2:.4f}")

        line = f"[{epoch+1:>5}/{self.epochs}] train={train_loss:.6f}  val={val_loss:.6f}  R²={r2:.4f}"
        self.log_lines.append(line)

        # Update chart every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=self.train_losses, mode="lines", name="Train Loss",
                line=dict(color=COLORS['cyan'], width=2)
            ))
            fig.add_trace(go.Scatter(
                y=self.val_losses, mode="lines", name="Val Loss",
                line=dict(color=COLORS['orange'], width=2)
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS['bg'],
                plot_bgcolor=COLORS['bg_card'],
                title="Loss Curves",
                xaxis_title="Epoch", yaxis_title="Loss",
                height=350,
                font=dict(family="JetBrains Mono, monospace", color=COLORS['text']),
                legend=dict(x=0.7, y=0.95)
            )
            self.chart_placeholder.plotly_chart(fig, use_container_width=True)

            visible_log = "\n".join(self.log_lines[-15:])
            self.log_placeholder.markdown(
                f'<div class="terminal-output">{visible_log}</div>',
                unsafe_allow_html=True
            )


def render():
    neon_header("TRAINING DASHBOARD", "🚀")

    config = get_state("model_config")

    if not get_state("model_ready") or config is None:
        terminal_block("[ BLOCKED ] Configure a model first.\n\n  ← Go to 'Model Builder'")
        return

    if not get_state("preprocessed"):
        terminal_block("[ BLOCKED ] Preprocess data first.\n\n  ← Go to 'Preprocessing'")
        return

    X_train = get_state("X_train")
    y_train = get_state("y_train")
    X_val = get_state("X_val")
    y_val = get_state("y_val")

    # ── Early Stopping Config ────────────────────────────────
    neon_header("EARLY STOPPING", "🛑")
    es_col1, es_col2, es_col3 = st.columns(3)
    with es_col1:
        use_es = st.checkbox("Enable Early Stopping", True)
    with es_col2:
        patience = st.number_input("Patience", 5, 200, 20, 5)
    with es_col3:
        min_delta = st.number_input("Min Delta", 0.0, 0.01, 1e-5, format="%.6f")

    # ── Training Controls ────────────────────────────────────
    st.markdown("---")

    if st.button("⚡  START TRAINING", use_container_width=True, type="primary"):
        # Build completely fresh model
        model = build_surrogate_model(config["input_dim"], 1, config["layers"])
        
        criterion = get_keras_loss(config["loss"])
        optimizer = get_keras_optimizer(config["optimizer"], config["lr"])
        
        model.compile(optimizer=optimizer, loss=criterion)

        epochs = config["epochs"]

        # UI placeholders
        progress = st.progress(0)
        metrics_row = st.columns(4)
        epoch_metric = metrics_row[0].empty()
        tloss_metric = metrics_row[1].empty()
        vloss_metric = metrics_row[2].empty()
        status_metric = metrics_row[3].empty()

        chart_placeholder = st.empty()
        log_placeholder = st.empty()

        st_callback = StreamlitUpdateCallback(epochs, X_val, y_val, {
            'epoch': epoch_metric,
            'tloss': tloss_metric,
            'vloss': vloss_metric,
            'status': status_metric,
            'progress': progress,
            'chart': chart_placeholder,
            'log': log_placeholder
        })

        callbacks = [st_callback]
        
        if use_es:
            es = EarlyStopping(
                monitor='val_loss',
                patience=int(patience),
                min_delta=float(min_delta),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(es)

        start_time = time.time()

        # Run Keras fit
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config["batch_size"],
            callbacks=callbacks,
            verbose=0
        )

        elapsed = time.time() - start_time
        
        train_losses = st_callback.train_losses
        val_losses = st_callback.val_losses
        best_val_loss = min(val_losses) if val_losses else float('inf')

        # Final R2 evaluation with best weights
        val_pred = model.predict(X_val, verbose=0).flatten()
        y_val_flat = y_val.flatten()
        ss_res = np.sum((y_val_flat - val_pred) ** 2)
        ss_tot = np.sum((y_val_flat - np.mean(y_val_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Save state
        set_state("model", model)  # Now we save the instantiated, trained model
        set_state("trained", True)
        set_state("train_losses", train_losses)
        set_state("val_losses", val_losses)
        set_state("training_metrics", {
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else 0,
            "r2": r2,
            "epochs_run": len(train_losses),
            "elapsed_seconds": elapsed
        })

        progress.progress(1.0)
        
        if use_es and len(train_losses) < epochs:
            st_callback.log_lines.append(f"\n⚠ Early stopping triggered at epoch {len(train_losses)}")
            visible_log = "\n".join(st_callback.log_lines[-15:])
            log_placeholder.markdown(
                f'<div class="terminal-output">{visible_log}</div>',
                unsafe_allow_html=True
            )

        summary = f"""
┌──────────────────────────────────────────┐
│  TRAINING COMPLETE                       │
├──────────────────────────────────────────┤
│  Epochs run   : {len(train_losses):<24}│
│  Best val loss: {best_val_loss:<24.6f}│
│  Final R²     : {r2:<24.4f}│
│  Time         : {elapsed:<24.1f}│
└──────────────────────────────────────────┘"""
        terminal_block(summary)
        status_badge("✓ Training complete — View Results →", "ready")

    # ── Show previous training results ───────────────────────
    elif get_state("trained"):
        metrics = get_state("training_metrics", {})
        train_l = get_state("train_losses", [])
        val_l = get_state("val_losses", [])

        if train_l:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Epochs", metrics.get("epochs_run", "—"))
            c2.metric("Best Val Loss", f"{metrics.get('best_val_loss', 0):.6f}")
            c3.metric("R²", f"{metrics.get('r2', 0):.4f}")
            c4.metric("Time (s)", f"{metrics.get('elapsed_seconds', 0):.1f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=train_l, mode="lines", name="Train Loss",
                line=dict(color=COLORS['cyan'], width=2)
            ))
            fig.add_trace(go.Scatter(
                y=val_l, mode="lines", name="Val Loss",
                line=dict(color=COLORS['orange'], width=2)
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS['bg'],
                plot_bgcolor=COLORS['bg_card'],
                title="Loss Curves (Previous Training)",
                xaxis_title="Epoch", yaxis_title="Loss",
                height=350,
                font=dict(family="JetBrains Mono, monospace", color=COLORS['text'])
            )
            st.plotly_chart(fig, use_container_width=True)

            status_badge("✓ Model trained — View Results →", "ready")
