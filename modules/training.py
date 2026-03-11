"""
Module 5 — Training Dashboard
Training loop with real-time loss curves, metrics, early stopping.
"""
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
import time
import copy

from utils.theme import neon_header, terminal_block, status_badge, COLORS
from utils.state import get_state, set_state


def render():
    neon_header("TRAINING DASHBOARD", "🚀")

    model = get_state("model")
    config = get_state("model_config")

    if model is None or config is None:
        terminal_block("[ BLOCKED ] Build a model first.\n\n  ← Go to 'Model Builder'")
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
        # Loss & optimizer setup
        loss_map = {
            "MSELoss": nn.MSELoss,
            "L1Loss (MAE)": nn.L1Loss,
            "HuberLoss": nn.HuberLoss,
            "SmoothL1Loss": nn.SmoothL1Loss,
        }
        criterion = loss_map.get(config["loss"], nn.MSELoss)()

        optim_map = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
        }
        optimizer = optim_map.get(config["optimizer"], torch.optim.Adam)(
            model.parameters(), lr=config["lr"]
        )

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        epochs = config["epochs"]
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        # UI placeholders
        progress = st.progress(0)
        metrics_row = st.columns(4)
        epoch_metric = metrics_row[0].empty()
        tloss_metric = metrics_row[1].empty()
        vloss_metric = metrics_row[2].empty()
        status_metric = metrics_row[3].empty()

        chart_placeholder = st.empty()
        log_placeholder = st.empty()

        log_lines = []
        start_time = time.time()

        for epoch in range(epochs):
            # ── Train ────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / n_batches

            # ── Validate ─────────────────────────────────
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

                # R² score
                ss_res = ((y_val - val_pred) ** 2).sum().item()
                ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            train_losses.append(avg_train)
            val_losses.append(val_loss)

            # ── Early Stopping ───────────────────────────
            improved = False
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1

            # ── Update UI ────────────────────────────────
            pct = (epoch + 1) / epochs
            progress.progress(pct)
            epoch_metric.metric("Epoch", f"{epoch + 1}/{epochs}")
            tloss_metric.metric("Train Loss", f"{avg_train:.6f}")
            vloss_metric.metric("Val Loss", f"{val_loss:.6f}")
            status_metric.metric("R²", f"{r2:.4f}")

            line = f"[{epoch+1:>5}/{epochs}] train={avg_train:.6f}  val={val_loss:.6f}  R²={r2:.4f}"
            if improved:
                line += "  ★ best"
            log_lines.append(line)

            # Update chart every 5 epochs or last epoch
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1 or (use_es and patience_counter >= patience):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=train_losses, mode="lines", name="Train Loss",
                    line=dict(color=COLORS['cyan'], width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=val_losses, mode="lines", name="Val Loss",
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
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Show last 15 log lines
                visible_log = "\n".join(log_lines[-15:])
                log_placeholder.markdown(
                    f'<div class="terminal-output">{visible_log}</div>',
                    unsafe_allow_html=True
                )

            # Check early stopping
            if use_es and patience_counter >= patience:
                log_lines.append(f"\n⚠ Early stopping at epoch {epoch+1} (patience={patience})")
                break

        elapsed = time.time() - start_time

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save state
        set_state("model", model)
        set_state("trained", True)
        set_state("train_losses", train_losses)
        set_state("val_losses", val_losses)
        set_state("best_model_state", best_state)
        set_state("training_metrics", {
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1],
            "r2": r2,
            "epochs_run": len(train_losses),
            "elapsed_seconds": elapsed
        })

        progress.progress(1.0)

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
