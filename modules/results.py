"""
Module 6 — Results Visualization
Predicted vs Actual, residuals, feature importance, metrics, export.
"""
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io
import os

from utils.theme import neon_header, terminal_block, status_badge, COLORS
from utils.state import get_state


def render():
    neon_header("RESULTS & ANALYSIS", "📊")

    model = get_state("model")
    if model is None or not get_state("trained"):
        terminal_block("[ BLOCKED ] Train a model first.\n\n  ← Go to 'Training Dashboard'")
        return

    X_test = get_state("X_test")
    y_test = get_state("y_test")
    scaler_y = get_state("scaler_y")
    input_cols = get_state("input_columns")
    output_col = get_state("output_column")

    # ── Predictions ──────────────────────────────────────────
    y_pred = model.predict(X_test, verbose=0)
    y_true = y_test

    # Inverse transform if scaled
    if scaler_y is not None:
        y_pred_orig = scaler_y.inverse_transform(y_pred)
        y_true_orig = scaler_y.inverse_transform(y_true)
    else:
        y_pred_orig = y_pred
        y_true_orig = y_true

    y_p = y_pred_orig.flatten()
    y_t = y_true_orig.flatten()

    # ── Metrics ──────────────────────────────────────────────
    neon_header("TEST METRICS", "📈")

    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-10))) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MSE", f"{mse:.6f}")
    c2.metric("RMSE", f"{rmse:.6f}")
    c3.metric("MAE", f"{mae:.6f}")
    c4.metric("R²", f"{r2:.4f}")
    c5.metric("MAPE %", f"{mape:.2f}")

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Pred vs Actual", "📊 Residuals", "🔑 Feature Importance", "💾 Export"
    ])

    # ── Predicted vs Actual ──────────────────────────────────
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_t, y=y_p,
            mode="markers",
            marker=dict(color=COLORS['cyan'], size=5, opacity=0.6),
            name="Predictions"
        ))
        # Perfect line
        min_v = min(y_t.min(), y_p.min())
        max_v = max(y_t.max(), y_p.max())
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines",
            line=dict(color=COLORS['green'], width=2, dash="dash"),
            name="Perfect Prediction"
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg_card'],
            title=f"Predicted vs Actual — {output_col}",
            xaxis_title="Actual", yaxis_title="Predicted",
            height=500,
            font=dict(family="JetBrains Mono, monospace", color=COLORS['text'])
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Residuals ────────────────────────────────────────────
    with tab2:
        residuals = y_t - y_p

        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_p, y=residuals,
            mode="markers",
            marker=dict(color=COLORS['orange'], size=5, opacity=0.6),
            name="Residuals"
        ))
        fig_res.add_hline(y=0, line=dict(color=COLORS['green'], width=1, dash="dash"))
        fig_res.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg_card'],
            title="Residuals vs Predicted",
            xaxis_title="Predicted", yaxis_title="Residual",
            height=400,
            font=dict(family="JetBrains Mono, monospace", color=COLORS['text'])
        )
        st.plotly_chart(fig_res, use_container_width=True)

        # Residual histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            marker_color=COLORS['magenta'],
            opacity=0.7,
            nbinsx=40
        ))
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg_card'],
            title="Residual Distribution",
            xaxis_title="Residual", yaxis_title="Count",
            height=350,
            font=dict(family="JetBrains Mono, monospace", color=COLORS['text'])
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Feature Importance (Permutation) ─────────────────────
    with tab3:
        neon_header("PERMUTATION IMPORTANCE", "🔑")

        if st.button("⚡  Compute Feature Importance", use_container_width=True):
            X_test_np = X_test.copy()
            baseline_mse = mean_squared_error(y_t, y_p)
            importances = []

            for i, col_name in enumerate(input_cols):
                X_perm = X_test_np.copy()
                np.random.shuffle(X_perm[:, i])
                
                y_perm = model.predict(X_perm, verbose=0)
                
                if scaler_y is not None:
                    y_perm = scaler_y.inverse_transform(y_perm).flatten()
                else:
                    y_perm = y_perm.flatten()
                    
                perm_mse = mean_squared_error(y_t, y_perm)
                importances.append(perm_mse - baseline_mse)

            imp_df = pd.DataFrame({
                "Feature": input_cols,
                "Importance": importances
            }).sort_values("Importance", ascending=True)

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker=dict(
                    color=imp_df["Importance"],
                    colorscale=[[0, COLORS['cyan']], [1, COLORS['magenta']]],
                )
            ))
            fig_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS['bg'],
                plot_bgcolor=COLORS['bg_card'],
                title="Permutation Feature Importance (ΔMSE)",
                xaxis_title="Importance (ΔMSE)",
                height=max(300, len(input_cols) * 35),
                font=dict(family="JetBrains Mono, monospace", color=COLORS['text'])
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── Export ────────────────────────────────────────────────
    with tab4:
        neon_header("EXPORT RESULTS", "💾")

        # Results DataFrame
        results_df = pd.DataFrame({
            "Actual": y_t,
            "Predicted": y_p,
            "Residual": y_t - y_p,
            "Abs Error": np.abs(y_t - y_p)
        })

        st.dataframe(results_df.head(20), use_container_width=True)

        col_dl1, col_dl2, col_dl3 = st.columns(3)

        # CSV download
        with col_dl1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv,
                "results.csv",
                "text/csv",
                use_container_width=True
            )

        # Excel download
        with col_dl2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Results')

                # Add metrics sheet
                metrics_df = pd.DataFrame({
                    "Metric": ["MSE", "RMSE", "MAE", "R²", "MAPE (%)"],
                    "Value": [mse, rmse, mae, r2, mape]
                })
                metrics_df.to_excel(writer, index=False, sheet_name='Metrics')

            st.download_button(
                "📊 Download Excel",
                buffer.getvalue(),
                "results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Model download
        with col_dl3:
            # Keras requires saving to disk to get bytes easily
            temp_path = "temp_model.h5"
            model.save(temp_path)
            
            with open(temp_path, "rb") as f:
                model_bytes = f.read()
                
            os.remove(temp_path)

            st.download_button(
                "🧠 Download Model (.h5)",
                model_bytes,
                "surrogate_model.h5",
                "application/octet-stream",
                use_container_width=True
            )

        # Metrics summary
        metrics_text = f"""┌──────────────────────────────────────────┐
│  TEST SET METRICS SUMMARY                │
├──────────────────────────────────────────┤
│  MSE    : {mse:<30.6f}│
│  RMSE   : {rmse:<30.6f}│
│  MAE    : {mae:<30.6f}│
│  R²     : {r2:<30.4f}│
│  MAPE % : {mape:<30.2f}│
│  Samples: {len(y_t):<30}│
└──────────────────────────────────────────┘"""
        terminal_block(metrics_text)
