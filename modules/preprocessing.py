"""
Module 2 — Preprocessing
Normalization, train/val/test split, pair plot visualization.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

from utils.theme import neon_header, terminal_block, status_badge, COLORS
from utils.state import get_state, set_state


def render():
    neon_header("PREPROCESSING", "🔧")

    df = get_state("raw_data")
    input_cols = get_state("input_columns")
    output_col = get_state("output_column")

    if df is None or not input_cols or output_col is None:
        terminal_block("[ BLOCKED ] Load data and select columns first.\n\n  ← Go to 'Data Loading'")
        return

    all_cols = input_cols + [output_col]
    data = df[all_cols].copy().dropna()

    # ── Compact: Normalization + Split side by side ───────────
    left_col, right_col = st.columns(2)

    with left_col:
        neon_header("NORMALIZATION", "📏")
        norm_method = st.selectbox(
            "Method",
            ["minmax", "standard", "none"],
            index=["minmax", "standard", "none"].index(get_state("normalization", "minmax")),
            format_func=lambda x: {
                "minmax": "Min-Max Scaling [0, 1]",
                "standard": "Standard Scaling (Z-score)",
                "none": "No Normalization"
            }[x]
        )
        set_state("normalization", norm_method)

    with right_col:
        neon_header("DATA SPLIT", "✂")
        c1, c2, c3 = st.columns(3)
        with c1:
            train_r = st.slider("Train %", 50, 90, int(get_state("train_ratio", 0.7) * 100), 5)
        with c2:
            val_r = st.slider("Val %", 5, 30, int(get_state("val_ratio", 0.15) * 100), 5)
        with c3:
            test_r = 100 - train_r - val_r
            st.markdown(f"""
            <div style="text-align:center; margin-top:1.6rem;">
                <span style="color:{COLORS['orange']}; font-size:1.3rem; font-weight:700">{test_r}%</span>
                <br><span style="color:{COLORS['text_dim']}; font-size:0.75rem">Test (auto)</span>
            </div>
            """, unsafe_allow_html=True)

    if test_r < 5:
        status_badge("⚠ Test split too small (< 5%)", "warning")
        return

    set_state("train_ratio", train_r / 100)
    set_state("val_ratio", val_r / 100)
    set_state("test_ratio", test_r / 100)

    # ── Apply Button ─────────────────────────────────────────
    st.markdown("---")

    if st.button("⚡  APPLY PREPROCESSING", use_container_width=True, type="primary"):
        X = data[input_cols].values.astype(np.float32)
        y = data[output_col].values.astype(np.float32).reshape(-1, 1)

        # Normalization
        scaler_X, scaler_y = None, None
        if norm_method == "minmax":
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)
        elif norm_method == "standard":
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)

        # Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_r / 100, random_state=42
        )
        relative_val = val_r / (train_r + val_r)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val, random_state=42
        )

        # Convert to tensors
        set_state("X_train", torch.tensor(X_train, dtype=torch.float32))
        set_state("X_val", torch.tensor(X_val, dtype=torch.float32))
        set_state("X_test", torch.tensor(X_test, dtype=torch.float32))
        set_state("y_train", torch.tensor(y_train, dtype=torch.float32))
        set_state("y_val", torch.tensor(y_val, dtype=torch.float32))
        set_state("y_test", torch.tensor(y_test, dtype=torch.float32))
        set_state("scaler_X", scaler_X)
        set_state("scaler_y", scaler_y)
        set_state("preprocessed", True)

        # Reset downstream
        set_state("model", None)
        set_state("trained", False)
        set_state("best_params", None)

        st.rerun()

    # ── Show status ──────────────────────────────────────────
    if get_state("preprocessed"):
        X_tr = get_state("X_train")
        X_v = get_state("X_val")
        X_te = get_state("X_test")

        summary = f"""┌────────────────────────────────────────┐
│  PREPROCESSING COMPLETE                │
├────────────────────────────────────────┤
│  Normalization: {norm_method:<22}│
│  Train samples: {X_tr.shape[0]:<22}│
│  Val samples  : {X_v.shape[0]:<22}│
│  Test samples : {X_te.shape[0]:<22}│
│  Input dim    : {X_tr.shape[1]:<22}│
│  Rows (clean) : {data.shape[0]:<22}│
└────────────────────────────────────────┘"""
        terminal_block(summary)

        # Clickable proceed button
        if st.button("✓ READY — Proceed to Model Builder →", use_container_width=True, type="primary"):
            st.session_state["nav_target"] = "🏗  Model Builder"
            st.rerun()

        # ── Pair Plot ────────────────────────────────────────
        st.markdown("---")
        neon_header("PAIR PLOT", "📊")

        # Build a dataframe from training data for the pair plot
        X_tr_np = X_tr.numpy()
        y_tr_np = get_state("y_train").numpy().flatten()

        plot_df = pd.DataFrame(X_tr_np, columns=input_cols)
        plot_df[output_col] = y_tr_np

        # Limit to max 6 cols for readability
        plot_cols = all_cols[:6]
        plot_subset = plot_df[plot_cols] if len(plot_cols) <= len(plot_df.columns) else plot_df

        fig = px.scatter_matrix(
            plot_subset,
            dimensions=plot_cols,
            color=output_col if output_col in plot_cols else None,
            color_continuous_scale=["#00e5ff", "#ff00ff"],
            opacity=0.5,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg_card'],
            height=max(500, len(plot_cols) * 120),
            font=dict(family="JetBrains Mono, monospace", size=9, color=COLORS['text']),
            margin=dict(l=40, r=20, t=30, b=20),
        )
        fig.update_traces(
            diagonal_visible=True,
            marker=dict(size=3),
        )
        st.plotly_chart(fig, use_container_width=True)
