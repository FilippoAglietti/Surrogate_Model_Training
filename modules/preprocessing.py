"""
Module 2 — Preprocessing
NaN handling, normalization, train/val/test split, distribution visualization.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    data = df[all_cols].copy()

    # ── NaN Handling ─────────────────────────────────────────
    neon_header("NaN HANDLING", "🧹")
    nan_count = int(data.isnull().sum().sum())
    st.metric("Total Missing Values", nan_count)

    nan_strategy = st.selectbox(
        "Strategy",
        ["drop", "fill_mean", "fill_median", "fill_zero"],
        index=["drop", "fill_mean", "fill_median", "fill_zero"].index(get_state("nan_strategy", "drop")),
        format_func=lambda x: {
            "drop": "Drop rows with NaN",
            "fill_mean": "Fill with column mean",
            "fill_median": "Fill with column median",
            "fill_zero": "Fill with zero"
        }[x]
    )
    set_state("nan_strategy", nan_strategy)

    # Apply NaN strategy
    if nan_strategy == "drop":
        data = data.dropna()
    elif nan_strategy == "fill_mean":
        data = data.fillna(data.mean())
    elif nan_strategy == "fill_median":
        data = data.fillna(data.median())
    elif nan_strategy == "fill_zero":
        data = data.fillna(0)

    remaining_nan = int(data.isnull().sum().sum())
    if remaining_nan == 0:
        status_badge(f"✓ Clean — {data.shape[0]} rows remaining", "ready")
    else:
        status_badge(f"⚠ {remaining_nan} NaN still present", "warning")

    # ── Normalization ────────────────────────────────────────
    st.markdown("---")
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

    # ── Train / Val / Test Split ─────────────────────────────
    st.markdown("---")
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
            <span style="color:{COLORS['orange']}; font-size:1.5rem; font-weight:700">{test_r}%</span>
            <br><span style="color:{COLORS['text_dim']}; font-size:0.8rem">Test (auto)</span>
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
│  NaN strategy : {nan_strategy:<22}│
│  Normalization: {norm_method:<22}│
│  Train samples: {X_tr.shape[0]:<22}│
│  Val samples  : {X_v.shape[0]:<22}│
│  Test samples : {X_te.shape[0]:<22}│
│  Input dim    : {X_tr.shape[1]:<22}│
└────────────────────────────────────────┘"""
        terminal_block(summary)
        status_badge("✓ READY — Proceed to Model Builder →", "ready")

        # ── Distribution Plots ───────────────────────────────
        st.markdown("---")
        neon_header("FEATURE DISTRIBUTIONS", "📊")

        X_tr_np = X_tr.numpy()
        fig = make_subplots(
            rows=1, cols=min(len(input_cols), 4),
            subplot_titles=input_cols[:4]
        )
        for i, col_name in enumerate(input_cols[:4]):
            fig.add_trace(
                go.Histogram(
                    x=X_tr_np[:, i],
                    marker_color=COLORS['cyan'],
                    opacity=0.7,
                    name=col_name,
                    showlegend=False
                ),
                row=1, col=i + 1
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg_card'],
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="JetBrains Mono, monospace", color=COLORS['text'])
        )
        st.plotly_chart(fig, use_container_width=True)
