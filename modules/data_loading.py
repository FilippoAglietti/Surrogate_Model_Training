"""
Module 1 — Data Loading
Upload Excel/CSV files, select input/output columns, preview data.
"""
import streamlit as st
import pandas as pd
from utils.theme import neon_header, neon_card, terminal_block, status_badge, COLORS
from utils.state import set_state, get_state


def render():
    neon_header("DATA LOADING", "📂")

    # ── File Upload ──────────────────────────────────────────
    st.markdown(f"""
    <div class="neon-card">
        <span style="color:{COLORS['cyan']}; font-weight:600;">
            ▸ Upload your dataset (.xlsx, .xls, .csv)
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop file here",
        type=["xlsx", "xls", "csv"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            set_state("raw_data", df)
            status_badge(f"✓ Loaded: {uploaded.name}  |  {df.shape[0]} rows × {df.shape[1]} cols", "ready")
        except Exception as e:
            status_badge(f"✗ Error: {e}", "error")
            return

    df = get_state("raw_data")
    if df is None:
        terminal_block("[ WAITING ] No dataset loaded.\n\n  Supported formats:\n   • .xlsx / .xls (Excel)\n   • .csv (Comma Separated)\n\n  Drag & drop or click 'Browse files' above.")
        return

    # ── Data Preview ─────────────────────────────────────────
    neon_header("DATA PREVIEW", "👁")
    st.dataframe(df.head(20), use_container_width=True, height=350)

    # ── Column Stats ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric", df.select_dtypes(include="number").shape[1])
    col4.metric("Missing", int(df.isnull().sum().sum()))

    # ── Column Selection ─────────────────────────────────────
    st.markdown("---")
    neon_header("COLUMN SELECTION", "⚙")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        status_badge("✗ Need at least 2 numeric columns", "error")
        return

    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"<span style='color:{COLORS['cyan']}'>Input Features (X):</span>", unsafe_allow_html=True)
        input_cols = st.multiselect(
            "Select input columns",
            options=numeric_cols,
            default=get_state("input_columns") or numeric_cols[:-1],
            label_visibility="collapsed"
        )

    with right:
        st.markdown(f"<span style='color:{COLORS['orange']}'>Target Output (y):</span>", unsafe_allow_html=True)
        remaining = [c for c in numeric_cols if c not in input_cols]
        if remaining:
            default_idx = 0
            current_out = get_state("output_column")
            if current_out and current_out in remaining:
                default_idx = remaining.index(current_out)
            output_col = st.selectbox(
                "Select output column",
                options=remaining,
                index=default_idx,
                label_visibility="collapsed"
            )
        else:
            output_col = None
            status_badge("⚠ No columns left for output", "warning")

    if input_cols and output_col:
        set_state("input_columns", input_cols)
        set_state("output_column", output_col)

        # Summary
        summary = f"""┌─────────────────────────────────────────┐
│  COLUMN MAPPING                         │
├─────────────────────────────────────────┤
│  Input features ({len(input_cols)}):
"""
        for c in input_cols:
            summary += f"│    ▸ {c}\n"
        summary += f"""│                                         │
│  Target output:                         │
│    ▸ {output_col}
└─────────────────────────────────────────┘"""
        terminal_block(summary)

        status_badge("✓ READY — Proceed to Preprocessing →", "ready")
    else:
        status_badge("⚠ Select input and output columns", "warning")
