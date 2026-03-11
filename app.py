"""
Surrogate Builder
Main Streamlit app entry point.
"""
import streamlit as st

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="⚡ Surrogate Builder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.theme import inject_custom_css, COLORS
from utils.state import init_all_defaults
from modules import data_loading, preprocessing, model_builder, hyperopt, training, results

# ── Initialize ───────────────────────────────────────────────
init_all_defaults()
inject_custom_css()

# Navigation page list
PAGE_LIST = [
    "📂  Data Loading",
    "🔧  Preprocessing",
    "🏗  Model Builder",
    "🔍  Hyperopt",
    "🚀  Training",
    "📊  Results",
]
PAGE_MAP = {
    "📂  Data Loading": "data_loading",
    "🔧  Preprocessing": "preprocessing",
    "🏗  Model Builder": "model_builder",
    "🔍  Hyperopt": "hyperopt",
    "🚀  Training": "training",
    "📊  Results": "results",
}

# ── Sidebar Navigation ──────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:1rem 0;">
        <span style="color:{COLORS['green']}; font-size:1.5rem; font-weight:700;">⚡</span>
        <br>
        <span style="color:{COLORS['cyan']}; font-size:0.75rem; letter-spacing:3px;">
            SURROGATE<br>BUILDER
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "MODULES",
        PAGE_LIST,
        label_visibility="collapsed",
        key="nav_page"
    )

    st.markdown("---")

    # Pipeline status
    st.markdown(f"<span style='color:{COLORS['text_dim']}; font-size:0.7rem;'>PIPELINE STATUS</span>",
                unsafe_allow_html=True)

    steps = [
        ("Data Loaded", st.session_state.get("raw_data") is not None),
        ("Preprocessed", st.session_state.get("preprocessed", False)),
        ("Model Built", st.session_state.get("model") is not None),
        ("Trained", st.session_state.get("trained", False)),
    ]

    for label, done in steps:
        icon = f"<span style='color:{COLORS['green']}'>✓</span>" if done else f"<span style='color:{COLORS['text_dim']}'>○</span>"
        st.markdown(f"{icon} <span style='color:{COLORS['text']}; font-size:0.8rem;'>{label}</span>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="color:{COLORS['text_dim']}; font-size:0.65rem;">
            v1.0 · TensorFlow · Optuna<br>
            Built with ⚡ by Antigravity
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ─────────────────────────────────────────────

selected = PAGE_MAP[page]

if selected == "data_loading":
    data_loading.render()
elif selected == "preprocessing":
    preprocessing.render()
elif selected == "model_builder":
    model_builder.render()
elif selected == "hyperopt":
    hyperopt.render()
elif selected == "training":
    training.render()
elif selected == "results":
    results.render()
