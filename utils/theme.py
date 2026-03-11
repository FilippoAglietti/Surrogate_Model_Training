"""
Neon terminal theme engine for Streamlit.
Injects custom CSS and provides color constants + ASCII art.
"""
import streamlit as st

# ── Color Palette ──────────────────────────────────────────────────
COLORS = {
    "bg":          "#0a0a0a",
    "bg_card":     "#0f0f0f",
    "bg_surface":  "#141414",
    "border":      "#1a1a2e",
    "green":       "#00ff41",
    "cyan":        "#00e5ff",
    "orange":      "#ff6e40",
    "magenta":     "#ff00ff",
    "yellow":      "#ffd600",
    "red":         "#ff1744",
    "text":        "#c0c0c0",
    "text_dim":    "#666666",
    "white":       "#e0e0e0",
}

NEON_GRADIENT = "linear-gradient(135deg, #00ff41 0%, #00e5ff 50%, #ff00ff 100%)"

ASCII_BANNER = r"""
███████╗██╗   ██╗██████╗ ██████╗  ██████╗  ██████╗  █████╗ ████████╗███████╗
██╔════╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗╚══██╔══╝██╔════╝
███████╗██║   ██║██████╔╝██████╔╝██║   ██║██║  ███╗███████║   ██║   █████╗  
╚════██║██║   ██║██╔══██╗██╔══██╗██║   ██║██║   ██║██╔══██║   ██║   ██╔══╝  
███████║╚██████╔╝██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████╗
╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
        ⚡  M O D E L   T R A I N I N G   E N G I N E  ⚡
"""

ASCII_MINI = "[ ⚡ SURROGATE MODEL TRAINER ⚡ ]"


def inject_custom_css():
    """Inject the full neon terminal CSS into the Streamlit app."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

        /* ── Global Reset ─────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        .stApp {{
            background-color: {COLORS['bg']};
        }}

        /* ── Sidebar ──────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['bg_card']};
            border-right: 1px solid {COLORS['border']};
        }}

        section[data-testid="stSidebar"] .stRadio > label {{
            color: {COLORS['green']} !important;
        }}

        /* ── Cards / Containers ───────────────────────── */
        div[data-testid="stExpander"] {{
            background-color: {COLORS['bg_surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        }}

        .neon-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.05);
        }}

        .neon-card:hover {{
            border-color: {COLORS['cyan']};
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.15);
        }}

        /* ── Buttons ──────────────────────────────────── */
        .stButton > button {{
            background: transparent;
            color: {COLORS['cyan']};
            border: 1px solid {COLORS['cyan']};
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background: {COLORS['cyan']};
            color: {COLORS['bg']};
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
        }}

        /* ── Inputs ───────────────────────────────────── */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {{
            background-color: {COLORS['bg_surface']} !important;
            border: 1px solid {COLORS['border']} !important;
            color: {COLORS['white']} !important;
            font-family: 'JetBrains Mono', monospace !important;
        }}

        .stSlider > div > div > div {{
            color: {COLORS['cyan']};
        }}

        /* ── Metrics ──────────────────────────────────── */
        div[data-testid="stMetric"] {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 0.8rem;
        }}

        div[data-testid="stMetric"] label {{
            color: {COLORS['text_dim']} !important;
        }}

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: {COLORS['green']} !important;
        }}

        /* ── Tables / Dataframes ──────────────────────── */
        .stDataFrame {{
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        }}

        /* ── Progress Bar ─────────────────────────────── */
        .stProgress > div > div > div {{
            background-color: {COLORS['cyan']} !important;
        }}

        /* ── Tabs ─────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            background-color: {COLORS['bg_card']};
            border-radius: 4px;
            padding: 2px;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {COLORS['text_dim']};
            font-family: 'JetBrains Mono', monospace;
            border-radius: 4px;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['bg_surface']};
            color: {COLORS['cyan']} !important;
        }}

        /* ── ASCII Banner ─────────────────────────────── */
        .ascii-banner {{
            color: {COLORS['green']};
            font-size: 0.45rem;
            line-height: 1.1;
            text-align: center;
            white-space: pre;
            font-family: 'JetBrains Mono', monospace;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
            margin-bottom: 0.5rem;
        }}

        .ascii-subtitle {{
            color: {COLORS['cyan']};
            font-size: 0.85rem;
            text-align: center;
            letter-spacing: 4px;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 2rem;
            text-shadow: 0 0 8px rgba(0, 229, 255, 0.4);
        }}

        /* ── Section Headers ──────────────────────────── */
        .section-header {{
            color: {COLORS['cyan']};
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 600;
        }}

        /* ── Terminal Output ──────────────────────────── */
        .terminal-output {{
            background: #050505;
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: {COLORS['green']};
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }}

        /* ── Status Badge ─────────────────────────────── */
        .status-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}

        .status-ready {{
            background: rgba(0, 255, 65, 0.1);
            color: {COLORS['green']};
            border: 1px solid {COLORS['green']};
        }}

        .status-warning {{
            background: rgba(255, 110, 64, 0.1);
            color: {COLORS['orange']};
            border: 1px solid {COLORS['orange']};
        }}

        .status-error {{
            background: rgba(255, 23, 68, 0.1);
            color: {COLORS['red']};
            border: 1px solid {COLORS['red']};
        }}

        /* ── Scrollbar ────────────────────────────────── */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {COLORS['bg']};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {COLORS['border']};
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['cyan']};
        }}

        /* ── File Uploader ────────────────────────────── */
        section[data-testid="stFileUploader"] {{
            background: {COLORS['bg_card']};
            border: 1px dashed {COLORS['border']};
            border-radius: 6px;
            padding: 0.5rem;
        }}

        section[data-testid="stFileUploader"]:hover {{
            border-color: {COLORS['cyan']};
        }}

        /* ── Hide Streamlit branding ──────────────────── */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


def render_ascii_banner():
    """Render the main ASCII art banner."""
    st.markdown(f'<div class="ascii-banner">{ASCII_BANNER}</div>', unsafe_allow_html=True)
    st.markdown('<div class="ascii-subtitle">v1.0 — Neural Network Surrogate Modeling</div>', unsafe_allow_html=True)


def neon_header(text: str, icon: str = "▸"):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{icon} {text}</div>', unsafe_allow_html=True)


def neon_card(content: str):
    """Wrap content in a neon-styled card."""
    st.markdown(f'<div class="neon-card">{content}</div>', unsafe_allow_html=True)


def terminal_block(text: str):
    """Render text in a terminal-style block."""
    st.markdown(f'<div class="terminal-output">{text}</div>', unsafe_allow_html=True)


def status_badge(text: str, status: str = "ready"):
    """Render a colored status badge. status: ready, warning, error."""
    st.markdown(f'<span class="status-badge status-{status}">{text}</span>', unsafe_allow_html=True)
