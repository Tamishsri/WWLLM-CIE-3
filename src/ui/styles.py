"""
styles.py — Dark-Themed Industrial Minimalist CSS for Streamlit
===============================================================
Injected via ``st.markdown(unsafe_allow_html=True)`` at app startup.

Design tokens:
    • Background: #0E1117 (Streamlit native dark) / #161B22 (cards)
    • Accent:     #58A6FF (blue) / #3FB950 (green for success)
    • Text:       #C9D1D9 (body) / #F0F6FC (headings)
    • Font stack: 'Inter', 'SF Pro Display', system-ui, sans-serif
"""

from __future__ import annotations

DARK_THEME_CSS: str = """
<style>
/* ── Import Inter font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global resets ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'SF Pro Display', system-ui, -apple-system, sans-serif;
    color: #C9D1D9;
}

/* ── Main container ────────────────────────────────────────────────── */
.stApp {
    background-color: #0E1117;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #F0F6FC;
}

/* ── Headings ──────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: #F0F6FC !important;
    font-weight: 600;
    letter-spacing: -0.02em;
}
h1 {
    font-size: 1.9rem !important;
    border-bottom: 2px solid #58A6FF;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}

/* ── Cards / containers ────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background-color: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
}
div.stTabs [data-baseweb="tab-panel"] {
    background-color: #161B22;
    border-radius: 0 0 8px 8px;
    padding: 1rem;
}

/* ── Buttons ───────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2EA043 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2EA043 0%, #3FB950 100%);
    box-shadow: 0 0 12px rgba(63, 185, 80, 0.3);
    transform: translateY(-1px);
}

/* ── File uploader ─────────────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background-color: #161B22;
    border: 2px dashed #30363D;
    border-radius: 8px;
    padding: 1rem;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #58A6FF;
}

/* ── Select boxes / inputs ─────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background-color: #0D1117;
    border: 1px solid #30363D;
    color: #C9D1D9;
    border-radius: 6px;
}

/* ── Metrics ───────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background-color: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
div[data-testid="stMetric"] label {
    color: #8B949E;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #58A6FF;
    font-weight: 700;
}

/* ── Progress bar ──────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #58A6FF, #3FB950);
    border-radius: 4px;
}

/* ── Code blocks ───────────────────────────────────────────────────── */
pre, code {
    background-color: #161B22 !important;
    border: 1px solid #21262D;
    border-radius: 6px;
}

/* ── Status / alerts ───────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 6px;
    border-left-width: 4px;
}

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0E1117;
}
::-webkit-scrollbar-thumb {
    background: #30363D;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #484F58;
}

/* ── Quiz card styling (used with st.markdown) ─────────────────────── */
.quiz-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.quiz-card h4 {
    color: #58A6FF !important;
    margin-bottom: 0.6rem;
}
.quiz-option {
    padding: 0.4rem 0.8rem;
    margin: 0.3rem 0;
    border-radius: 6px;
    background: #0D1117;
    border: 1px solid #21262D;
    font-size: 0.92rem;
}
.quiz-option.correct {
    border-color: #3FB950;
    background: rgba(63, 185, 80, 0.08);
}
.quiz-explanation {
    margin-top: 0.6rem;
    padding: 0.6rem;
    background: rgba(88, 166, 255, 0.06);
    border-left: 3px solid #58A6FF;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
    color: #8B949E;
}

/* ── Footer ────────────────────────────────────────────────────────── */
.footer-text {
    text-align: center;
    color: #484F58;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #21262D;
}
</style>
"""


def inject_css() -> None:
    """Inject the dark-theme CSS into the current Streamlit page.

    Call once at the top of ``app.py``::

        from src.ui.styles import inject_css
        inject_css()
    """
    import streamlit as st

    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
