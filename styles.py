import streamlit as st

_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════
   DAAD — Design System  ·  Sidebar IDE (Enfoque 2)
   paper #fafaf7  ·  paperAlt #f3f0e8  ·  ink #1a1a1a
   pencil #6b6b6b  ·  ghost #d8d4cb  ·  accent #d2502a
   IBM Plex Sans  +  IBM Plex Mono
   border-radius: 3px  ·  border: 1.4px solid
   ═══════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&family=Caveat:wght@400;600&display=swap');

/* ── 1. APP SHELL ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif;
}
/* Zoom por defecto de la app (equivalente al 110% de Chrome) */
html {
    zoom: 1.1;
}
.stApp {
    background-color: #fafaf7;
}
/* Remove default top padding so page headers can sit flush */
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 40px;
}

/* ── 2. SIDEBAR SHELL ─────────────────────────────────────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {
    background-color: #f3f0e8 !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid #d8d4cb !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 !important;
}
/* All direct content inside sidebar gets consistent side padding */
[data-testid="stSidebarContent"] > div {
    padding-left: 16px;
    padding-right: 16px;
    padding-bottom: 20px;
}

/* ── 3. SIDEBAR CHROME HEADER (injected HTML) ──────────────── */
.daad-chrome {
    display: flex;
    align-items: center;
    height: 34px;
    padding: 0 14px;
    background: #ede9df;
    border-bottom: 1px solid #d8d4cb;
    margin: 0 -16px 16px;
    gap: 8px;
    flex-shrink: 0;
}
.daad-chrome-dots {
    display: flex;
    gap: 5px;
    align-items: center;
}
/* Círculos de la barra estilo macOS */
.daad-chrome-dots span {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    border: none;          /* sin borde */
    display: inline-block;
}

/* Asignar color a cada círculo */
.daad-chrome-dots span:nth-child(1) { background-color: #FF5F57; }  /* rojo   */
.daad-chrome-dots span:nth-child(2) { background-color: #FFBD2E; }  /* amarillo */
.daad-chrome-dots span:nth-child(3) { background-color: #28CA41; }  /* verde  */
.daad-chrome-url {
    flex: 1;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #6b6b6b;
    text-align: center;
    letter-spacing: 0.5px;
}
.daad-chrome-mark {
    font-family: 'Caveat', cursive;
    font-size: 17px;
    color: #d2502a;
    font-weight: 600;
    line-height: 1;
}

/* ── 4. SIDEBAR SECTION LABELS ────────────────────────────── */
.daad-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #6b6b6b;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    padding: 14px 0 6px;
    border-top: 1px solid #d8d4cb;
    margin-top: 4px;
    display: block;
}
.daad-section-label:first-of-type {
    border-top: none;
    padding-top: 4px;
}

/* Authors block */
.daad-authors {
    padding: 0px 0 50px;
}
.daad-authors p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #6b6b6b !important;
    line-height: 1.8 !important;
    margin: 0 !important;
}

/* ── 5. SIDEBAR TYPOGRAPHY OVERRIDES ──────────────────────── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 13px !important;
    color: #1a1a1a;
}
[data-testid="stSidebar"] .stCaption p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #6b6b6b !important;
}
/* Suppress default Streamlit subheaders in sidebar — we use .daad-section-label */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    color: #6b6b6b !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 14px 0 6px !important;
}

/* ── 6. PAGE HEADER (main area, injected HTML) ─────────────── */
.daad-page-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 16px 0 14px;
    border-bottom: 1px solid #d8d4cb;
    margin-bottom: 28px;
    width: 100%;
}
.daad-page-icon {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #d2502a;
    background: rgba(210,80,42,0.1);
    border: 1.3px solid #d2502a;
    border-radius: 3px;
    padding: 2px 7px;
    letter-spacing: 0;
    line-height: 1.4;
    white-space: nowrap;
}
.daad-page-title {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.3px;
    line-height: 1;
}
.daad-page-crumb {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6b6b6b;
    letter-spacing: 0;
    line-height: 1;
}

/* ── 7. GLOBAL TYPOGRAPHY ─────────────────────────────────── */
h1 {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.3px;
    margin-bottom: 0 !important;
}
h2 {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
}
h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    color: #6b6b6b !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 6px !important;
}
p, .stMarkdown p {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif;
    font-size: 13px;
    color: #1a1a1a;
    line-height: 1.55;
}

/* ── 8. METRIC CARDS ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #f3f0e8 !important;
    border: 1.4px solid #d8d4cb !important;
    border-left: 2px solid #d2502a !important;
    border-radius: 3px !important;
    padding: 14px 16px 12px !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 500 !important;
    color: #6b6b6b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.9px !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
    line-height: 1.1 !important;
    padding-top: 2px;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #6b6b6b !important;
}

/* ── 9. DATAFRAME ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    overflow: auto !important;
    width: 100% !important;
    max-width: 100% !important;
}
/* Glide data grid (Streamlit >= 1.18) */
.glideDataEditor {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    width: 100% !important;
}
[data-testid="stDataFrame"] > div {
    justify-content: flex-start !important; /* AÑADIDO: Alinea la tabla a la izquierda si el zoom la descentra */
}
/* Classic table fallback */
[data-testid="stDataFrame"] table {
    border-collapse: collapse !important;
    width: 100% !important;
}
[data-testid="stDataFrame"] thead th {
    background: #f3f0e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    color: #6b6b6b !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border-bottom: 1.2px solid #d8d4cb !important;
    padding: 7px 10px !important;
    white-space: nowrap;
}
[data-testid="stDataFrame"] tbody td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #1a1a1a !important;
    border-bottom: 1px solid #d8d4cb !important;
    padding: 5px 10px !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: rgba(210,80,42,0.04) !important;
}

/* ── 10. BUTTONS ──────────────────────────────────────────── */
.stButton > button {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    background-color: transparent !important;
    border: 1.4px solid #1a1a1a !important;
    border-radius: 3px !important;
    color: #1a1a1a !important;
    padding: 5px 14px !important;
    min-height: 32px !important;
    line-height: 1.2 !important;
    transition: background 0.1s ease, border-color 0.1s ease !important;
    box-shadow: none !important;
    letter-spacing: 0.1px;
}
.stButton > button:hover {
    background-color: #d8d4cb !important;
    border-color: #1a1a1a !important;
    color: #1a1a1a !important;
    box-shadow: none !important;
}
.stButton > button:active {
    background-color: #c8c4bb !important;
    transform: translateY(1px);
}
.stButton > button:focus {
    box-shadow: 0 0 0 2px rgba(210,80,42,0.25) !important;
    outline: none !important;
}
/* Primary variant */
.stButton > button[kind="primary"] {
    background-color: #d2502a !important;
    border-color: #d2502a !important;
    color: #fafaf7 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #b8442a !important;
    border-color: #b8442a !important;
    color: #fafaf7 !important;
}

/* ── 11. TEXT INPUT ───────────────────────────────────────── */
[data-testid="stTextInputRootElement"] > div,
.stTextInput > div > div {
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    background: #fafaf7 !important;
    box-shadow: none !important;
}
[data-testid="stTextInputRootElement"] input,
.stTextInput input {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: #1a1a1a !important;
    background: transparent !important;
    border: none !important;
    padding: 6px 10px !important;
    box-shadow: none !important;
}
[data-testid="stTextInputRootElement"]:focus-within > div,
.stTextInput > div > div:focus-within {
    border-color: #d2502a !important;
    box-shadow: 0 0 0 1px rgba(210,80,42,0.2) !important;
}
/* Number input */
[data-testid="stNumberInputContainer"],
.stNumberInput > div > div {
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    background: #fafaf7 !important;
}
[data-testid="stNumberInputContainer"] input {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    background: transparent !important;
}
/* Textarea */
.stTextArea > div > div {
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
}
.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    background: #fafaf7 !important;
}
/* Input labels */
[data-testid="stWidgetLabel"] p,
.stTextInput label p,
.stSelectbox label p,
.stMultiSelect label p,
.stNumberInput label p,
.stTextArea label p,
.stFileUploader label p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    color: #6b6b6b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    margin-bottom: 5px !important;
}

/* ── 12. PASSWORD INPUT ───────────────────────────────────── */
.stTextInput input[type="password"] {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 2px;
}

/* ── 13. SELECTBOX ────────────────────────────────────────── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 13px !important;
    background: #fafaf7 !important;
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    box-shadow: none !important;
    min-height: 36px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: #d2502a !important;
    box-shadow: 0 0 0 1px rgba(210,80,42,0.2) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 13px !important;
    color: #1a1a1a !important;
}
/* Dropdown list */
[data-baseweb="popover"] {
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    box-shadow: 2px 4px 16px rgba(0,0,0,0.1) !important;
    background: #fafaf7 !important;
}
[data-baseweb="popover"] ul li {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 13px !important;
    color: #1a1a1a !important;
    padding: 7px 12px !important;
}
[data-baseweb="popover"] ul li:hover {
    background: #f3f0e8 !important;
}
[data-baseweb="popover"] [aria-selected="true"] {
    background: rgba(210,80,42,0.1) !important;
    color: #d2502a !important;
}

/* ── 14. MULTISELECT ──────────────────────────────────────── */
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    background: #fafaf7 !important;
    min-height: 36px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within > div {
    border-color: #d2502a !important;
}
[data-baseweb="tag"] {
    background-color: #d2502a !important;
    border-radius: 3px !important;
    border: none !important;
    height: 22px !important;
    padding: 0 8px !important;
}
[data-baseweb="tag"] span {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #fafaf7 !important;
    font-weight: 500;
}
[data-baseweb="tag"] button svg { fill: #fafaf7 !important; }

/* ── 15. TABS ─────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #d8d4cb !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #6b6b6b !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 9px 16px !important;
    margin: 0 !important;
    transition: color 0.1s;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #1a1a1a !important;
    background: rgba(0,0,0,0.03) !important;
}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
    color: #d2502a !important;
    border-bottom-color: #d2502a !important;
    font-weight: 600 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #d2502a !important;
    height: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: #d8d4cb !important;
    height: 1px !important;
}

/* ── 16. RADIO ────────────────────────────────────────────── */
[data-testid="stRadio"] > label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    color: #6b6b6b !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}
[data-testid="stRadio"] div[role="radiogroup"] label > div:last-child p {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 13px !important;
    color: #1a1a1a !important;
    transition: color 0.1s;
}
[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] > div {
    border-color: #d8d4cb !important;
    background: transparent !important;
}
[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] input:checked ~ div {
    border-color: #d2502a !important;
    background: #d2502a !important;
}

/* Navigation radio: full-bleed styled items */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 2px !important;
    display: flex;
    flex-direction: column;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 7px 10px !important;
    border-radius: 3px !important;
    margin: 0 !important;
    border: 1.4px solid transparent !important;
    transition: background 0.1s, border-color 0.1s;
    cursor: pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(0,0,0,0.06) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: #d2502a !important;
    border-color: #d2502a !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) > div:last-child p {
    color: #fafaf7 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) [data-baseweb="radio"] > div {
    border-color: #fafaf7 !important;
    background: #fafaf7 !important;
}

/* ── 17. CHECKBOX ─────────────────────────────────────────── */
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child {
    border: 1.4px solid #d8d4cb !important;
    border-radius: 2px !important;
    background: transparent !important;
}
[data-testid="stCheckbox"] input:checked ~ [data-baseweb="checkbox"] > div:first-child {
    background: #d2502a !important;
    border-color: #d2502a !important;
}
[data-testid="stCheckbox"] label p {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 13px !important;
    color: #1a1a1a !important;
}

/* ── 18. FILE UPLOADER ────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    border: 1.4px dashed #d8d4cb !important;
    border-radius: 3px !important;
    background: #f3f0e8 !important;
    transition: border-color 0.12s, background 0.12s;
    padding: 16px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #d2502a !important;
    background: rgba(210,80,42,0.04) !important;
}
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section span {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 12px !important;
    color: #6b6b6b !important;
}
[data-testid="stFileUploader"] section button {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    background: #fafaf7 !important;
    color: #1a1a1a !important;
    padding: 4px 12px !important;
}
[data-testid="stFileUploader"] section button:hover {
    border-color: #d2502a !important;
    color: #d2502a !important;
}

/* ── 19. DIVIDERS ─────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #d8d4cb !important;
    margin: 14px 0 !important;
    opacity: 1 !important;
}

/* ── 20. ALERTS ───────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 3px !important;
    padding: 10px 14px !important;
}
[data-testid="stAlert"] p {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}
/* success */
div[data-baseweb="notification"][role="alert"] {
    border-radius: 3px !important;
}

/* ── 21. CODE BLOCKS ──────────────────────────────────────── */
code {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11.5px !important;
    background: #f3f0e8 !important;
    border: 1px solid #d8d4cb !important;
    border-radius: 3px !important;
    padding: 1px 5px !important;
    color: #d2502a !important;
}
pre {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    background: #f3f0e8 !important;
    border: 1.4px solid #d8d4cb !important;
    border-radius: 3px !important;
    padding: 12px 16px !important;
    line-height: 1.5;
}
pre code {
    color: #1a1a1a !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    font-size: inherit !important;
}

/* ── 22. CAPTION / SMALL TEXT ─────────────────────────────── */
.stCaption p, small {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #6b6b6b !important;
}

/* ── 23. SPINNER ──────────────────────────────────────────── */
[data-testid="stSpinner"] > div > div {
    border-color: #d2502a transparent transparent transparent !important;
}

/* ── 24. PROGRESS BAR ─────────────────────────────────────── */
[data-testid="stProgressBar"] > div {
    background: #d8d4cb !important;
    border-radius: 1px !important;
    height: 3px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: #d2502a !important;
    border-radius: 1px !important;
}

/* ── 25. CUSTOM WELCOME SCREEN ────────────────────────────── */
.daad-welcome {
    padding: 48px 8px;
}
.daad-welcome-title {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.5px;
    line-height: 1;
    margin-bottom: 12px;
}
.daad-welcome-accent {
    display: inline-block;
    width: 32px;
    height: 3px;
    background: #d2502a;
    border-radius: 0;
    margin-bottom: 20px;
}
.daad-welcome-sub {
    font-family: 'IBM Plex Sans', Helvetica, sans-serif;
    font-size: 14px;
    color: #6b6b6b;
    line-height: 1.6;
    max-width: 560px;
    margin-bottom: 32px;
}
.daad-source-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    max-width: 580px;
    margin-bottom: 12px;
}
.daad-source-card {
    border: 1.4px solid #d8d4cb;
    border-radius: 3px;
    padding: 12px 14px;
    background: #f3f0e8;
}
.daad-source-card .s-icon {
    font-size: 18px;
    margin-bottom: 6px;
    display: block;
    line-height: 1;
}
.daad-source-card .s-name {
    display: block;
    font-family: 'IBM Plex Sans', Helvetica, sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #1a1a1a;
}
.daad-source-card .s-fmt {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #6b6b6b;
    margin-top: 3px;
    letter-spacing: 0.3px;
}
.daad-hint {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6b6b6b;
    border: 1px solid #d8d4cb;
    border-radius: 3px;
    display: inline-block;
    padding: 5px 12px;
    background: #f3f0e8;
}

/* ── 26. WIP SECTION PLACEHOLDER ─────────────────────────── */
.daad-wip {
    border: 1.4px dashed #d8d4cb;
    border-radius: 3px;
    padding: 32px 24px;
    background: #f3f0e8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #6b6b6b;
    text-align: center;
    margin-top: 8px;
    line-height: 2;
}

/* ── 27. SCROLLBARS ───────────────────────────────────────── */
* { scrollbar-width: thin; scrollbar-color: #d8d4cb transparent; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: #d8d4cb;
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover { background: #6b6b6b; }

/* ── 28. COLUMN GAPS ──────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    gap: 16px !important;
    align-items: stretch;
}
/* Columns holding metrics should stretch */
[data-testid="stHorizontalBlock"] > div {
    flex: 1;
}

/* ── 29. HIDE DEFAULT STREAMLIT CHROME ────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── 30. SUBHEADER ACCENT LINE ────────────────────────────── */
.daad-subheader {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 600;
    color: #6b6b6b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #d8d4cb;
    margin-bottom: 12px;
    margin-top: 4px;
    display: block;
}
.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* ═══ SIDEBAR: sin espacio superior, flechas nativas, fondo transparente ═══ */

/* 1. Sidebar pegado al borde superior */
section[data-testid="stSidebar"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
    position: relative !important;   /* necesario para el botón absoluto */
}

/* 2. Contenido sin relleno extra */
[data-testid="stSidebarContent"] {
    padding-top: 0 !important;
}
[data-testid="stSidebarContent"] > *:first-child {
    margin-top: 0 !important;
}

/* 3. Contenedor del botón de colapso no ocupa espacio */
[data-testid="stSidebarHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* 4. Botón de colapso: absoluto, esquina superior derecha, fondo transparente */
[data-testid="stSidebarCollapseButton"] {
    position: absolute !important;
    top: 6px !important;            /* ajusta según necesites */
    right: 14px !important;        /* alineado con el padding de la barra */
    z-index: 300 !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;   /* ← sin fondo */
    border: none !important;
    cursor: pointer;
}

/* 5. Mostrar las flechas nativas (sin ocultar el SVG) */
/*    No añadimos regla para svg, así que se ven por defecto */
/* Reducir espacio entre la primera etiqueta y los autores */
.daad-section-label:first-of-type {
    padding-bottom: 0 !important;   /* elimina los 6px de abajo */
}
.daad-authors {
    padding-top: 0 !important;      /* ya lo tenías, por si acaso */
}
.daad-authors p:first-child {
    margin-top: -12px;               /* acerca aún más el primer nombre */
}
.daad-section-label:first-of-type {
    padding-top: 12px !important;   /* antes 4px; ajústalo a tu gusto */
}
</style>
"""


def apply_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
