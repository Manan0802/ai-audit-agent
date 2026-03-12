
# app.py — Automated QA Audit Dashboard
# Features: file_id + mcat_id, Context Mismatch verdict, 30s rate limiting,
# live streaming table, Gen-Z modern UI overhaul.

import os, re, time, io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from data_processor import (
    load_single_file,
    read_uploaded_file,
    get_all_file_ids,
    get_row_by_file_id,
    fetch_transcript_from_url,
    get_raw_json_keys,
    parse_and_filter_json,
)
from agent_logic import process_file_id

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QA Audit Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

RATE_LIMIT_SECONDS = 30  # Free tier: 4 RPM max

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Apply Inter font to content areas only — avoids breaking Streamlit internal UI elements */
.stApp, .stMarkdown, .stText, .stDataFrame,
[data-testid="stMarkdownContainer"], [data-testid="stText"],
.element-container p, .element-container span, .element-container div,
.stSelectbox label, .stToggle label, .stNumberInput label,
.stFileUploader label, .stTextInput label, .stTextArea label,
.stCaption, .stInfo, .stSuccess, .stError, .stWarning {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: radial-gradient(ellipse at top left, #0f1729 0%, #0a0e1a 50%, #060912 100%);
    color: #e8eaf2;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #080c18 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.18) !important;
}

.sidebar-brand {
    text-align: center;
    padding: 20px 0 28px;
}
.sidebar-brand .icon { font-size: 2.6rem; margin-bottom: 6px; }
.sidebar-brand h1 {
    font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #a78bfa 0%, #e879f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 4px;
}
.sidebar-brand p {
    font-size: 0.7rem; color: #4b5563; text-transform: uppercase;
    letter-spacing: 0.1em; margin: 0;
}

/* ── API Key badges ── */
.key-ok {
    background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3);
    border-radius: 10px; padding: 10px 14px; font-size: 0.82rem; color: #34d399;
    line-height: 1.6;
}
.key-missing {
    background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.28);
    border-radius: 10px; padding: 10px 14px; font-size: 0.82rem; color: #f87171;
    line-height: 1.6;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 16px; margin: 16px 0 24px; }

.mc {
    flex: 1; border-radius: 18px; padding: 22px 20px 18px;
    text-align: center; border: 1px solid transparent;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.25s ease;
}
.mc:hover { transform: translateY(-4px); }

.mc-total {
    background: rgba(139,92,246,0.12);
    border-color: rgba(139,92,246,0.3);
    box-shadow: 0 4px 24px rgba(139,92,246,0.12);
}
.mc-total:hover { box-shadow: 0 12px 40px rgba(139,92,246,0.25); }
.mc-correct {
    background: rgba(52,211,153,0.1);
    border-color: rgba(52,211,153,0.3);
    box-shadow: 0 4px 24px rgba(52,211,153,0.08);
}
.mc-correct:hover { box-shadow: 0 12px 40px rgba(52,211,153,0.2); }
.mc-wrong {
    background: rgba(248,113,113,0.1);
    border-color: rgba(248,113,113,0.3);
    box-shadow: 0 4px 24px rgba(248,113,113,0.08);
}
.mc-wrong:hover { box-shadow: 0 12px 40px rgba(248,113,113,0.2); }
.mc-mismatch {
    background: rgba(251,191,36,0.1);
    border-color: rgba(251,191,36,0.3);
    box-shadow: 0 4px 24px rgba(251,191,36,0.08);
}
.mc-mismatch:hover { box-shadow: 0 12px 40px rgba(251,191,36,0.18); }
.mc-accuracy {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
}

.mc .label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 10px;
}
.mc .num {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1;
}
.mc .num.purple  { color: #a78bfa; }
.mc .num.green   { color: #34d399; }
.mc .num.red     { color: #f87171; }
.mc .num.amber   { color: #fbbf24; }
.mc .num.indigo  { color: #818cf8; font-size: 2rem; }

.acc-bar-bg {
    background: rgba(255,255,255,0.07); border-radius: 99px;
    height: 5px; margin-top: 10px; overflow: hidden;
}
.acc-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #34d399, #6ee7b7);
    transition: width 0.5s ease;
}

/* ── Section headings ── */
.sh {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.75rem; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin: 32px 0 16px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(139,92,246,0.15);
}

/* ── Processing banners ── */
.banner {
    background: linear-gradient(90deg, rgba(139,92,246,0.12), rgba(232,121,249,0.06));
    border: 1px solid rgba(139,92,246,0.25); border-radius: 12px;
    padding: 12px 18px; font-size: 0.9rem; color: #c4b5fd;
    margin: 10px 0; line-height: 1.5;
}
.banner-rate {
    background: linear-gradient(90deg, rgba(251,191,36,0.1), rgba(245,158,11,0.05));
    border: 1px solid rgba(251,191,36,0.25); border-radius: 12px;
    padding: 12px 18px; font-size: 0.88rem; color: #fbbf24;
    margin: 8px 0;
}

/* ── Verdict pills ── */
.v-correct {
    display:inline-block; background:rgba(52,211,153,0.15);
    color:#34d399; border:1px solid rgba(52,211,153,0.3);
    border-radius:99px; padding:2px 10px; font-size:0.75rem; font-weight:600;
}
.v-wrong {
    display:inline-block; background:rgba(248,113,113,0.15);
    color:#f87171; border:1px solid rgba(248,113,113,0.3);
    border-radius:99px; padding:2px 10px; font-size:0.75rem; font-weight:600;
}
.v-mismatch {
    display:inline-block; background:rgba(251,191,36,0.15);
    color:#fbbf24; border:1px solid rgba(251,191,36,0.3);
    border-radius:99px; padding:2px 10px; font-size:0.75rem; font-weight:600;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 10px 24px !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] > div > div,
div[data-testid="stNumberInput"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.22) !important;
    border-radius: 10px !important; color: #e8eaf2 !important;
    font-size: 0.9rem !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02); border: 1px dashed rgba(139,92,246,0.3);
    border-radius: 14px; padding: 10px;
}

/* ── Toggle ── */
/* Multi-selector to hit Streamlit's toggle label regardless of DOM version */
[data-testid="stToggle"] label,
[data-testid="stToggle"] label p,
[data-testid="stToggle"] label span,
[class*="stToggle"] label {
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

/* ── Expander title ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
details summary p, details summary span {
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* ── General body text — bright on dark bg ── */
.stApp p, .stApp span, .stApp label,
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {
    color: #d1d5db;
}

/* ── Input / widget labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stFileUploader"] label,
[data-testid="stTextArea"] label {
    color: #d1d5db !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 14px !important; overflow: hidden; }
[data-testid="stDataFrame"] { font-size: 0.88rem !important; }
.stDataFrame td, .stDataFrame th {
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
    color: #d1d5db !important;
}

/* ── Dividers ── */
hr { border-color: rgba(139,92,246,0.15) !important; margin: 28px 0 !important; }

/* ── Captions ── */
.stCaption, .stCaption p { color: #9ca3af !important; font-size: 0.78rem !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #e879f9) !important;
    border-radius: 99px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# API KEY
# ──────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("LLM_GATEWAY_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "results": [],
        "processed_file_ids": set(),
        "data_df": None,
        "all_file_ids": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def results_as_dataframe() -> pd.DataFrame:
    if not st.session_state.results:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.results)
    # Ensure both columns exist
    for col in ["file_id", "mcat_id", "parameter", "extracted_value",
                "transcript_context", "verdict", "reason"]:
        if col not in df.columns:
            df[col] = ""
    return df[["file_id", "mcat_id", "parameter", "extracted_value",
               "transcript_context", "verdict", "reason"]]


def get_counts():
    df = results_as_dataframe()
    if df.empty:
        return 0, 0, 0, 0
    verdicts = df["verdict"].str.strip().str.lower()
    total = len(df)
    correct = (verdicts == "correct").sum()
    wrong = (verdicts == "wrong").sum()
    mismatch = (verdicts == "context mismatch").sum()
    return int(total), int(correct), int(wrong), int(mismatch)


def render_metric_cards(total, correct, wrong, mismatch):
    accuracy = (correct / total * 100) if total > 0 else 0
    # 5 metric cards in one row
    st.markdown(f"""
    <div class="metric-row">
      <div class="mc mc-total">
        <div class="label">📊 Total Verified</div>
        <div class="num purple">{total}</div>
      </div>
      <div class="mc mc-correct">
        <div class="label">✅ Correct</div>
        <div class="num green">{correct}</div>
      </div>
      <div class="mc mc-wrong">
        <div class="label">❌ Wrong</div>
        <div class="num red">{wrong}</div>
      </div>
      <div class="mc mc-mismatch">
        <div class="label">⚠️ Context Mismatch</div>
        <div class="num amber">{mismatch}</div>
      </div>
      <div class="mc mc-accuracy">
        <div class="label">🎯 Accuracy</div>
        <div class="num indigo">{accuracy:.1f}%</div>
        <div class="acc-bar-bg">
          <div class="acc-bar-fill" style="width:{accuracy:.1f}%"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_chart():
    df = results_as_dataframe()
    if df.empty:
        st.info("📈 Chart will appear after processing starts.", icon="ℹ️")
        return

    param_counts = df.groupby(["parameter", "verdict"]).size().reset_index(name="count")
    params = df["parameter"].unique().tolist()

    correct_c, wrong_c, mismatch_c = [], [], []
    for p in params:
        sub = param_counts[param_counts["parameter"] == p]
        def _get(v): return int(sub.loc[sub["verdict"].str.lower() == v, "count"].sum())
        correct_c.append(_get("correct"))
        wrong_c.append(_get("wrong"))
        mismatch_c.append(_get("context mismatch"))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="✅ Correct",          x=params, y=correct_c,
                          marker_color="#34d399", marker_line_width=0))
    fig.add_trace(go.Bar(name="❌ Wrong",             x=params, y=wrong_c,
                          marker_color="#f87171", marker_line_width=0))
    fig.add_trace(go.Bar(name="⚠️ Context Mismatch", x=params, y=mismatch_c,
                          marker_color="#fbbf24", marker_line_width=0))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", family="Inter", size=12),
        xaxis=dict(tickangle=-35, gridcolor="rgba(255,255,255,0.04)", title=None,
                   tickfont=dict(size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Count",
                   tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#c4b5fd", size=12)),
        margin=dict(l=20, r=20, t=44, b=60), height=340,
        bargap=0.18, bargroupgap=0.06,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_results_table(df: pd.DataFrame, placeholder=None):
    """Render the styled results table. Optionally into a st.empty() placeholder."""
    if df.empty:
        msg = st.empty() if placeholder is None else placeholder
        msg.info("No results yet — process a File ID to see results here.", icon="🔍")
        return

    def highlight_verdict(row):
        styles = [""] * len(row)
        try:
            col = next((c for c in row.index if c.lower() == "verdict"), None)
            if col is None:
                return styles
            v = str(row[col]).strip().lower()
            idx = list(row.index).index(col)
            if v == "correct":
                styles[idx] = "background-color:rgba(52,211,153,0.13);color:#34d399;font-weight:600"
            elif v == "context mismatch":
                styles[idx] = "background-color:rgba(251,191,36,0.13);color:#fbbf24;font-weight:600"
            else:
                styles[idx] = "background-color:rgba(248,113,113,0.13);color:#f87171;font-weight:600"
        except Exception:
            pass
        return styles

    styled = df.style.apply(highlight_verdict, axis=1)
    try:
        styled = styled.relabel_index(
            ["File ID", "mcat_id", "Parameter", "Extracted Value",
             "Transcript Context", "Verdict", "Reason"],
            axis="columns",
        )
    except Exception:
        pass

    target = placeholder if placeholder is not None else st
    target.dataframe(styled, use_container_width=True, height=460)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class="sidebar-brand">
        <div class="icon">🔍</div>
        <h1>QA Audit Agent</h1>
        <p>Powered by Gemini 2.5 Flash</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── API Key ────────────────────────────────────────────────────────────────
    st.markdown("### 🔑 API Key")
    if API_KEY:
        masked = API_KEY[:8] + "•" * 10 + API_KEY[-4:]
        st.markdown(f'<div class="key-ok">✅ Loaded from .env<br>'
                    f'<code style="font-size:0.78rem">{masked}</code></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="key-missing">⚠️ LLM_GATEWAY_API_KEY not found in .env<br>'
                    'Add: <code>LLM_GATEWAY_API_KEY=your_key</code></div>',

                    unsafe_allow_html=True)

    st.divider()

    # ── File Upload ────────────────────────────────────────────────────────────
    st.markdown("### 📂 Upload Input File")
    st.caption("CSV/Excel with: `file_id` · `mcat_id` · `TranscriptionURL` · `llm_extracted_json`")
    uploaded_file = st.file_uploader(
        "Input File (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="main_uploader",
    )

    if uploaded_file:
        try:
            # Step 1: Read raw to get column names — no processing yet
            raw_df = read_uploaded_file(uploaded_file)
            raw_cols = list(raw_df.columns)

            # ── DEBUG: Show exactly what was read ─────────────────────────────
            st.caption(f"📋 Columns detected ({len(raw_cols)}): `{'` · `'.join(raw_cols)}`")
            with st.expander("🔬 DEBUG: Raw file contents (first 3 rows)", expanded=False):
                st.write(f"**Shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
                st.write(f"**Column names:** {raw_cols}")
                st.write("**First 3 data rows:**")
                st.dataframe(raw_df.head(3), use_container_width=True)

            # Auto-detect which column looks like file_id
            _lc = {c.lower().replace(" ", "").replace("_", ""): c for c in raw_cols}
            _auto_fid = next(
                (v for k, v in _lc.items()
                 if k in ("fileid", "file_id", "uniqueid", "uid", "unnamed0")),
                None
            )
            _default_idx = raw_cols.index(_auto_fid) if _auto_fid else 0

            # Step 2: Dropdown selector
            fid_col_sel = st.selectbox(
                "🔑 File ID Column (from detected columns)",
                options=raw_cols,
                index=_default_idx,
                key="fid_col_sel",
                help="Pick whichever column holds the unique File ID",
            )

            # Step 2b: Manual text override — use this if the column isn't in the list
            manual_fid = st.text_input(
                "✏️ Or type exact column name:",
                value="",
                placeholder="e.g.  file_id",
                key="fid_manual_override",
                help="Type the exact column name from the list above if the dropdown doesn't select it correctly",
            )
            if manual_fid.strip():  # manual input takes priority
                fid_col_sel = manual_fid.strip()

            # Step 3: Load with the user's explicit column override
            with st.spinner("Loading…"):
                df = load_single_file(raw_df, file_id_col_override=fid_col_sel)
                all_fids = get_all_file_ids(df)
                st.session_state.data_df = df
                st.session_state.all_file_ids = all_fids

            unprocessed = [f for f in all_fids if f not in st.session_state.processed_file_ids]
            st.success(f"✅ **{len(all_fids)}** records · **{len(unprocessed)}** pending")

            # Step 4: Preview — let user verify the mapping is correct
            with st.expander("👁️ Verify: file_id vs mcat_id mapping", expanded=True):
                st.caption("Check that file_id and mcat_id columns show the correct values:")
                st.dataframe(
                    df[["file_id", "mcat_id"]].head(8).reset_index(drop=True),
                    use_container_width=True,
                    height=220,
                )
                if (df["file_id"] == df["mcat_id"]).all():
                    st.warning(
                        "⚠️ file_id = mcat_id for ALL rows. "
                        "Use the text box above to type the correct column name.",
                        icon="⚠️"
                    )
                else:
                    st.success("✅ file_id and mcat_id are distinct — mapping looks correct!")

        except Exception as e:
            st.error(f"❌ {e}")
            st.session_state.data_df = None
            st.session_state.all_file_ids = []
    else:
        st.caption("Upload your file above to begin.")


    st.divider()

    # ── Rate Limit Config ──────────────────────────────────────────────────────
    st.markdown("### ⏱️ Rate Limit (Batch Mode)")
    rate_sleep = st.number_input(
        "Seconds between API calls",
        min_value=5, max_value=120, value=RATE_LIMIT_SECONDS, step=5,
        help="Free tier: max 4 RPM → 30s safe. Decrease if on paid plan.",
    )

    st.divider()

    # ── URL Test ───────────────────────────────────────────────────────────────
    st.markdown("### 🔗 Test Transcript URL")
    test_url = st.text_input("URL to test fetch", placeholder="https://…", key="test_url")
    if st.button("🧪 Fetch & Preview", key="btn_url_test"):
        if test_url.strip():
            with st.spinner("Fetching…"):
                try:
                    txt = fetch_transcript_from_url(test_url.strip())
                    st.success(f"✅ {len(txt)} characters")
                    st.text_area("Preview", txt[:400], height=110)
                except Exception as e:
                    st.error(f"❌ {e}")

    st.divider()
    if st.button("🗑️ Clear All Results"):
        st.session_state.results = []
        st.session_state.processed_file_ids = set()
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 12px 0 4px;">
  <h1 style="background:linear-gradient(135deg,#a78bfa,#e879f9);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             font-size:2.1rem;font-weight:900;letter-spacing:-0.03em;margin-bottom:6px;">
    🔍 Automated QA Audit Dashboard
  </h1>
  <p style="color:#4b5563;font-size:0.85rem;margin:0;">
    AI cross-verification of call data against transcripts &nbsp;·&nbsp; Gemini 2.5 Flash
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# LIVE ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
total, correct, wrong, mismatch = get_counts()
st.markdown('<div class="sh">📊 Live Analytics</div>', unsafe_allow_html=True)
render_metric_cards(total, correct, wrong, mismatch)
st.markdown('<div class="sh">📈 Parameter Accuracy Chart</div>', unsafe_allow_html=True)
render_chart()
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# CONTROL HUB
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh">⚙️ Control Hub</div>', unsafe_allow_html=True)

data_df     = st.session_state.get("data_df")
all_fids    = st.session_state.get("all_file_ids", [])
pending_ids = [f for f in all_fids if f not in st.session_state.processed_file_ids]
ready       = bool(data_df is not None and all_fids and API_KEY)

if data_df is None:
    st.warning("⬅️ Upload your input file in the sidebar.", icon="📂")
elif not API_KEY:
    st.warning("⚠️ LLM_GATEWAY_API_KEY missing from .env", icon="🔑")

# ── Mode toggle ────────────────────────────────────────────────────────────────
auto_mode = st.toggle(
    "🤖  Batch Auto-Process All File IDs  (OFF = Audit Mode)",
    value=False, disabled=not ready, key="auto_toggle",
)

# ── JSON Key Inspector ─────────────────────────────────────────────────────────
if data_df is not None and all_fids:
    with st.expander("🔬 JSON Key Inspector — debug parameter mapping", expanded=False):
        inspect_fid = st.selectbox("Pick file_id to inspect", options=all_fids, key="inspect_sel")
        if inspect_fid:
            irow = data_df[data_df["file_id"] == inspect_fid]
            if not irow.empty:
                raw_json = irow.iloc[0]["llm_extracted_json"]
                all_keys, ierr = get_raw_json_keys(raw_json)
                ifiltered = parse_and_filter_json(raw_json)
                matched = set(ifiltered.keys())
                if ierr:
                    st.error(f"❌ {ierr}")
                else:
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown("**📋 All JSON keys found:**")
                        for k in all_keys:
                            nk = re.sub(r'[\s_\-]', '', k.lower())
                            mp = next((p for p in matched
                                       if re.sub(r'[\s_\-]', '', p.lower()) == nk), None)
                            if mp:
                                st.markdown(f"✅ `{k}` → **{mp}**")
                            else:
                                st.markdown(f"⚪ `{k}` *(not a target param)*")
                    with cb:
                        st.markdown("**🎯 Matched parameters:**")
                        if matched:
                            for p in matched:
                                st.markdown(f"✅ **{p}**: `{str(ifiltered[p])[:70]}`")
                        else:
                            st.error("❌ No parameters matched!")
                st.caption(f"Raw JSON (first 300 chars): `{str(raw_json)[:300]}`")

st.markdown("<br>", unsafe_allow_html=True)
ctrl1, ctrl2 = st.columns([3, 1])

# ──────────────────────────────────────────────────────────────────────────────
# AUDIT MODE (manual single ID)
# ──────────────────────────────────────────────────────────────────────────────
if not auto_mode:
    with ctrl1:
        sel_fid = st.selectbox(
            "Select File ID to Audit",
            options=all_fids if all_fids else ["— upload file first —"],
            disabled=not ready, key="single_sel",
        )
    with ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        do_single = st.button("▶ Process Single ID", disabled=not ready, key="btn_single")

    if do_single and ready and sel_fid:
        if sel_fid in st.session_state.processed_file_ids:
            st.info(f"file_id **{sel_fid}** already processed. Clear results or pick another.",
                    icon="ℹ️")
        else:
            ph = st.empty()
            try:
                ph.markdown(f'<div class="banner">🌐 Fetching transcript for <b>{sel_fid}</b>…</div>',
                            unsafe_allow_html=True)
                row_data = get_row_by_file_id(data_df, sel_fid)
                ph.markdown(f'<div class="banner">🤖 Auditing <b>{sel_fid}</b> with Gemini 2.5 Flash…</div>',
                            unsafe_allow_html=True)
                results, err = process_file_id(
                    api_key=API_KEY,
                    file_id=sel_fid,
                    mcat_id=row_data["mcat_id"],
                    transcript=row_data["transcript"],
                    filtered_json=row_data["filtered_json"],
                )
                ph.empty()
                if err:
                    st.error(f"❌ {err}")
                else:
                    st.session_state.results.extend(results)
                    st.session_state.processed_file_ids.add(sel_fid)
                    st.success(
                        f"✅ **{sel_fid}** (mcat_id: {row_data['mcat_id']}) — "
                        f"{len(results)} parameters verified."
                    )
                    st.rerun()
            except Exception as e:
                ph.empty()
                st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# BATCH MODE (auto-process all with rate limiting + live table)
# ──────────────────────────────────────────────────────────────────────────────
else:
    with ctrl1:
        st.markdown(
            f'<div class="banner">🚀 <b>Batch Mode</b> — '
            f'{len(pending_ids)} of {len(all_fids)} file IDs remaining &nbsp;·&nbsp; '
            f'Rate limit: {rate_sleep}s between calls</div>',
            unsafe_allow_html=True,
        )
    with ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_ph = st.empty()
        start_batch = btn_ph.button(
            "🚀 Start Auto Processing",
            disabled=not ready or not pending_ids,
            key="btn_batch",
        )

    if start_batch and ready and pending_ids:
        # Swamp in a Stop Button during execution
        # (clicking it causes a restart logic where start_batch is False)
        btn_ph.button("🛑 Stop Auto Processing", key="btn_stop", help="Click to safely halt the batch process")
        
        n = len(pending_ids)
        progress_bar = st.progress(0, text="Starting…")
        status_box   = st.empty()
        countdown_box = st.empty()

        st.markdown('<div class="sh">🔄 Live Results Stream</div>', unsafe_allow_html=True)
        live_table = st.empty()

        for i, fid in enumerate(list(pending_ids)):
            pct = int(i / n * 100)
            progress_bar.progress(pct, text=f"Processing {i+1}/{n}: {fid}")

            # Step 1: Fetch transcript
            status_box.markdown(
                f'<div class="banner">🌐 [{i+1}/{n}] Fetching transcript for <b>{fid}</b>…</div>',
                unsafe_allow_html=True)
            try:
                row_data = get_row_by_file_id(data_df, fid)
            except Exception as fe:
                status_box.error(f"❌ {fid}: Transcript fetch failed — {fe}")
                continue

            # Step 2: Gemini audit
            status_box.markdown(
                f'<div class="banner">🤖 [{i+1}/{n}] Auditing <b>{fid}</b> '
                f'(mcat: {row_data["mcat_id"]}) with Gemini 2.5 Flash…</div>',
                unsafe_allow_html=True)
            try:
                results, err = process_file_id(
                    api_key=API_KEY,
                    file_id=fid,
                    mcat_id=row_data["mcat_id"],
                    transcript=row_data["transcript"],
                    filtered_json=row_data["filtered_json"],
                )
                if err:
                    status_box.error(f"❌ {fid}: {err}")
                else:
                    st.session_state.results.extend(results)
                    st.session_state.processed_file_ids.add(fid)
                    status_box.success(
                        f"✅ [{i+1}/{n}] {fid} — {len(results)} params verified"
                    )
            except Exception as e:
                status_box.error(f"❌ Unexpected error for {fid}: {e}")

            # ── Live table update ─────────────────────────────────────────────
            cur_df = results_as_dataframe()
            if not cur_df.empty:
                render_results_table(cur_df, live_table)

            # ── Rate limit countdown (skip after last item) ───────────────────
            if i < n - 1:
                for t in range(int(rate_sleep), 0, -1):
                    countdown_box.markdown(
                        f'<div class="banner-rate">'
                        f'⏳ Rate limit pause &nbsp;—&nbsp; next call in <b>{t}s</b> '
                        f'&nbsp;·&nbsp; {i+1}/{n} done</div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(1)
                countdown_box.empty()

        progress_bar.progress(100, text="✅ All file IDs processed!")
        status_box.success("🎉 Batch processing complete!")
        time.sleep(1.5)
        st.rerun()

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE (bottom, persistent)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh">📋 Audit Results</div>', unsafe_allow_html=True)

results_df = results_as_dataframe()

if not results_df.empty:
    # Filter bar
    fa, fb, fc, fd = st.columns([2, 2, 2, 2])
    with fa:
        fv = st.selectbox("Verdict", ["All", "Correct", "Wrong", "Context Mismatch"], key="fv")
    with fb:
        all_params = sorted(results_df["parameter"].unique().tolist())
        fp = st.selectbox("Parameter", ["All"] + all_params, key="fp")
    with fc:
        all_mcat = sorted(results_df["mcat_id"].unique().tolist())
        fm = st.selectbox("mcat_id", ["All"] + all_mcat, key="fm")
    with fd:
        all_fid = sorted(results_df["file_id"].unique().tolist())
        ff = st.selectbox("File ID", ["All"] + all_fid, key="ff")

    disp = results_df.copy()
    if fv != "All":    disp = disp[disp["verdict"].str.lower() == fv.lower()]
    if fp != "All":    disp = disp[disp["parameter"] == fp]
    if fm != "All":    disp = disp[disp["mcat_id"] == fm]
    if ff != "All":    disp = disp[disp["file_id"] == ff]

    st.caption(f"Showing **{len(disp)}** of **{len(results_df)}** results")
    render_results_table(disp)

    # Download
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    _, dc, _ = st.columns([1, 2, 1])
    with dc:
        st.download_button(
            "⬇️ Download Full Results as CSV",
            data=csv_bytes, file_name="qa_audit_results.csv",
            mime="text/csv", use_container_width=True,
        )
else:
    st.info(
        "Results will appear here once you process a File ID. "
        "Upload your file and click **▶ Process Single ID** to begin.",
        icon="🔍",
    )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#374151;font-size:0.73rem;'>"
    "QA Audit Dashboard &nbsp;·&nbsp; Gemini 2.5 Flash &nbsp;·&nbsp; "
    "python-dotenv &nbsp;·&nbsp; Streamlit</p>",
    unsafe_allow_html=True,
)
