"""
DataLite — Auto-Insight EDA
===========================

Phase 1: DataLite now leads with *auto-insights*. The moment you load a
dataset, a deterministic engine (insight_engine.py) scans the full data and
surfaces the handful of findings that actually matter, ranked by how
interesting they are — no questions to ask, no API key required.

The classic manual EDA lives below for when you want to drill in yourself.
"""

import re
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

import insight_engine as ie

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="DataLite — Auto-Insight EDA",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10})

st.markdown(
    """
    <style>
      .insight-badge {
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:11px; font-weight:600; letter-spacing:.3px;
        color:#fff; margin-bottom:6px;
      }
      .insight-headline { font-size:15px; font-weight:650; line-height:1.35;
        margin:2px 0 4px 0; }
      .insight-detail { font-size:12.5px; color:#5b6470; line-height:1.4; }
      .hero-sub { color:#5b6470; font-size:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Visual identity for each detector type
KIND_META = {
    "segment_difference": ("Group difference", "#4338ca"),
    "correlation":        ("Correlation",      "#0e7490"),
    "missingness_pattern":("Missing pattern",  "#b45309"),
    "missingness":        ("Missing data",     "#b45309"),
    "imbalance":          ("Imbalance",        "#9333ea"),
    "outliers":           ("Outliers",         "#dc2626"),
    "duplicates":         ("Duplicates",       "#be185d"),
    "hygiene":            ("Data hygiene",     "#475569"),
}

# --------------------------------------------------------------------------- #
# Sidebar: data source (+ optional, clearly non-blocking AI narration key)
# --------------------------------------------------------------------------- #
SAMPLE_CSV = """Name,Age,Gender,Score,Passed
Alice,23,Female,85,Yes
Bob,25,Male,75,Yes
Charlie,22,Male,50,No
Diana,24,Female,95,Yes
Edward,21,Male,45,No
Fay,26,Female,88,Yes"""

with st.sidebar:
    st.title("📁 Data source")
    data_source = st.radio("Choose data source:", ("Use sample data", "Upload CSV"))

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.markdown("---")
    with st.expander("🤖 Optional: AI narration (Groq)"):
        st.caption(
            "Insights work fully without this. A key only upgrades the wording "
            "and unlocks the experimental chat — it's never required."
        )
        api_key = st.text_input("Groq API key", type="password",
                                label_visibility="collapsed",
                                placeholder="gsk_… (optional)")
        st.caption("Get one from [GroqCloud](https://console.groq.com/).")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
df = None
if data_source == "Use sample data":
    df = pd.read_csv(StringIO(SAMPLE_CSV))
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read that CSV: {e}")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown("## ✨ DataLite")
st.markdown(
    "<div class='hero-sub'>Drop in a dataset and DataLite tells you what's "
    "interesting — before you ask a single question.</div>",
    unsafe_allow_html=True,
)
st.write("")

if df is None:
    st.info("📂 Upload a CSV or pick **Use sample data** in the sidebar to begin.")
    st.stop()


# --------------------------------------------------------------------------- #
# Chart rendering for a single finding
# --------------------------------------------------------------------------- #
def render_chart(finding, frame):
    spec = finding.chart or {}
    kind = spec.get("type")
    try:
        if kind == "metric":
            st.metric(spec["label"], spec["value"])
            return

        fig, ax = plt.subplots(figsize=(4.2, 2.5))

        if kind == "scatter":
            sub = frame[[spec["x"], spec["y"]]].dropna()
            ax.scatter(sub[spec["x"]], sub[spec["y"]], s=14, alpha=0.6,
                       color="#0e7490", edgecolor="none")
            ax.set_xlabel(spec["x"]); ax.set_ylabel(spec["y"])

        elif kind == "group_bar":
            means = spec["means"]
            ax.bar(list(means.keys()), list(means.values()), color="#4338ca")
            ax.set_ylabel(spec["value"]); ax.tick_params(axis="x", rotation=30)

        elif kind == "value_counts":
            counts = dict(sorted(spec["counts"].items(),
                                 key=lambda kv: kv[1], reverse=True)[:8])
            ax.bar(list(counts.keys()), list(counts.values()), color="#9333ea")
            ax.set_ylabel("count"); ax.tick_params(axis="x", rotation=30)

        elif kind == "box":
            sns.boxplot(y=frame[spec["col"]].dropna(), ax=ax, color="#fca5a5")
            ax.set_ylabel(spec["col"])

        elif kind == "missing_by_group":
            rates = {k: v * 100 for k, v in spec["rates"].items()}
            ax.bar(list(rates.keys()), list(rates.values()), color="#b45309")
            ax.set_ylabel("% missing"); ax.tick_params(axis="x", rotation=30)
        else:
            plt.close(fig); return

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        # a single chart failing should never break the report
        st.caption("_(chart unavailable)_")


def render_card(finding, frame):
    label, color = KIND_META.get(finding.kind, (finding.kind, "#475569"))
    with st.container(border=True):
        st.markdown(
            f"<span class='insight-badge' style='background:{color}'>{label}</span>"
            f"<div class='insight-headline'>{finding.headline}</div>"
            + (f"<div class='insight-detail'>{finding.detail}</div>"
               if finding.detail else ""),
            unsafe_allow_html=True,
        )
        render_chart(finding, frame)
        with st.expander("🔍 Verify"):
            st.caption("Computed directly from your full dataset — no AI involved.")
            st.json(finding.evidence)


# --------------------------------------------------------------------------- #
# HERO: auto-insights
# --------------------------------------------------------------------------- #
summary = ie.dataset_summary(df)
m = st.columns(5)
m[0].metric("Rows", f"{summary['rows']:,}")
m[1].metric("Columns", summary["cols"])
m[2].metric("Numeric", summary["numeric"])
m[3].metric("Missing", f"{summary['missing_pct']:.0f}%")
m[4].metric("Duplicate rows", summary["duplicates"])

st.markdown("### ✨ Auto-insights")

with st.spinner("Scanning your data…"):
    findings = ie.analyze(df, limit=8)

if not findings:
    st.success("No strong patterns jumped out — the data looks clean and "
               "fairly uniform. Explore it yourself below. 👇")
else:
    st.caption(f"{len(findings)} findings, most interesting first. "
               "Tap **Verify** on any card to see the exact numbers behind it.")
    cols = st.columns(2)
    for i, f in enumerate(findings):
        with cols[i % 2]:
            render_card(f, df)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Classic EDA (manual exploration)
# --------------------------------------------------------------------------- #
with st.expander("🧾 Explore the data yourself", expanded=False):
    st.markdown("#### Preview")
    st.dataframe(df.head(20), width='stretch')

    st.markdown("#### Summary statistics")
    st.dataframe(df.describe(include="all").T, width='stretch')

    st.markdown("#### Missing values")
    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing count"}),
        width='stretch',
    )

    st.markdown("#### Visualize a column")
    selected = st.selectbox("Choose a column", df.columns)
    fig, ax = plt.subplots(figsize=(6, 3))
    if pd.api.types.is_numeric_dtype(df[selected]) and not \
            pd.api.types.is_bool_dtype(df[selected]):
        sns.histplot(df[selected].dropna(), kde=True, ax=ax, color="#0e7490")
    else:
        order = df[selected].value_counts().index[:15]
        sns.countplot(y=df[selected].astype(str), order=[str(o) for o in order],
                      ax=ax, color="#4338ca")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Experimental: ask questions in plain English (optional, needs Groq key)
# --------------------------------------------------------------------------- #
with st.expander("💬 Ask a question (experimental)"):
    if not api_key:
        st.info("Add a Groq API key in the sidebar to try natural-language Q&A. "
                "Auto-insights above need no key.")
    else:
        st.caption("⚠️ For your safety, DataLite shows any code the model writes "
                   "but does **not** auto-run it. Sandboxed execution is planned.")
        q = st.text_area("Ask about your dataset", key="chat_input")
        if st.button("Ask"):
            prompt = (
                "You are a senior data analyst. Using the dataset preview below, "
                "answer the user's question concisely. If a chart helps, include "
                "a python code block using matplotlib/seaborn (variable `df`).\n\n"
                f"DATA (first 10 rows):\n{df.head(10).to_csv(index=False)}\n\n"
                f"QUESTION:\n{q}"
            )
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json={"model": "llama3-70b-8192",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0.3, "max_tokens": 800},
                    timeout=30,
                )
                if resp.status_code == 200:
                    answer = resp.json()["choices"][0]["message"]["content"]
                    code = None
                    match = re.search(r"```python(.*?)```", answer, re.DOTALL)
                    if match:
                        code = match.group(1).strip()
                        answer = answer.replace(match.group(0), "").strip()
                    st.markdown(answer)
                    if code:
                        st.caption("Suggested chart code (not executed):")
                        st.code(code, language="python")
                else:
                    st.error(f"Groq API error ({resp.status_code}).")
            except Exception as e:
                st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("Made with Streamlit · DataLite — auto-insight EDA")
