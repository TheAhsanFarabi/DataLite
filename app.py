"""
DataLite — Auto-Insight EDA  ("data newspaper" edition)
=======================================================

Drop in a dataset and DataLite writes its front page: a lead story for the
strongest finding, then a grid of smaller stories — each one computed directly
from your full data (no LLM, no API key) and verifiable on the spot.

Presentation lives in render.py; the analysis engine in insight_engine.py.
"""

import re
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

import insight_engine as ie
import render

st.set_page_config(
    page_title="DataLite — Auto-Insight EDA",
    layout="wide",
    page_icon="✦",
    initial_sidebar_state="expanded",
)

st.markdown(render.CSS, unsafe_allow_html=True)

SAMPLE_CSV = """Name,Age,Gender,Score,Passed
Alice,23,Female,85,Yes
Bob,25,Male,75,Yes
Charlie,22,Male,50,No
Diana,24,Female,95,Yes
Edward,21,Male,45,No
Fay,26,Female,88,Yes"""

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("Data source")
    data_source = st.radio("Source", ("Use sample data", "Upload CSV"),
                           label_visibility="collapsed")
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

    st.markdown("---")
    with st.expander("Optional · AI narration (Groq)"):
        st.caption("Insights work fully without this. A key only unlocks the "
                   "experimental chat — it's never required.")
        api_key = st.text_input("Groq API key", type="password",
                                label_visibility="collapsed",
                                placeholder="gsk_… (optional)")
        st.caption("Get one from [GroqCloud](https://console.groq.com/).")

# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
df = None
if data_source == "Use sample data":
    df = pd.read_csv(StringIO(SAMPLE_CSV))
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read that CSV: {e}")

if df is None:
    st.markdown(
        "<div class='masthead'><div class='edition'>Auto-Insight Edition</div>"
        "<div class='wordmark'><span class='spark'>✦</span> DataLite</div>"
        "<div class='dateline'>Your data — read all about it</div></div>"
        "<div class='rule'></div><div class='rule thin'></div>"
        "<div class='quiet'>Upload a CSV or pick “Use sample data” in the "
        "sidebar to print today's edition.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Front page: masthead + chips + insight cards (one HTML block)
# --------------------------------------------------------------------------- #
summary = ie.dataset_summary(df)
with st.spinner("Reading your data…"):
    findings = ie.analyze(df, limit=8)

try:
    date_str = datetime.now().strftime("%A, %B %-d, %Y")
except ValueError:  # some platforms lack the %-d directive
    date_str = datetime.now().strftime("%A, %B %d, %Y")

st.markdown(
    render.masthead_html(summary, len(findings), date_str)
    + render.chips_html(summary)
    + render.insights_html(findings, df),
    unsafe_allow_html=True,
)

st.write("")

# --------------------------------------------------------------------------- #
# Classic EDA (manual exploration)
# --------------------------------------------------------------------------- #
with st.expander("Explore the data yourself"):
    st.markdown("**Preview**")
    st.dataframe(df.head(20), width="stretch")

    st.markdown("**Summary statistics**")
    st.dataframe(df.describe(include="all").T, width="stretch")

    st.markdown("**Missing values**")
    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing count"}),
        width="stretch",
    )

    st.markdown("**Visualize a column**")
    sns.set_theme(style="whitegrid")
    selected = st.selectbox("Column", df.columns, label_visibility="collapsed")
    fig, ax = plt.subplots(figsize=(6, 3))
    if pd.api.types.is_numeric_dtype(df[selected]) and not \
            pd.api.types.is_bool_dtype(df[selected]):
        sns.histplot(df[selected].dropna(), kde=True, ax=ax, color=render.ACCENT)
    else:
        order = [str(o) for o in df[selected].value_counts().index[:15]]
        sns.countplot(y=df[selected].astype(str), order=order, ax=ax,
                      color="#2C5478")
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Experimental chat (optional Groq; code shown, never auto-run)
# --------------------------------------------------------------------------- #
with st.expander("Ask a question · experimental"):
    if not api_key:
        st.info("Add a Groq API key in the sidebar to try natural-language Q&A. "
                "The front-page insights need no key.")
    else:
        st.caption("⚠️ For safety, DataLite shows any code the model writes but "
                   "does not auto-run it. Sandboxed execution is planned.")
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

st.markdown("<div class='rule thin' style='margin-top:32px'></div>"
            "<div class='dateline' style='text-align:center'>DataLite · "
            "auto-insight EDA · every number verifiable</div>",
            unsafe_allow_html=True)
