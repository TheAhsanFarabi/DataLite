import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
from io import StringIO

# ========== Page Config ==========
st.set_page_config(
    page_title="DataLite: Smart EDA with GROQ",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ========== Sidebar: API and Dataset ==========
with st.sidebar:
    st.title("🔐 GROQ API Setup")
    api_key = st.text_input("Enter your GROQ API Key", type="password")
    st.markdown("---")
    st.info("Need a key? Get one from [GroqCloud](https://console.groq.com/)")

    st.title("📁 Data Source")
    data_source = st.radio("Choose data source:", ("Upload CSV", "Use Sample Data"))

# ========== Title ==========
st.markdown("## 📈 DataLite: Smart EDA with GROQ AI")
st.markdown("Upload your dataset or use a sample. Explore insights visually or ask questions in plain English!")

# ========== Sample Data ==========
sample_csv = """
Name,Age,Gender,Score
Alice,23,Female,88
Bob,27,Male,72
Charlie,22,Male,95
Diana,24,Female,80
Evan,26,Male,91
Fay,25,Female,84
"""

df = None
if data_source == "Use Sample Data":
    df = pd.read_csv(StringIO(sample_csv))
    st.success("✅ Sample data loaded successfully.")
else:
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

# ========== Arrow-Safe Conversion ==========
def make_arrow_safe(df):
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == 'object':
            safe_df[col] = safe_df[col].astype(str)
    return safe_df

# ========== Initialize Session Chat Memory ==========
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ========== EDA + Chat ==========
if df is not None:
    safe_df = make_arrow_safe(df)

    st.markdown("### 🧾 Data Preview")
    st.dataframe(safe_df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    with st.expander("📊 Summary Statistics", expanded=False):
        st.dataframe(make_arrow_safe(df.describe(include="all").T), use_container_width=True)

    with st.expander("❓ Missing Values", expanded=False):
        st.dataframe(
            pd.DataFrame(df.isnull().sum()).reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
            use_container_width=True
        )

    # ========== Column Visualization ==========
    st.markdown("### 📌 Visualize a Column")
    selected_col = st.selectbox("Choose a column to visualize", df.columns)

    if df[selected_col].dtype in ["int64", "float64"]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Histogram")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)
        with col2:
            st.markdown("#### Box Plot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[selected_col], ax=ax)
            st.pyplot(fig)
    else:
        st.markdown("#### Count Plot")
        fig, ax = plt.subplots()
        sns.countplot(y=make_arrow_safe(df[selected_col]), ax=ax)
        st.pyplot(fig)

    # ========== GROQ AI Chat ==========
    st.markdown("---")
    st.markdown("## 💬 Chat with GROQ AI")

    if not api_key:
        st.warning("🔐 Please enter your GROQ API Key in the sidebar to use AI features.")
    else:
        user_input = st.text_area("Ask a question about your dataset (GROQ may return charts too)", key="chat_input")

        if st.button("Ask"):
            prompt = f"""
You are a senior data analyst. Respond to this user query using the following dataset preview.

DATA (first 10 rows):
{df.head(10).to_csv(index=False)}

QUERY:
{user_input}

If appropriate, include a Python code block to generate a chart using matplotlib or seaborn.
"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 800
                }
            )

            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]

                # Store in chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("ai", result))

                st.success("✅ Answer received!")
            else:
                st.error("❌ Failed to get response from GROQ API.")

        # Display chat
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").markdown(msg)
            else:
                st.chat_message("assistant").markdown(msg)

                # Execute Python code if present
                code_match = re.search(r"```python(.*?)```", msg, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    st.markdown("🔧 Executing chart code:")
                    st.code(code, language="python")
                    try:
                        exec(code, {'df': df, 'plt': plt, 'sns': sns})
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception as e:
                        st.error(f"⚠️ Error running code: {e}")
else:
    st.info("📂 Please upload a CSV or use sample data to begin.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + GROQ | © 2025 DataLite")
