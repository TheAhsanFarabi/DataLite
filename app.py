import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="DataLite: Smart EDA with GROQ",
    layout="wide",
    page_icon="📊"
)

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("🔐 GROQ API Setup")
    api_key = st.text_input("Enter your GROQ API Key", type="password")
    st.markdown("---")
    st.info("Need a key? Get one from [GroqCloud](https://console.groq.com/)")

    st.title("📁 Data Source")
    data_source = st.radio("Choose data source:", ("Upload CSV", "Use Sample Data"))

# ========== HEADER ==========
st.markdown("## 📈 DataLite: Smart EDA with GROQ AI")
st.markdown("Upload your dataset or use a sample. Explore insights visually or ask questions in plain English!")

# ========== DATA LOADING ==========
df = None

if data_source == "Use Sample Data":
    try:
        df = pd.read_csv("data/sample.csv")
        st.success("✅ Loaded sample data successfully.")
    except FileNotFoundError:
        st.error("⚠️ Sample file not found. Please upload your own CSV file.")
else:
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

# ========== EDA AND VISUALIZATION ==========
if df is not None:
    st.markdown("### 🧾 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])

    with st.expander("📊 Summary Statistics", expanded=False):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with st.expander("❓ Missing Values", expanded=False):
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Missing Count", "index": "Column"}))

    # Column-based visualization
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
        sns.countplot(y=df[selected_col], ax=ax)
        st.pyplot(fig)

    # ========== GROQ AI ANALYSIS ==========
    st.markdown("---")
    st.markdown("## 🧠 Ask AI About Your Data")

    if not api_key:
        st.warning("🔐 Please enter your GROQ API Key in the sidebar to use AI features.")
    else:
        user_query = st.text_area("💬 Type your question (e.g. 'Top 5 frequent values in column X')")

        if user_query:
            prompt = f"""You are a data analyst. Based on this dataset (first 10 rows shown), analyze and respond to the user query below:
            
DATA (first 10 rows):
{df.head(10).to_csv(index=False)}

USER QUERY:
{user_query}"""

            with st.spinner("🔍 Asking GROQ AI..."):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-70b-8192",
                        "messages": [
                            {"role": "system", "content": "You are a helpful and expert data analysis assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 800
                    }
                )

                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                    st.success("✅ AI Response")
                    st.markdown(result)
                else:
                    st.error("❌ Failed to get response from GROQ API. Please check your key or query.")
else:
    st.info("📂 Please upload a CSV or use sample data to get started.")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + GROQ | [GitHub](https://github.com) | © 2025 DataLite")
