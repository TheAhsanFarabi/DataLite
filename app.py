import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="DataLite: Smart EDA with GROQ",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Sidebar - API Key & Data Source
with st.sidebar:
    st.title("🔐 GROQ API Setup")
    api_key = st.text_input("Enter your GROQ API Key", type="password")
    st.markdown("---")
    st.info("Need a key? Get one from [GroqCloud](https://console.groq.com/)")

    st.title("📁 Data Source")
    data_source = st.radio("Choose data source:", ("Upload CSV", "Use Sample Data"))

# App Title
st.markdown("## 📈 DataLite: Smart EDA with GROQ AI")
st.markdown("Upload your dataset or use a sample. Explore insights visually or ask questions in plain English!")

# Built-in sample data (for cloud-safe use)
sample_csv = """
Name,Age,Gender,Score
Alice,23,Female,88
Bob,27,Male,72
Charlie,22,Male,95
Diana,24,Female,80
Evan,26,Male,91
Fay,25,Female,84
"""

# Load data
df = None
if data_source == "Use Sample Data":
    df = pd.read_csv(StringIO(sample_csv))
    st.success("✅ Sample data loaded successfully.")
else:
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

# Convert to Arrow-safe DataFrame
def make_arrow_safe(df):
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == 'object':
            safe_df[col] = safe_df[col].astype(str)
    return safe_df

# Display + Analyze Data
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

    # GROQ AI Query Section
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

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + GROQ | [GitHub](https://github.com) | © 2025 DataLite")
