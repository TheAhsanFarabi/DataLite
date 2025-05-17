import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Sidebar for GROQ API Key
st.sidebar.title("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# App Title
st.title("DataLite: Simplified Data Mining Tool (GROQ Enhanced)")
st.write("Upload a CSV file to explore and analyze your data, or use the sample data. Ask questions and let AI do the rest!")

# Option to use sample data or upload a CSV
use_sample_data = st.radio("Choose data source:", ('Upload CSV', 'Use Sample Data'))

df = None

if use_sample_data == 'Use Sample Data':
    sample_file = 'data/sample.csv'
    try:
        df = pd.read_csv(sample_file)
        st.success("Loaded sample data.")
        st.write(df.head())
    except FileNotFoundError:
        st.error(f"Sample file '{sample_file}' not found. Please upload a file instead.")
else:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded file successfully.")
        st.write(df.head())

if df is not None:
    # Data Summary
    st.subheader("Quick Data Summary")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("Basic Statistics:")
    st.write(df.describe(include='all'))

    # Missing Values
    st.subheader("Missing Value Count")
    st.write(df.isnull().sum())

    # Visualization
    st.subheader("Visualization")
    column = st.selectbox("Choose a column to visualize", df.columns)

    if df[column].dtype in ['int64', 'float64']:
        st.write("### Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("### Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(y=df[column], ax=ax)
        st.pyplot(fig)
    else:
        st.write("### Bar Chart")
        fig, ax = plt.subplots()
        sns.countplot(y=df[column], ax=ax)
        st.pyplot(fig)

    # GROQ AI Query
    st.subheader("💡 Ask AI About Your Data (Powered by GROQ)")
    user_query = st.text_area("Example: 'What are the top 5 most frequent values in the column X?'")

    if not api_key:
        st.warning("Please enter your GROQ API key in the sidebar to use AI features.")
    elif user_query:
        prompt = f"""You are a data analyst. Based on the preview of this dataset, answer the following user query with insight and stats:
Dataset:
{df.head(10).to_csv(index=False)}

User Query:
{user_query}"""

        with st.spinner("Processing your query with GROQ..."):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a helpful data science assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }
            )

            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                st.markdown("### 🧠 AI Analysis Result:")
                st.write(result)
            else:
                st.error("Failed to get a response from GROQ API. Check your key or input.")
else:
    st.info("Please upload a CSV or use the sample data to begin analysis.")

st.markdown("---")
