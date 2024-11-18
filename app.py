import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("DataLite: Simplified Data Mining Tool")
st.write("Upload a CSV file to explore and analyze your data easily, or use the sample data below.")

# Option to use sample data or upload a CSV
use_sample_data = st.radio("Choose data source:", ('Upload CSV', 'Use Sample Data'))

if use_sample_data == 'Use Sample Data':
    # Load sample data from the 'data/sample.csv' file
    sample_file = 'data/sample.csv'
    df = pd.read_csv(sample_file)
    st.write("Using the sample data:")
    st.write(df.head())
else:
    # File Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.write(df.head())

# Data Summary
st.subheader("Data Summary")
st.write("Shape of the data:")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.write("Summary Statistics:")
st.write(df.describe())

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Visualization
st.subheader("Visualization")
column = st.selectbox("Choose a column to visualize", df.columns)

# Determine column type and provide three graphs
if df[column].dtype in ['int64', 'float64']:
    # Histogram
    st.write("### Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)

    # Box Plot
    st.write("### Box Plot")
    fig, ax = plt.subplots()
    sns.boxplot(y=df[column], ax=ax)
    st.pyplot(fig)
else:
    # Bar Chart
    st.write("### Bar Chart")
    fig, ax = plt.subplots()
    sns.countplot(y=df[column], ax=ax)
    st.pyplot(fig)

# Correlation Matrix
st.subheader("Correlation Matrix")
if df.select_dtypes(include=['int64', 'float64']).shape[1] > 1:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.write("Not enough numeric data for a correlation matrix.")

# Footer
st.markdown("---")
