import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("DataLite: Simplified Data Mining Tool")
st.write("Upload a CSV file to explore and analyze your data easily.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read and display the uploaded file
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



# Footer
st.markdown("---")
