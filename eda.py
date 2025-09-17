import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(page_title="EDA Web App", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA) App")
st.write("Upload your dataset (CSV) and explore it interactively!")

# ===================================================
# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # ===================================================
    # Dataset Overview
    st.header("🔍 Dataset Overview")
    st.write("**Shape of dataset:**", df.shape)
    st.write("**Columns:**", list(df.columns))

    with st.expander("Preview Data"):
        st.dataframe(df.head())

    with st.expander("Summary Statistics"):
        st.write(df.describe(include="all").T)

    with st.expander("Missing Values"):
        st.write(df.isnull().sum())

    with st.expander("Data Types"):
        st.write(df.dtypes)

    # ===================================================
    # Univariate Analysis
    st.header("📈 Univariate Analysis")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(num_cols) > 0:
        st.subheader("Numerical Features")
        col = st.selectbox("Select a numerical column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    if len(cat_cols) > 0:
        st.subheader("Categorical Features")
        col = st.selectbox("Select a categorical column", cat_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ===================================================
    # Bivariate Analysis
    st.header("🔗 Bivariate Analysis")
    if len(num_cols) > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Scatterplot
    if len(num_cols) > 1:
        st.subheader("Scatterplot")
        col_x = st.selectbox("X-axis", num_cols, key="scatter_x")
        col_y = st.selectbox("Y-axis", num_cols, key="scatter_y")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax)
        st.pyplot(fig)

    # ===================================================
    # Outlier Detection
    st.header("⚠ Outlier Detection (Boxplot)")
    if len(num_cols) > 0:
        col = st.selectbox("Select a column for boxplot", num_cols, key="boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    # ===================================================
    st.success("✅ EDA Completed! Dataset is ready for further analysis.")

else:
    st.info("👆 Please upload a CSV file to begin.")
