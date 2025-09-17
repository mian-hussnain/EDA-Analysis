# ===================================================
# Professional Streamlit EDA App
# Author: Your Name
# ===================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Config
st.set_page_config(page_title="EDA Web App", layout="wide")

# Title
st.title("📊 Professional Exploratory Data Analysis (EDA) App")
st.markdown("Upload your dataset (CSV) and perform professional EDA interactively.")

# ===================================================
# Sidebar
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ===================================================
# Load Dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📌 Overview", "🧹 Data Cleaning", "📈 Univariate", "🔗 Bivariate", "⚠ Outliers"]
    )

    # ===================================================
    # Overview
    with tab1:
        st.header("📌 Dataset Overview")
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
    # Data Cleaning
    with tab2:
        st.header("🧹 Data Cleaning")

        # Missing Values
        missing = df.isnull().sum()
        if missing.any():
            st.warning("⚠ Missing values detected.")
            option = st.radio("Handle missing values:",
                              ("Do nothing", "Drop rows", "Fill with mean/median/mode"))

            if option == "Drop rows":
                df = df.dropna()
                st.success("✅ Missing values dropped.")
            elif option == "Fill with mean/median/mode":
                for col in df.columns:
                    if df[col].dtype in ["int64", "float64"]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                st.success("✅ Missing values imputed.")
        else:
            st.info("No missing values detected.")

        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠ Found {dup_count} duplicate rows.")
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.success("✅ Duplicates removed.")
        else:
            st.info("No duplicate rows detected.")

        # Download cleaned data
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned Dataset", data=csv, file_name="cleaned_data.csv", mime="text/csv")

    # ===================================================
    # Univariate Analysis
    with tab3:
        st.header("📈 Univariate Analysis")
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        # Numerical
        if len(num_cols) > 0:
            st.subheader("Numerical Features")
            col = st.selectbox("Select a numerical column", num_cols, key="num_univariate")
            fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        # Categorical
        if len(cat_cols) > 0:
            st.subheader("Categorical Features")
            col = st.selectbox("Select a categorical column", cat_cols, key="cat_univariate")
            fig = px.bar(df[col].value_counts().reset_index(),
                         x="index", y=col,
                         title=f"Count Plot of {col}",
                         labels={"index": col, col: "Count"})
            st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # Bivariate Analysis
    with tab4:
        st.header("🔗 Bivariate Analysis")

        if len(num_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

            st.subheader("Scatterplot")
            col_x = st.selectbox("X-axis", num_cols, key="scatter_x")
            col_y = st.selectbox("Y-axis", num_cols, key="scatter_y")
            fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_x} vs {col_y}")
            st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # Outlier Detection
    with tab5:
        st.header("⚠ Outlier Detection (Boxplot)")
        if len(num_cols) > 0:
            col = st.selectbox("Select a column for boxplot", num_cols, key="boxplot")
            fig = px.box(df, y=col, title=f"Boxplot of {col}")
            st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    st.success("✅ Professional EDA Completed! Dataset is clean and insights extracted.")

else:
    st.info("👆 Please upload a CSV file to begin.")
