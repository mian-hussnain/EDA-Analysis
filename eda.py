import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Title
st.title("📊 Exploratory Data Analysis (EDA) App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show first rows
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Tabs for analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dataset Info", "Univariate Analysis", "Categorical Analysis", 
        "Bivariate Analysis", "Correlation Heatmap"
    ])

    # Dataset Info
    with tab1:
        st.subheader("Dataset Information")
        st.write("**Shape of dataset:**", df.shape)
        st.write("**Columns:**", df.columns.tolist())
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

    # Univariate Analysis
    with tab2:
        st.subheader("📊 Univariate Analysis")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column:", numeric_cols)
            fig, ax = plt.subplots(figsize=(5, 3))  # smaller graph
            sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found.")

    # Categorical Analysis
    with tab3:
        st.subheader("🟦 Categorical Analysis")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            col = st.selectbox("Select categorical column:", categorical_cols)
            fig, ax = plt.subplots(figsize=(5, 3))  # smaller graph
            df[col].value_counts().plot(kind='bar', ax=ax, color="lightgreen")
            ax.set_title(f"Count Plot of {col}")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns found.")

    # Bivariate Analysis
    with tab4:
        st.subheader("🔗 Bivariate Analysis")
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Select X-axis column:", numeric_cols, index=0)
            col2 = st.selectbox("Select Y-axis column:", numeric_cols, index=1)
            fig, ax = plt.subplots(figsize=(6, 4))  # smaller graph
            sns.scatterplot(x=df[col1], y=df[col2], ax=ax, color="coral")
            ax.set_title(f"Scatter Plot: {col1} vs {col2}")
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for bivariate analysis.")

    # Correlation Heatmap
    with tab5:
        st.subheader("🔥 Correlation Heatmap")
        if numeric_cols:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 4))  # smaller graph
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found.")
