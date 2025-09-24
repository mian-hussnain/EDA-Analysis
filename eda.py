# eda_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("E-commerce Data EDA")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Dataset Info
    st.subheader("Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    st.write(df.describe())
    st.write(df.isnull().sum())

    # Univariate Analysis
    st.subheader("Univariate Analysis")
    col = st.selectbox("Select column for distribution plot", df.select_dtypes(include=['int64','float64']).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # Categorical Analysis
    st.subheader("Categorical Analysis")
    cat_col = st.selectbox("Select categorical column", df.select_dtypes(include=['object']).columns)
    fig, ax = plt.subplots()
    df[cat_col].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    fig, ax = plt.subplots()
    sns.boxplot(x="category", y="price", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Time-based Analysis
    st.subheader("Time-based Analysis")
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['date'] = df['order_date'].dt.date
    daily_sales = df.groupby('date')['price'].sum()
    fig, ax = plt.subplots()
    daily_sales.plot(ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
