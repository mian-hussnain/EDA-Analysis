# eda_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Title
st.title("E-commerce Data EDA")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Convert date
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Drop IDs from numeric analysis
    id_cols = ['order_id', 'customer_id', 'product_id']
    numeric_cols = [col for col in df.select_dtypes(include=['int64','float64']).columns if col not in id_cols]

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write(df.describe(include="all"))
    st.write("Missing values:", df.isnull().sum())

    # Univariate Analysis
    st.subheader("Univariate Analysis (Numeric Variables)")
    col = st.selectbox("Select numeric column", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # Categorical Analysis
    st.subheader("Categorical Analysis")
    cat_col = st.selectbox("Select categorical column", df.select_dtypes(include=['object']).columns)
    fig, ax = plt.subplots()
    df[cat_col].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # Bivariate: Category vs Price
    st.subheader("Bivariate Analysis (Category vs Price)")
    fig, ax = plt.subplots()
    sns.boxplot(x="category", y="price", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Time Series Candlestick Chart
    st.subheader("Time-based Analysis (Candlestick Chart)")
    df['date'] = df['order_date'].dt.date
    daily = df.groupby('date').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last')
    ).reset_index()

    fig = go.Figure(data=[go.Candlestick(
        x=daily['date'],
        open=daily['open'],
        high=daily['high'],
        low=daily['low'],
        close=daily['close']
    )])

    fig.update_layout(
        title="Daily Price Movement (OHLC)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
