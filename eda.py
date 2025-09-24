# eda_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Title
st.title("📊 E-commerce Data EDA with Cleaning")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Convert date column
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # -----------------------
    # Data Cleaning Section
    # -----------------------
    st.subheader("🧹 Data Cleaning")

    # 1. Remove duplicates
    before_dupes = df.shape[0]
    df = df.drop_duplicates()
    after_dupes = df.shape[0]
    st.write(f"✅ Removed {before_dupes - after_dupes} duplicate rows")

    # 2. Handle missing values
    st.write("### Missing Values Before Cleaning")
    st.write(df.isnull().sum())

    missing_option = st.radio(
        "How do you want to handle missing values?",
        ("Drop rows", "Fill with mean/median/mode", "Fill with constant (e.g., 0)")
    )

    if missing_option == "Drop rows":
        df = df.dropna()
    elif missing_option == "Fill with mean/median/mode":
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)  # numeric → median
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)  # categorical → mode
    elif missing_option == "Fill with constant (e.g., 0)":
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

    st.write("### Missing Values After Cleaning")
    st.write(df.isnull().sum())

    # 3. Remove invalid data
    invalid_rows = df[(df['price'] < 0) | (df['quantity'] <= 0)].shape[0]
    df = df[(df['price'] >= 0) & (df['quantity'] > 0)]
    st.write(f"✅ Removed {invalid_rows} invalid rows (negative price or zero quantity)")

    # -----------------------
    # Ready Dataset
    # -----------------------
    st.subheader("Cleaned Dataset Preview")
    st.write(df.head())

    # Define ID columns to exclude
    id_cols = ['order_id', 'customer_id', 'product_id']
    numeric_cols = [col for col in df.select_dtypes(include=['int64','float64']).columns if col not in id_cols]
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    st.write(f"Dataset Shape after cleaning: {df.shape}")

    # --------------------
    # Univariate Analysis
    # --------------------
    st.subheader("📈 Univariate Analysis (Numeric Variables)")
    col = st.selectbox("Select numeric column", numeric_cols)

    fig, ax = plt.subplots()
    if df[col].nunique() < 10:  # discrete
        sns.countplot(x=col, data=df, ax=ax)
    else:  # continuous
        sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # --------------------
    # Categorical Analysis
    # --------------------
    st.subheader("📊 Categorical Analysis")
    if categorical_cols:
        cat_col = st.selectbox("Select categorical column", categorical_cols)
        fig, ax = plt.subplots()
        df[cat_col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # --------------------
    # Bivariate: Category vs Price
    # --------------------
    st.subheader("📦 Bivariate Analysis (Category vs Price)")
    fig, ax = plt.subplots()
    sns.boxplot(x="category", y="price", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --------------------
    # Time Series Candlestick
    # --------------------
    st.subheader("⏳ Time-based Analysis (Candlestick Chart)")
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

    # --------------------
    # Correlation Heatmap
    # --------------------
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
