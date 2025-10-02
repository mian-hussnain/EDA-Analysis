# eda_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="E-commerce EDA", layout="wide")

# Global style (smaller fonts everywhere)
sns.set_context("paper", font_scale=0.7)

# Title
st.title("E-commerce Data EDA with Cleaning")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # -----------------------
    # Dashboard Layout
    # -----------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧹 Data Cleaning", 
        "📈 Univariate", 
        "📊 Categorical", 
        "📦 Bivariate", 
        "📉 Time & Correlation"
    ])

    # ========== Tab 1: Data Cleaning ==========
    with tab1:
        st.subheader("Data Cleaning")

        # Remove duplicates
        before_dupes = df.shape[0]
        df = df.drop_duplicates()
        after_dupes = df.shape[0]
        st.success(f"✅ Removed {before_dupes - after_dupes} duplicate rows")

        # Missing values
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Missing Values Before Cleaning")
            st.dataframe(df.isnull().sum())
        with col2:
            missing_option = st.radio(
                "Handle Missing Values:",
                ("Drop rows", "Fill with mean/median/mode", "Fill with constant (e.g., 0)")
            )

        if missing_option == "Drop rows":
            df = df.dropna()
        elif missing_option == "Fill with mean/median/mode":
            for col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        elif missing_option == "Fill with constant (e.g., 0)":
            for col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)

        st.write("### Missing Values After Cleaning")
        st.dataframe(df.isnull().sum())

        # Remove invalid
        invalid_rows = df[(df['price'] < 0) | (df['quantity'] <= 0)].shape[0]
        df = df[(df['price'] >= 0) & (df['quantity'] > 0)]
        st.success(f"✅ Removed {invalid_rows} invalid rows")

        st.write("### Cleaned Dataset Preview")
        st.dataframe(df.head())

        st.info(f"📐 Dataset Shape after cleaning: {df.shape}")

    # Identify cols
    id_cols = ['order_id', 'customer_id', 'product_id']
    numeric_cols = [col for col in df.select_dtypes(include=['int64','float64']).columns if col not in id_cols]
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # ========== Tab 2: Univariate ==========
    with tab2:
        st.subheader("Univariate Analysis (Numeric)")
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots(figsize=(4, 2))
            if df[col].nunique() < 10:
                sns.countplot(x=col, data=df, ax=ax)
            else:
                sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"{col} Distribution", fontsize=8)
            ax.set_xlabel(col, fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            ax.tick_params(axis='both', labelsize=7)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No numeric columns found!")

    # ========== Tab 3: Categorical ==========
    with tab3:
        st.subheader("Categorical Analysis")
        if categorical_cols:
            cat_col = st.selectbox("Select categorical column", categorical_cols)
            fig, ax = plt.subplots(figsize=(4, 2))
            df[cat_col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"{cat_col} Counts", fontsize=8)
            ax.set_xlabel(cat_col, fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            ax.tick_params(axis='both', labelsize=7)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No categorical columns found!")

    # ========== Tab 4: Bivariate ==========
    with tab4:
        st.subheader("Bivariate Analysis (Category vs Price)")
        if "category" in df.columns and "price" in df.columns:
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.boxplot(x="category", y="price", data=df, ax=ax)
            ax.set_title("Price Distribution by Category", fontsize=8)
            ax.set_xlabel("Category", fontsize=7)
            ax.set_ylabel("Price", fontsize=7)
            ax.tick_params(axis='both', labelsize=7)
            plt.xticks(rotation=45, fontsize=7)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("Category or Price column not found!")

    # ========== Tab 5: Time & Correlation ==========
    with tab5:
        # Time series
        st.subheader("Time-based Candlestick Chart")
        if "order_date" in df.columns and "price" in df.columns:
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
                title_font=dict(size=12),
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_title_font=dict(size=10),
                yaxis_title_font=dict(size=10),
                font=dict(size=9),
                xaxis_rangeslider_visible=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date or Price column missing!")

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        if numeric_cols:
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm",
                        annot_kws={"size":6}, ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No numeric columns for correlation heatmap!")
