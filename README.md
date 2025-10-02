# E-commerce EDA with Data Cleaning  

This is a **Streamlit-based web application** for performing **Exploratory Data Analysis (EDA)** and **basic data cleaning** on e-commerce datasets.  
The app provides interactive dashboards for univariate, categorical, bivariate, time-series, and correlation analysis.  

---

## Features  

- **Data Cleaning**
  - Remove duplicates  
  - Handle missing values (drop, fill with mean/median/mode, or constant)  
  - Remove invalid rows (negative prices, zero/negative quantities)  

- **Univariate Analysis**
  - Histograms & KDE plots for numeric columns  
  - Count plots for small-cardinality numeric features  

- **Categorical Analysis**
  - Bar plots of categorical variable counts  

- **Bivariate Analysis**
  - Boxplot of `price` distribution across categories  

- **Time & Correlation**
  - Interactive **candlestick chart** of daily price movement (OHLC)  
  - Correlation heatmap of numeric variables

## 📦 Installation  

1. Clone this repository or copy the code:  

[git clone https://github.com/yourusername/ecommerce-eda-app.git
   cd ecommerce-eda-app](https://github.com/mian-hussnain/EDA-Analysis/tree/main)
   ## Requirements

## Requirements

Create a file named requirements.txt in the same folder as eda_app.py and add the following:
streamlit
pandas
matplotlib
seaborn
plotly

## Dashboard Tabs

🧹 Data Cleaning – Remove duplicates, handle missing values, and filter invalid rows

📈 Univariate – Histograms and count plots of numeric variables

📊 Categorical – Frequency distribution of categorical variables

📦 Bivariate – Price distribution across product categories

📉 Time & Correlation – Candlestick OHLC chart + correlation heatmap

## Author
Muhammad Hussnain Mamoon

