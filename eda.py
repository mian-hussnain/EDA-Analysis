

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots pretty
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ================================================
# 2. Load Dataset
# Replace 'your_dataset.csv' with your file
df = pd.read_csv("your_dataset.csv")

# ================================================
# 3. Basic Info
print("\n--- Dataset Overview ---")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nSummary Statistics:\n", df.describe(include="all").T)

# ================================================
# 4. Univariate Analysis

# Numerical Features
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Categorical Features
cat_cols = df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    plt.figure()
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# ================================================
# 5. Bivariate Analysis

# Correlation Heatmap for Numerical Features
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Scatterplots for Numerical Relationships
if len(num_cols) > 1:
    sns.pairplot(df[num_cols])
    plt.show()

# ================================================
# 6. Outlier Detection (Boxplots)
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# ================================================
# 7. Insights & Next Steps
print("\n--- Key EDA Checks Completed ---")
print("✔ Missing values checked")
print("✔ Duplicate rows checked")
print("✔ Univariate analysis done")
print("✔ Bivariate correlations explored")
print("✔ Outliers visualized")
print("Dataset is now ready for feature engineering / modeling.")
