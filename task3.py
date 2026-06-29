import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("visuals", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset.csv")

print("===== DATASET OVERVIEW =====")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# -----------------------------
# 1. Gender-wise Spending
# -----------------------------
gender_spending = df.groupby("Gender")["Total Amount"].sum()
print("\nGender-wise Spending:")
print(gender_spending)

plt.figure(figsize=(6,4))
gender_spending.plot(kind='bar')
plt.title("Gender-wise Total Spending")
plt.ylabel("Total Amount")
plt.savefig("visuals/gender_spending.png")
plt.show()

# -----------------------------
# 2. Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("visuals/age_distribution.png")
plt.show()

# -----------------------------
# 3. Product Category Sales
# -----------------------------
category_sales = df.groupby("Product Category")["Total Amount"].sum()
print("\nCategory Sales:")
print(category_sales)

plt.figure(figsize=(8,5))
category_sales.plot(kind='bar')
plt.title("Sales by Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.savefig("visuals/category_sales.png")
plt.show()

# -----------------------------
# 4. Quantity Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Quantity"])
plt.title("Quantity Distribution")
plt.savefig("visuals/quantity_boxplot.png")
plt.show()

# -----------------------------
# 5. Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(6,4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("visuals/heatmap.png")
plt.show()

# -----------------------------
# Insights
# -----------------------------
top_category = category_sales.idxmax()
print("\n===== INSIGHTS =====")
print(f"Highest selling category: {top_category}")

top_gender = gender_spending.idxmax()
print(f"Highest spending gender: {top_gender}")

print("\nTask 3 Completed Successfully!")