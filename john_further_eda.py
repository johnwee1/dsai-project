# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime

# THIS DATASET IS SYNTHETICALLY GENERATED! Therefore some relationships may not make sense.
# The dataset is a simplified version of a retail dataset.


# %%
df = pd.read_parquet("processed.parquet")
df.columns.to_list()
# columns: ['Customer ID',
#  'City',
#  'Country',
#  'Gender', # either M F or D (diverse)
#  'Age',
#  'Invoice ID',
#  'Unit Price',
#  'Quantity',
#  'Date', # it is a timestamp, i.e. 2024-02-03 20:48:00.
#  'Discount', # in decimal. 0 means no discount.
#  'Line Total', # Quantity * Unit Price * (1 - Discount)
#  'Currency'] # either USD, EUR, GBP, CNY

# %%
# Convert Line Total to USD
conversion_rates = {"GBP": 1.27, "CNY": 0.1389, "EUR": 1.084, "USD": 1.0}

df["Line Total"] = df.apply(
    lambda row: row["Line Total"] * conversion_rates[row["Currency"]], axis=1
)

# %%
# Remove Gender = D
df = df[df["Gender"] != "D"]

# Load and explore data
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# %%
# Customer Segmentation
# Segment by age groups
age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
age_labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df["Age Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)

# Segment by gender
gender_groups = df.groupby("Gender").agg(
    {"Line Total": "sum", "Customer ID": "nunique"}
)

# Segment by geography
geo_groups = df.groupby(["Country", "City"]).agg(
    {"Line Total": "sum", "Customer ID": "nunique"}
)

# %%
# Spending Pattern Analysis
# Analyze spending by age group
age_spending = df.groupby("Age Group").agg(
    {"Line Total": "sum", "Customer ID": "nunique"}
)

# Analyze spending by gender
gender_spending = df.groupby("Gender").agg(
    {"Line Total": "sum", "Customer ID": "nunique"}
)

# Analyze spending by geography
geo_spending = df.groupby(["Country", "City"]).agg(
    {"Line Total": "sum", "Customer ID": "nunique"}
)

# %%
# Identify High-Value Customers
# Define high-value customers as those in the top 10% of spending
high_value_threshold = df["Line Total"].quantile(0.9)
high_value_customers = df[df["Line Total"] > high_value_threshold]

# Identify one-time purchasers
one_time_purchasers = df[df.duplicated("Customer ID", keep=False) == False]

# %%
# Visualization
plt.figure(figsize=(10, 6))
sb.barplot(x="Age Group", y="Line Total", data=age_spending.reset_index())
plt.title("Spending by Age Group")
plt.show()

plt.figure(figsize=(10, 6))
sb.barplot(x="Gender", y="Line Total", data=gender_spending.reset_index())
plt.title("Spending by Gender")
plt.show()

plt.figure(figsize=(10, 6))
sb.barplot(x="Country", y="Line Total", data=geo_spending.reset_index())
plt.title("Spending by Geography")
plt.show()

# %%
# Calculate average purchase value by age group and gender
avg_purchase_by_age = df.groupby("Age Group")["Line Total"].mean().reset_index()
avg_purchase_by_gender = df.groupby("Gender")["Line Total"].mean().reset_index()

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.barplot(x="Age Group", y="Line Total", data=avg_purchase_by_age)
plt.title("Average Purchase Value by Age Group")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sb.barplot(x="Gender", y="Line Total", data=avg_purchase_by_gender)
plt.title("Average Purchase Value by Gender")
plt.tight_layout()
plt.show()

# %%
# Analyze purchase frequency per customer
purchase_frequency = df.groupby("Customer ID").size().reset_index(name="Purchase Count")

# Distribution of purchase frequency
plt.figure(figsize=(10, 6))
sb.histplot(purchase_frequency["Purchase Count"], kde=True, bins=30)
plt.title("Distribution of Purchase Frequency")
plt.xlabel("Number of Purchases")
plt.show()

# Categorize customers by purchase frequency
purchase_frequency["Frequency Category"] = pd.cut(
    purchase_frequency["Purchase Count"],
    bins=[0, 1, 3, 10, float("inf")],
    labels=["One-time", "Occasional", "Regular", "Frequent"],
)

# Count customers by frequency category
frequency_counts = purchase_frequency["Frequency Category"].value_counts().sort_index()
plt.figure(figsize=(10, 6))
frequency_counts.plot(kind="bar")
plt.title("Customer Distribution by Purchase Frequency")
plt.ylabel("Number of Customers")
plt.show()

# %%
# Analyze impact of discounts on purchase behavior
discount_groups = pd.cut(
    df["Discount"],
    bins=[0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
    labels=["No Discount", "<10%", "<20%", "<30%", "<40%", "<50%", "<60%", "<100%"],
    right=False,
)
df["Discount Group"] = discount_groups

df[df["Discount"] == 0.2]  ## evaluates to 0. wtf

# Average purchase amount by discount group
discount_analysis = (
    df.groupby("Discount Group")
    .agg({"Line Total": "mean", "Quantity": "mean", "Customer ID": "nunique"})
    .reset_index()
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.barplot(x="Discount Group", y="Line Total", data=discount_analysis)
plt.title("Average Purchase Amount by Discount Level")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sb.barplot(x="Discount Group", y="Quantity", data=discount_analysis)
plt.title("Average Purchase Quantity by Discount Level")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
