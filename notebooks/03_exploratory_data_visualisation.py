import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_cleaned = pd.read_csv("data/cleaned/retail_cleaned.csv", parse_dates=["InvoiceDate"])

df_customers = pd.read_csv("data/cleaned/retail_customers.csv")

sns.set_theme(style="whitegrid")

print("\n── Revenue Trend Visualization ─────────────────────────")

monthly_sales = (
    df_cleaned.groupby(
        df_cleaned["InvoiceDate"].dt.to_period("M")
    )["Revenue"]
    .sum()
    .reset_index()
)

monthly_sales["InvoiceDate"] = (monthly_sales["InvoiceDate"].dt.to_timestamp())

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["InvoiceDate"],
    monthly_sales["Revenue"],
    linewidth=2
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig("assets/monthly_revenue_trend.png")

plt.show()
plt.close()

print("\n── Product Performance Visualization ───────────────────")

top_products = (
    df_cleaned.groupby("Description")["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 7))

ax = sns.barplot(
    data=top_products,
    x="Revenue",
    y="Description"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

plt.title("Top 10 Revenue-Generating Products")
plt.xlabel("Revenue")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig("assets/top_revenue_products.png")

plt.show()
plt.close()

print("\n── Customer Performance Visualization ──────────────────")

top_customers = (
    df_customers.groupby("CustomerID")["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 7))

ax = sns.barplot(
    data=top_customers,
    x="CustomerID",
    y="Revenue"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

plt.title("Top 10 Revenue-Generating Customers")
plt.xlabel("Customer ID")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig("assets/top_revenue_customers.png")

plt.show()
plt.close()

print("\n── Geographic Performance Visualization ────────────────")

top_countries = (
    df_cleaned.groupby("Country")["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 7))

ax = sns.barplot(
    data=top_countries,
    x="Revenue",
    y="Country"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

plt.title("Top 10 Revenue-Generating Countries")
plt.xlabel("Revenue")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig("assets/top_revenue_countries.png")

plt.show()
plt.close()

print("\n── Revenue Distribution Visualization ──────────────────")

revenue_percentiles = (
    df_cleaned["Revenue"]
    .quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
)

plt.figure(figsize=(10, 5))

plt.plot(
    [25, 50, 75, 90, 95, 99],
    revenue_percentiles.values,
    marker="o"
)

plt.title("Revenue Distribution by Percentile")
plt.xlabel("Percentile")
plt.ylabel("Revenue")

plt.grid(True)

plt.tight_layout()

plt.savefig("assets/revenue_distribution.png")

plt.show()
plt.close()

'''
Key Insights

1. Revenue follows a clear monthly trend suitable for demand forecasting.
2. A small number of products generate a significant share of total revenue.
3. High-value customers contribute disproportionately to business revenue.
4. Revenue is concentrated in a few countries.
5. Revenue distribution is highly right-skewed, indicating the presence of high-value transactions.
'''