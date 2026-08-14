import pandas as pd

df_cleaned = pd.read_csv("data/cleaned/retail_cleaned.csv", parse_dates=["InvoiceDate"])

df_customers = pd.read_csv("data/cleaned/retail_customers.csv")

print("\n========== CORE BUSINESS KPI REPORT ==========")

print("\n── Revenue Report ─────────────────────────")

total_revenue = df_cleaned["Revenue"].sum()
print(f"Total Revenue: {total_revenue:,.2f}")

revenue_countries = (
    df_cleaned.groupby("Country")["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)

print("\nTop Revenue-Generating Countries:")
print(revenue_countries)

monthly_sales = (
    df_cleaned.groupby(
        df_cleaned["InvoiceDate"].dt.to_period("M")
    )["Revenue"]
    .sum()
    .reset_index()
)

print("\nMonthly Sales Trend:")
print(monthly_sales)

print("\n── Orders Report ──────────────────────────")

total_orders = df_cleaned["Invoice"].nunique()
average_order_value = total_revenue / total_orders

print(f"Total Orders: {total_orders:,}")
print(f"Average Order Value: {average_order_value:.2f}")

print("\n── Customer Report ────────────────────────")

total_customers = df_customers["CustomerID"].nunique()
print(f"Total Customers: {total_customers:,}")

top_customers = (
    df_customers.groupby("CustomerID")
    .size()
    .reset_index(name="Orders")
    .sort_values("Orders", ascending=False)
    .head(10)
)

print("\nTop Purchasing Customers:")
print(top_customers)

top_revenue_customers = (
    df_customers.groupby("CustomerID")["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)

print("\nTop Revenue-Generating Customers:")
print(top_revenue_customers)

print("\n── Product Report ─────────────────────────")

total_products = df_cleaned["StockCode"].nunique()
print(f"Total Products: {total_products:,}")

best_selling_products = (
    df_cleaned.groupby(["StockCode", "Description"])
    .size()
    .reset_index(name="Orders")
    .sort_values("Orders", ascending=False)
    .head(10)
)

print("\nBest-Selling Products:")
print(best_selling_products)

top_revenue_products = (
    df_cleaned.groupby(["StockCode", "Description"])["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)

print("\nTop Revenue-Generating Products:")
print(top_revenue_products)

'''
Key Findings

1. The business generates revenue from customers across multiple countries.
2. A relatively small group of customers contributes a large share of total revenue.
3. Best-selling products are not always the highest revenue-generating products.
4. Monthly sales trends indicate seasonality, making the dataset suitable for demand forecasting.
'''