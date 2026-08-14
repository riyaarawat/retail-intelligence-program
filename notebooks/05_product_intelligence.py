import pandas as pd
import matplotlib.pyplot as plt

df_products = pd.read_csv(
    "data/cleaned/retail_cleaned.csv"
)

print("\n========== PRODUCT INTELLIGENCE REPORT ==========")

print("\n── Product Revenue Analysis ─────────────────────")

product_revenue = (
    df_products.groupby(
        ["StockCode", "Description"]
    )["Revenue"]
    .sum()
    .reset_index()
)

print(product_revenue.head())

print("\nRevenue Statistics\n")
print(product_revenue["Revenue"].describe())

top_revenue_products = (
    product_revenue
    .sort_values("Revenue", ascending=False)
    .head(10)
)

print("\nTop 10 Revenue-Generating Products\n")
print(top_revenue_products)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    top_revenue_products["Description"],
    top_revenue_products["Revenue"]
)

plt.bar_label(
    bars,
    fmt="%.0f",
    padding=3
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "assets/top_products_revenue.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

# ----------------------------------------------------------
# Insight
#
# A relatively small group of products contributes
# a disproportionately large share of total revenue.
# ----------------------------------------------------------

# ==========================================================
# Product Sales Volume Analysis
# ==========================================================

print("\n── Product Sales Volume Analysis ─────────────────")

product_quantity = (
    df_products.groupby(
        ["StockCode", "Description"]
    )["Quantity"]
    .sum()
    .reset_index(name="UnitsSold")
)

print(product_quantity.head())

print("\nSales Volume Statistics\n")
print(product_quantity["UnitsSold"].describe())

top_quantity_products = (
    product_quantity
    .sort_values("UnitsSold", ascending=False)
    .head(10)
)

print("\nTop 10 Products by Units Sold\n")
print(top_quantity_products)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    top_quantity_products["Description"],
    top_quantity_products["UnitsSold"]
)

plt.bar_label(
    bars,
    fmt="%.0f",
    padding=3
)

plt.title("Top 10 Products by Units Sold")
plt.xlabel("Product")
plt.ylabel("Units Sold")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "assets/top_products_quantity.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

# ----------------------------------------------------------
# Insight
#
# Products with the highest sales volume are not always
# the products generating the highest revenue.
# ----------------------------------------------------------

# ==========================================================
# Product Performance Summary
# ==========================================================

print("\n── Product Performance Summary ───────────────────")

product_summary = (
    product_revenue.merge(
        product_quantity,
        on=["StockCode", "Description"]
    )
    .sort_values("Revenue", ascending=False)
    .reset_index(drop=True)
)

print(product_summary.head(10))

print("\nProduct Summary Statistics\n")
print(product_summary.describe())

product_summary.to_csv(
    "data/processed/product_summary.csv",
    index=False
)

print("\nProduct summary saved successfully.")

# ==========================================================
# Product Revenue Distribution
# ==========================================================

print("\n── Product Revenue Distribution ─────────────────")

print(product_summary["Revenue"].describe())

plt.figure(figsize=(8, 5))

plt.hist(
    product_summary["Revenue"],
    bins=40
)

plt.yscale("log")

plt.title("Product Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Number of Products (Log Scale)")

plt.tight_layout()

plt.savefig(
    "assets/product_revenue_distribution.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

# ----------------------------------------------------------
# Insight
#
# Product revenue is highly right-skewed.
# Most products contribute relatively little revenue,
# while a small number generate exceptionally high sales.
# ----------------------------------------------------------

# ==========================================================
# Pareto Analysis (80/20 Rule)
# ==========================================================

print("\n── Pareto Analysis ─────────────────────────────")

# ==========================================================
# Pareto Analysis (80/20 Rule)
# ==========================================================

print("\n── Pareto Analysis ─────────────────────────────")

pareto = (
    product_summary
    .sort_values("Revenue", ascending=False)
    .reset_index(drop=True)
)

pareto["CumulativeRevenue"] = (
    pareto["Revenue"].cumsum()
)

pareto["CumulativeRevenuePercent"] = (
    pareto["CumulativeRevenue"]
    / pareto["Revenue"].sum()
    * 100
)

print(
    pareto[
        [
            "StockCode",
            "Description",
            "Revenue",
            "CumulativeRevenuePercent"
        ]
    ].head(15)
)

# ==========================================================
# Pareto Chart
# ==========================================================

fig, ax1 = plt.subplots(figsize=(12, 6))

# Revenue bars
ax1.bar(
    pareto.index + 1,
    pareto["Revenue"],
    color="steelblue"
)

ax1.set_xlabel("Products Ranked by Revenue")
ax1.set_ylabel("Revenue")

# Secondary Y-axis
ax2 = ax1.twinx()

ax2.plot(
    pareto.index + 1,
    pareto["CumulativeRevenuePercent"],
    color="red",
    linewidth=2,
    label="Cumulative Revenue (%)"
)

ax2.axhline(
    y=80,
    color="green",
    linestyle="--",
    linewidth=2,
    label="80% Threshold"
)

ax2.set_ylabel("Cumulative Revenue (%)")
ax2.set_ylim(0, 100)

plt.title("Pareto Analysis (80/20 Rule)")

ax2.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    "assets/pareto_analysis.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

top_products_80 = (
    pareto["CumulativeRevenuePercent"] <= 80
).sum()

print(
    f"\nTop {top_products_80} products contribute approximately 80% of total revenue."
)


# ----------------------------------------------------------
# Insight
#
# A relatively small percentage of products contributes
# the majority of business revenue, consistent with the
# Pareto Principle.
# ----------------------------------------------------------

# ==========================================================
# ABC Analysis
# ==========================================================

print("\n── ABC Analysis ───────────────────────────────")

def classify_product(cumulative_percent):

    if cumulative_percent <= 80:
        return "A"

    elif cumulative_percent <= 95:
        return "B"

    return "C"


pareto["Category"] = (
    pareto["CumulativeRevenuePercent"]
    .apply(classify_product)
)

abc_summary = (
    pareto.groupby("Category")
    .agg(
        Products=("StockCode", "count"),
        Revenue=("Revenue", "sum")
    )
    .reset_index()
)

abc_summary["RevenueShare (%)"] = (
    abc_summary["Revenue"]
    / abc_summary["Revenue"].sum()
    * 100
).round(2)

print("\nABC Summary\n")
print(abc_summary)

plt.figure(figsize=(6, 5))

bars = plt.bar(
    abc_summary["Category"],
    abc_summary["Products"]
)

plt.bar_label(
    bars,
    fmt="%.0f",
    padding=3
)

plt.title("ABC Product Classification")
plt.xlabel("Category")
plt.ylabel("Number of Products")

plt.tight_layout()

plt.savefig(
    "assets/abc_analysis.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================================
# Export Product Intelligence Outputs
# ==========================================================

pareto.to_csv(
    "data/processed/abc_analysis.csv",
    index=False
)

print("\nABC analysis saved successfully.")

# ==========================================================
# Key Insights
# ==========================================================

'''
Key Insights

1. A relatively small number of products contribute the majority of total revenue.
2. High sales volume does not necessarily imply high revenue generation.
3. Product revenue follows a highly right-skewed distribution.
4. Pareto analysis confirms that a small percentage of products drives most business revenue.
5. ABC analysis classifies products into strategic inventory groups for prioritization and inventory management.
'''