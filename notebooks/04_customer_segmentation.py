import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_customers = pd.read_csv("data/cleaned/retail_customers.csv", parse_dates=["InvoiceDate"])

print("\n========== CUSTOMER SEGMENTATION REPORT ==========")

print("\n── Customer Retention Analysis ───────────────────")

customer_orders = (
    df_customers.groupby("CustomerID")["Invoice"]
    .nunique()
    .reset_index(name="Orders")
)

repeat_customers = (customer_orders["Orders"] > 1).sum()
one_time_customers = (customer_orders["Orders"] == 1).sum()

total_customers = len(customer_orders)

repeat_rate = repeat_customers / total_customers * 100
one_time_rate = one_time_customers / total_customers * 100

print(f"Total Customers       : {total_customers:,}")
print(f"Repeat Customers      : {repeat_customers:,}")
print(f"Repeat Customer Rate  : {repeat_rate:.2f}%")
print(f"One-Time Customers    : {one_time_customers:,}")
print(f"One-Time Customer Rate: {one_time_rate:.2f}%")

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Repeat", "One-Time"],
    [repeat_customers, one_time_customers]
)

plt.bar_label(
    bars,
    fmt="%.0f",
    padding=3
)

plt.title("Customer Retention")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("assets/customer_retention.png")

plt.show()
plt.close()

print("\n── RFM Feature Engineering ───────────────────────")

analysis_date = (df_customers["InvoiceDate"].max() + pd.Timedelta(days=1))

customer_features = (
    df_customers.groupby("CustomerID")
    .agg(
        Recency=(
            "InvoiceDate",
            lambda x: (analysis_date - x.max()).days
        ),
        Frequency=(
            "Invoice",
            "nunique"
        ),
        Monetary=(
            "Revenue",
            "sum"
        )
    )
    .reset_index()
)

print(customer_features.head())

print("\nRFM Summary Statistics\n")
print(customer_features[["Recency", "Frequency", "Monetary"]].describe())

print("\n── Feature Scaling ───────────────────────────────")

features = customer_features[["Recency", "Frequency", "Monetary"]]

scaler = StandardScaler()

scaled_features = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns
)

scaled_features.insert(
    0,
    "CustomerID",
    customer_features["CustomerID"]
)

X = scaled_features.drop(columns="CustomerID")

print(scaled_features.head())

print("\n── Elbow Method ────────────────────────────────")

k_values = range(2, 11)
inertia = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.xticks(k_values)

plt.tight_layout()

plt.savefig("assets/elbow_method.png")

plt.show()
plt.close()

print("\n── Silhouette Score ───────────────────────────")

silhouette_scores = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    score = silhouette_score(
        X,
        labels
    )

    silhouette_scores.append(score)

    print(f"k = {k}: {score:.4f}")

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")

plt.xticks(k_values)

plt.tight_layout()

plt.savefig(
    "assets/silhouette_score.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\n── K-Means Customer Segmentation ───────────────")

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

customer_features["Cluster"] = (
    kmeans.fit_predict(X)
)

print(customer_features.head())

print("\n── Customer Cluster Summary ────────────────────")

cluster_summary = (
    customer_features.groupby("Cluster")
    .agg(
        TotalCustomers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum")
    )
    .round(2)
    .reset_index()
)

cluster_summary["RevenueShare (%)"] = (
    cluster_summary["TotalRevenue"]
    / cluster_summary["TotalRevenue"].sum()
    * 100
).round(2)

print(cluster_summary)

cluster_names = {
    0: "Regular Customers",
    1: "Dormant Customers",
    2: "Elite VIP Customers",
    3: "High-Value Loyal Customers",
    4: "Loyal Premium Customers"
}

customer_features["ClusterName"] = (
    customer_features["Cluster"]
    .map(cluster_names)
)

cluster_summary["ClusterName"] = (
    cluster_summary["Cluster"]
    .map(cluster_names)
)

cluster_summary = cluster_summary[
    [
        "Cluster",
        "ClusterName",
        "TotalCustomers",
        "AvgRecency",
        "AvgFrequency",
        "AvgMonetary",
        "TotalRevenue",
        "RevenueShare (%)"
    ]
]

print("\nNamed Customer Clusters\n")
print(cluster_summary)

# ==========================================================
# Customer Distribution by Cluster
# ==========================================================

print("\n── Customer Cluster Visualizations ─────────────")

plt.figure(figsize=(10, 5))

bars = plt.bar(
    cluster_summary["ClusterName"],
    cluster_summary["TotalCustomers"]
)

plt.bar_label(
    bars,
    fmt="%.0f",
    padding=3
)

plt.title("Customer Distribution by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("assets/customer_cluster_distribution.png")

plt.show()
plt.close()

# ==========================================================
# Revenue Contribution by Cluster
# ==========================================================

plt.figure(figsize=(10, 5))

bars = plt.bar(
    cluster_summary["ClusterName"],
    cluster_summary["RevenueShare (%)"]
)

plt.bar_label(
    bars,
    fmt="%.1f%%",
    padding=3
)

plt.title("Revenue Contribution by Customer Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Revenue Share (%)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("assets/customer_cluster_revenue.png")

plt.show()
plt.close()

# ==========================================================
# Customer Cluster Scatter Plot
# ==========================================================

plt.figure(figsize=(10, 6))

for cluster in sorted(customer_features["Cluster"].unique()):

    subset = customer_features[
        customer_features["Cluster"] == cluster
    ]

    plt.scatter(
        subset["Recency"],
        subset["Monetary"],
        alpha=0.7,
        label=cluster_names[cluster]
    )

plt.yscale("log")

plt.title("Customer Segments (K-Means)")
plt.xlabel("Recency (Days)")
plt.ylabel("Monetary Value (Log Scale)")

plt.legend()
plt.tight_layout()

plt.savefig("assets/customer_cluster_scatter.png")

plt.show()
plt.close()

customer_features.to_csv(
    "data/processed/customer_clusters.csv",
    index=False
)

cluster_summary.to_csv(
    "data/processed/customer_cluster_summary.csv",
    index=False
)

print("\nCustomer intelligence outputs saved successfully.")

'''
Key Insights

1. Most customers belong to a small number of behavioral segments.
2. Revenue is concentrated among high-value customer clusters.
3. Dormant customers represent a potential re-engagement opportunity.
4. VIP and loyal customer segments contribute disproportionately to total revenue.
5. RFM-based clustering enables targeted marketing and customer retention strategies.
'''