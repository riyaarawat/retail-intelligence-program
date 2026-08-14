import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

df_sales = pd.read_csv(
    "data/cleaned/retail_cleaned.csv",
    parse_dates=["InvoiceDate"]
)

print("\n========== DEMAND FORECASTING REPORT ==========")

print("\n── Monthly Revenue ─────────────────────────────")

monthly_sales = (
    df_sales.groupby(
        pd.Grouper(
            key="InvoiceDate",
            freq="ME"
        )
    )["Revenue"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "InvoiceDate": "InvoiceMonth"
        }
    )
)

print(monthly_sales)

print("\nMonthly Revenue Statistics\n")
print(monthly_sales["Revenue"].describe())

monthly_sales.to_csv(
    "data/processed/monthly_sales.csv",
    index=False
)

print("\nMonthly revenue saved successfully.")

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_sales["InvoiceMonth"],
    monthly_sales["Revenue"],
    marker="o",
    linewidth=2
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "assets/monthly_revenue_trend.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\n── Preparing Data for Prophet ──────────────────")

forecast_data = (
    monthly_sales.rename(
        columns={
            "InvoiceMonth": "ds",
            "Revenue": "y"
        }
    )
)

print(forecast_data.head())

print("\n── Training Prophet Model ─────────────────────")

model = Prophet()

model.fit(forecast_data)

print("Model trained successfully.")

print("\n── Forecasting Next 6 Months ───────────────────")

future = model.make_future_dataframe(
    periods=6,
    freq="ME"
)

forecast = model.predict(future)

forecast_summary = (
    forecast[
        [
            "ds",
            "trend",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ]
    .rename(
        columns={
            "ds": "Month",
            "trend": "Trend",
            "yhat": "ForecastRevenue",
            "yhat_lower": "LowerBound",
            "yhat_upper": "UpperBound"
        }
    )
)

print("\nForecast Summary\n")
print(forecast_summary.tail(6))

model.plot(forecast)

plt.title("Six-Month Revenue Forecast")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig(
    "assets/revenue_forecast.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

model.plot_components(forecast)

plt.tight_layout()

plt.savefig(
    "assets/forecast_components.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

forecast_summary.to_csv(
    "data/processed/monthly_forecast.csv",
    index=False
)

print("\nMonthly forecast saved successfully.")

# key findings 
'''
1. Prophet was used to forecast monthly revenue for the next
   six months using historical transaction data.

2. The model estimates expected monthly revenue together
   with lower and upper confidence intervals.

3. The forecast trend can support inventory planning,
   budgeting and resource allocation.

4. The forecasting model should be retrained periodically
   as new sales data becomes available.
'''