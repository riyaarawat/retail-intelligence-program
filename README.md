# Retail Analytics & Customer Intelligence Platform

End-to-end retail analytics project built using **Python and Power BI** to analyze transaction-level retail data and generate actionable business insights.

This project demonstrates the complete analytics workflow: **data cleaning → exploratory analysis → customer segmentation → revenue analysis → interactive dashboarding**.

---

## Project Overview

The project uses the **Online Retail II** dataset to answer key business questions such as:

- Which products generate the highest revenue?
- What are the monthly sales trends and seasonality patterns?
- Who are the most valuable customers?
- Which customers are inactive or at risk of churn?
- How can business stakeholders make data-driven decisions?

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboarding | Power BI |
| Machine Learning | K-Means Clustering, Prophet |
| Environment | Visual Studio Code |

---

## Key Features

### Data Cleaning & Preparation
- Removed cancelled and invalid transactions
- Handled missing values
- Converted date columns to proper datetime format
- Created additional analytical features

### Sales & Revenue Analysis
- Total revenue calculation
- Monthly revenue trends
- Top-performing products
- Product-wise contribution analysis

### Customer Intelligence
- Customer-level aggregation
- Recency, Frequency, Monetary (RFM) analysis
- Identification of high-value customers
- Detection of inactive customers

### Dashboarding
- KPI overview
- Revenue trend visualization
- Customer segmentation charts
- Product performance analysis

---

## Dashboard Preview

### Revenue & KPI Overview
![Overview](assets/overview.png)

### Monthly Revenue Trend
![Revenue](assets/revenue.png)

### Customer Segmentation
![RFM](assets/rfm.png)

---

## Project Structure

```text
retail-analytics-and-customer-intelligence-platform/
├── assets/            # Dashboard screenshots
├── dashboard/         # Power BI / Tableau dashboard files
├── data/              # Processed datasets
├── notebooks/         # Jupyter notebooks for analysis
├── requirements.txt   # Python dependencies
└── README.md
```

---

## How to Run

### Clone the repository

```bash
git clone https://github.com/riyaarawat/retail-analytics-and-customer-intelligence-platform.git
cd retail-analytics-and-customer-intelligence-platform
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks inside the `notebooks/` folder to reproduce the analysis.

---

## Dataset

The project is based on the **Online Retail II** dataset.

**Source:** https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

> The raw dataset is not included in this repository due to GitHub file size limits.

---

## Business Insights

Some insights generated from the analysis include:

- A small group of customers contributes a disproportionately high share of revenue.
- Sales show strong seasonal patterns across months.
- Certain products consistently outperform others in both revenue and purchase frequency.
- Inactive customers can be targeted through retention and re-engagement campaigns.

---

## What This Project Demonstrates

- Real-world data cleaning
- Exploratory Data Analysis (EDA)
- Customer analytics
- Business intelligence dashboarding
- Data storytelling
- End-to-end analytics workflow

---

## Future Improvements

- Churn prediction model
- Customer lifetime value estimation
- Sales forecasting
- Automated reporting pipeline
- Deployment as a web dashboard

---

## Author

**Riya Rawat**

- GitHub: https://github.com/riyaarawat
- LinkedIn: https://www.linkedin.com/in/riyarawat2003

---
