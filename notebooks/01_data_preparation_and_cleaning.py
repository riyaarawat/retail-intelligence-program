import pandas as pd

file_path = "data/raw/online_retail_II.csv"

df = pd.read_csv(file_path)

print("\nDataset Loaded Successfully!")

print("\n── Dataset Info Report ────────────────────")
df.info()                     #modify datatypes: InvoiceDate from str to datetime, customerID from float to str
print("\nNumber of Rows and Columns:", df.shape)
print("\nSample Data Rows:-\n",df.sample(5))

print("\n── Missing Values Report ────────────────────")
print("Missing Values per Column:", df.isnull().sum())              #missing values found: description-4382, customerID-243007 (guest purchase)
print("Percentage of Missing Values:", df.isnull().sum()/len(df) * 100)

print("\n── Duplicate Rows Report ────────────────────")
print("Duplicate Rows:", df.duplicated().sum())         #34335 duplicate rows found
print("Percentage of Duplicate Rows:", df.duplicated().sum()/len(df) * 100)

#minimum quantity value
print("\nMinimum Quantity:", df["Quantity"].min())       #-ve quantities indicate returns,refunds,or cancelled orders, if not data entry error

#maximum quantity value
print("\nMaximum Quantity:", df['Quantity'].max())       #get rid of reversal transactions

#minimum price value
print("\nMinimum Price:", df["Price"].min())             #-ve price indicate either refund adjustment or data entry error

#maximum price value
print("\nMaximum Price:", df["Price"].max())

#return percentage
return_rate = (df["Quantity"] < 0).mean() * 100
print("\nReturn Rate:", return_rate,"%")


#extract sales-only dataset
df_cleaned = (df[
        (df["Quantity"] > 0) &
        (df["Price"] > 0)
    ].drop_duplicates().copy()
)

#type conversions
df_cleaned["InvoiceDate"] = pd.to_datetime(df_cleaned["InvoiceDate"])
df_cleaned["StockCode"] = df_cleaned["StockCode"].astype(str)
df_cleaned["Revenue"] = (df_cleaned["Quantity"] * df_cleaned["Price"]
)
df_cleaned.rename(columns={"Customer ID": "CustomerID"},inplace=True)

df_customers = df_cleaned.dropna(subset=["CustomerID"]).copy()
df_customers["CustomerID"] = (df_customers["CustomerID"].astype(int).astype(str))

#final validation
print("Original Rows:", len(df), "Updated Rows:", len(df_cleaned))
df_cleaned.info()
print(df_cleaned.sample(5))

print("\nMissing Values:", df_cleaned.isnull().sum())

print("\nMinimum Quantity:", df_cleaned["Quantity"].min())

print("\nMinimum Price:", df_cleaned["Price"].min())

df_cleaned.to_csv(
    "data/cleaned/retail_cleaned.csv",
    index=False
)
df_customers.to_csv(
    "data/cleaned/retail_customers.csv",
    index=False
)

#conclusion: transformed the raw retail transaction dataset into a clean, analysis-ready dataset that can be reliably used for business intelligence, dashboarding, SQL analytics, customer segmentation, and machine learning.