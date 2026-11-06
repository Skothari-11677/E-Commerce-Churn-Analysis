import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
fact_orders_path = os.path.join(PROJECT_ROOT, "data", "star_schema", "Fact_Orders.csv")
dim_customer_path = os.path.join(PROJECT_ROOT, "data", "star_schema", "Dim_Customer.csv")

fact_orders = pd.read_csv(fact_orders_path)
dim_customer = pd.read_csv(dim_customer_path)

# these 4 columns are constant per customer across all their order rows —
# pull one value per customer
customer_scores = fact_orders.groupby("CustomerID").agg({
    "ChurnProbability": "first",
    "ChurnFlag": "first",
    "PredictedCLTV": "first",
    "IsScored": "first"
}).reset_index()

# sanity check: confirm they really are constant per customer, not varying by order
check = fact_orders.groupby("CustomerID")[
    ["ChurnProbability", "ChurnFlag", "PredictedCLTV", "IsScored"]
].nunique()
inconsistent = check[(check > 1).any(axis=1)]
if len(inconsistent) > 0:
    print(f"WARNING: {len(inconsistent)} customers have inconsistent scores "
          f"across their own orders! Something's wrong upstream — stop and "
          f"investigate before proceeding.")
    print(inconsistent.head())
else:
    print("Confirmed: churn scores are consistent per customer. Safe to proceed.")

dim_customer = dim_customer.merge(customer_scores, on="CustomerID", how="left")
dim_customer.to_csv(dim_customer_path, index=False)
print(f"Dim_Customer.csv updated: {len(dim_customer)} rows total, "
      f"{int(dim_customer['IsScored'].sum())} scored")
