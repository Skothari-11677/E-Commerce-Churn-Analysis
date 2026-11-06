import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
fact_orders_path = os.path.join(PROJECT_ROOT, "data", "star_schema", "Fact_Orders.csv")

df = pd.read_csv(fact_orders_path)

scored = df["IsScored"] == 1
df.loc[scored, "ChurnFlag"] = (df.loc[scored, "ChurnProbability"] >= 0.40).astype(int)

# sanity check before saving
print(df.loc[scored, "ChurnFlag"].value_counts())
print("Unscored rows untouched:", (df.loc[~scored, "ChurnFlag"] == -1).all())

df.to_csv(fact_orders_path, index=False)
print("Updated Fact_Orders.csv successfully.")
