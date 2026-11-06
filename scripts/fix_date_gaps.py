import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dim_date_path = os.path.join(PROJECT_ROOT, "data", "star_schema", "Dim_Date.csv")
fact_orders_path = os.path.join(PROJECT_ROOT, "data", "star_schema", "Fact_Orders.csv")

# Load existing files to determine the true date range and preserve schema
dim_date = pd.read_csv(dim_date_path)
fact_orders = pd.read_csv(fact_orders_path)

dim_date["FullDate"] = pd.to_datetime(dim_date["FullDate"])
fact_orders_dates = pd.to_datetime(fact_orders["DateKey"].astype(str), format="%Y%m%d")

start = min(dim_date["FullDate"].min(), fact_orders_dates.min())
end = max(dim_date["FullDate"].max(), fact_orders_dates.max())
print(f"Rebuilding Dim_Date from {start.date()} to {end.date()} "
      f"({(end - start).days + 1} days)")

full_range = pd.date_range(start=start, end=end, freq="D")
new_dim_date = pd.DataFrame({"FullDate": full_range})
new_dim_date["DateKey"] = new_dim_date["FullDate"].dt.strftime("%Y%m%d").astype(int)
new_dim_date["Year"] = new_dim_date["FullDate"].dt.year
new_dim_date["Quarter"] = new_dim_date["FullDate"].dt.quarter
new_dim_date["Month"] = new_dim_date["FullDate"].dt.month
new_dim_date["MonthName"] = new_dim_date["FullDate"].dt.strftime("%B")
new_dim_date["DayOfWeek"] = new_dim_date["FullDate"].dt.dayofweek + 1  # 1=Mon
new_dim_date["DayName"] = new_dim_date["FullDate"].dt.strftime("%A")
new_dim_date["IsWeekend"] = new_dim_date["FullDate"].dt.dayofweek >= 5

# Warn if the original file had extra columns this script doesn't know about
extra_cols = set(dim_date.columns) - set(new_dim_date.columns)
if extra_cols:
    print(f"WARNING: original Dim_Date.csv had columns not recreated here: "
          f"{extra_cols}. Check if you need to add logic for these.")

# Reorder to match original column order where possible, keep new cols at end
ordered_cols = [c for c in dim_date.columns if c in new_dim_date.columns]
ordered_cols += [c for c in new_dim_date.columns if c not in ordered_cols]
new_dim_date = new_dim_date[ordered_cols]

new_dim_date.to_csv(dim_date_path, index=False)
print(f"Done. New Dim_Date.csv has {len(new_dim_date)} rows, "
      f"no gaps, no duplicates.")
