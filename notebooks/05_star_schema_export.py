# %% [markdown]
# # 05 — Star Schema Export for Power BI
# Build dimension and fact tables, validate foreign keys, export as CSVs.

# %%
import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in globals() else os.path.abspath(os.path.join(os.getcwd(), '..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

from src.star_schema import (
    build_dim_customer, build_dim_date, build_dim_product,
    build_dim_support_ticket, build_fact_orders,
    validate_star_schema, export_star_schema,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'star_schema')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✓ Imports configured")

# %% [markdown]
# ## Load Source Data

# %%
# Load cleaned data
customers = pd.read_csv(
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'customers_cleaned.csv'),
    parse_dates=['SignupDate']
)
orders = pd.read_csv(
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'orders_cleaned.csv'),
    parse_dates=['OrderDate']
)
behaviour = pd.read_csv(
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'behaviour_support_cleaned.csv')
)

# Load feature matrix and scores
feature_matrix = pd.read_csv(
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_features.csv'),
    index_col='CustomerID'
)
scores = pd.read_csv(
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_scores.csv'),
    index_col='CustomerID'
)

print(f"Customers:      {customers.shape}")
print(f"Orders:         {orders.shape}")
print(f"Behaviour:      {behaviour.shape}")
print(f"Feature Matrix: {feature_matrix.shape}")
print(f"Scores:         {scores.shape}")

# %% [markdown]
# ## Build Dimension Tables

# %%
print("\n" + "="*50)
print("BUILDING DIMENSION TABLES")
print("="*50)

dim_customer = build_dim_customer(customers, feature_matrix)
print(f"\nDim_Customer sample:")
print(dim_customer.head(3).to_string())

# %%
dim_date = build_dim_date(orders)
print(f"\nDim_Date sample:")
print(dim_date.head(3).to_string())

# %%
dim_product = build_dim_product(orders)
print(f"\nDim_Product sample:")
print(dim_product.head(3).to_string())

# %%
dim_support = build_dim_support_ticket(behaviour)
print(f"\nDim_SupportTicket sample:")
print(dim_support.head(3).to_string())

# %% [markdown]
# ## Build Fact Table

# %%
print("\n" + "="*50)
print("BUILDING FACT TABLE")
print("="*50)

fact_orders = build_fact_orders(orders, scores, feature_matrix)
print(f"\nFact_Orders sample:")
print(fact_orders.head(3).to_string())
print(f"\nFact_Orders columns: {list(fact_orders.columns)}")
print(f"ChurnProbability stats:")
print(fact_orders['ChurnProbability'].describe())

# %% [markdown]
# ## Validate Foreign Key Relationships

# %%
print("\n" + "="*50)
print("FOREIGN KEY VALIDATION")
print("="*50)

validation = validate_star_schema(fact_orders, dim_customer, dim_date, dim_product)

print(f"\n{'Check':<30} {'Result':>10}")
print("-" * 42)
for key, value in validation.items():
    if key == 'all_passed':
        continue
    status = "✓ PASS" if (value == 0 or value is True) else f"✗ {value}"
    print(f"{key:<30} {status:>10}")

print(f"\n{'='*42}")
if validation['all_passed']:
    print("ALL VALIDATIONS PASSED ✓")
else:
    print("SOME VALIDATIONS FAILED ✗ — check orphan keys above")
print(f"{'='*42}")

# %% [markdown]
# ## Row Count Summary

# %%
tables = {
    "Dim_Customer": dim_customer,
    "Dim_Date": dim_date,
    "Dim_Product": dim_product,
    "Dim_SupportTicket": dim_support,
    "Fact_Orders": fact_orders,
}

print("\n" + "="*50)
print("STAR SCHEMA — ROW COUNTS")
print("="*50)
print(f"{'Table':<25} {'Rows':>10} {'Columns':>10}")
print("-" * 47)
for name, tbl in tables.items():
    print(f"{name:<25} {len(tbl):>10,} {len(tbl.columns):>10}")

# %% [markdown]
# ## Export to CSV

# %%
export_star_schema(tables, OUTPUT_DIR)

# Verify files exist and sizes
print(f"\nExported files in {OUTPUT_DIR}:")
for name in tables:
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    size_kb = os.path.getsize(path) / 1024
    print(f"  {name}.csv — {size_kb:.1f} KB")

# %% [markdown]
# ## Star Schema Relationship Diagram
# 
# ```
#                    ┌──────────────┐
#                    │  Dim_Date    │
#                    │  PK: DateKey │
#                    └──────┬───────┘
#                           │ 1:N
#     ┌──────────────┐      │      ┌──────────────┐
#     │ Dim_Customer │      │      │ Dim_Product  │
#     │ PK: CustID   │──1:N─┼─N:1──│ PK: ProdID   │
#     └──────┬───────┘      │      └──────────────┘
#            │              │
#            │       ┌──────┴───────┐
#            │       │ Fact_Orders  │
#            │       │ FK: CustID   │
#            │       │ FK: DateKey  │
#            │       │ FK: ProdID   │
#            │       └──────────────┘
#            │
#     ┌──────┴───────────┐
#     │ Dim_SupportTicket│
#     │ PK: CustID       │
#     └──────────────────┘
# ```

# %%
print("\n" + "="*50)
print("POWER BI IMPORT INSTRUCTIONS")
print("="*50)
print("""
1. Open Power BI Desktop
2. Get Data → Text/CSV → Select each CSV from data/star_schema/
3. Set up relationships in Model View:
   - Fact_Orders.CustomerID → Dim_Customer.CustomerID (Many-to-One)
   - Fact_Orders.DateKey    → Dim_Date.DateKey         (Many-to-One)
   - Fact_Orders.ProductID  → Dim_Product.ProductID    (Many-to-One)
   - Dim_Customer.CustomerID → Dim_SupportTicket.CustomerID (One-to-One)
4. Set cross-filter direction to 'Single' for all relationships
5. Mark Dim_Date as a Date Table (DateKey column)
""")

print("\n" + "="*50)
print("NOTEBOOK 05 COMPLETE")
print("="*50)
