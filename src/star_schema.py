"""
Star-schema builder and CSV exporter for Power BI.

Restructures the scored customer data into a star schema consisting of:
  - Dim_Customer  (CustomerID PK)
  - Dim_Date      (DateKey PK, YYYYMMDD integer)
  - Dim_Product   (ProductID PK)
  - Dim_SupportTicket (CustomerID PK)
  - Fact_Orders   (OrderID PK, with FKs + ChurnProbability, ChurnFlag, PredictedCLTV)

Includes FK validation to ensure referential integrity before export.

Author: Generated for Minor Project (Sem 5)
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Dimension Table Builders
# ---------------------------------------------------------------------------

def build_dim_customer(
    customers_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the Dim_Customer dimension table.

    Columns: CustomerID (PK), Age, Gender, Region, AcquisitionChannel,
             SignupDate, RFM_Score, CustomerSegment, Tenure (days).

    Parameters
    ----------
    customers_df : pd.DataFrame
        Raw/cleaned customer table.
    feature_matrix : pd.DataFrame
        Feature matrix indexed by CustomerID (must contain RFM_Score,
        CustomerSegment, DaysSinceSignup).

    Returns
    -------
    pd.DataFrame
    """
    cust = customers_df.copy()
    if "CustomerID" not in cust.columns and cust.index.name == "CustomerID":
        cust = cust.reset_index()

    # Pull RFM info from the feature matrix
    rfm_cols = ["RFM_Score", "CustomerSegment", "DaysSinceSignup"]
    available_cols = [c for c in rfm_cols if c in feature_matrix.columns]
    rfm_info = feature_matrix[available_cols].copy()
    if rfm_info.index.name != "CustomerID":
        rfm_info.index.name = "CustomerID"
    rfm_info = rfm_info.reset_index()

    dim = cust.merge(rfm_info, on="CustomerID", how="left")

    # Rename DaysSinceSignup → Tenure for BI clarity
    if "DaysSinceSignup" in dim.columns:
        dim = dim.rename(columns={"DaysSinceSignup": "Tenure"})

    # Ensure date format is Power BI friendly
    dim["SignupDate"] = pd.to_datetime(dim["SignupDate"]).dt.strftime("%Y-%m-%d")

    logger.info(f"Dim_Customer: {len(dim)} rows")
    return dim


def build_dim_date(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Dim_Date dimension table from all unique order dates.

    Columns: DateKey (PK, int YYYYMMDD), FullDate, Year, Quarter, Month,
             MonthName, DayOfWeek (0=Mon), DayName, IsWeekend.
    """
    dates = pd.to_datetime(orders_df["OrderDate"]).dt.normalize().drop_duplicates()
    dates = dates.sort_values().reset_index(drop=True)

    dim = pd.DataFrame({"FullDate": dates})
    dim["DateKey"] = dim["FullDate"].dt.strftime("%Y%m%d").astype(int)
    dim["Year"] = dim["FullDate"].dt.year
    dim["Quarter"] = dim["FullDate"].dt.quarter
    dim["Month"] = dim["FullDate"].dt.month
    dim["MonthName"] = dim["FullDate"].dt.strftime("%B")
    dim["DayOfWeek"] = dim["FullDate"].dt.dayofweek  # 0=Monday
    dim["DayName"] = dim["FullDate"].dt.strftime("%A")
    dim["IsWeekend"] = dim["DayOfWeek"].isin([5, 6]).astype(int)
    dim["FullDate"] = dim["FullDate"].dt.strftime("%Y-%m-%d")

    logger.info(f"Dim_Date: {len(dim)} rows ({dim['Year'].min()}-{dim['Year'].max()})")
    return dim


def build_dim_product(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Dim_Product dimension table from unique products in orders.

    Columns: ProductID (PK), ProductName, ProductCategory.
    """
    cols = ["ProductID", "ProductName", "ProductCategory"]
    dim = orders_df[cols].drop_duplicates(subset=["ProductID"]).reset_index(drop=True)
    dim = dim.sort_values("ProductID").reset_index(drop=True)
    logger.info(f"Dim_Product: {len(dim)} rows")
    return dim


def build_dim_support_ticket(behaviour_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Dim_SupportTicket dimension table.

    Columns: CustomerID (PK), SupportTicketCount,
             LastSupportResolutionTime, CartAbandonmentRate.
    """
    cols = ["CustomerID", "SupportTicketCount",
            "LastSupportResolutionTime", "CartAbandonmentRate"]
    dim = behaviour_df[cols].copy()
    if dim.index.name == "CustomerID":
        dim = dim.reset_index()
    logger.info(f"Dim_SupportTicket: {len(dim)} rows")
    return dim


def build_fact_orders(
    orders_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the Fact_Orders fact table.

    Columns: OrderID (PK), CustomerID (FK), DateKey (FK), ProductID (FK),
             OrderAmount, Quantity, DiscountUsed, ReturnFlag,
             ChurnProbability, ChurnFlag, PredictedCLTV.

    PredictedCLTV formula:
        CLTV = AvgOrderValue × PredictedFreq12m × (1 - ChurnProbability)
    where PredictedFreq12m = historical Frequency scaled to 12 months
    based on customer tenure.
    """
    fact = orders_df[
        ["OrderID", "CustomerID", "OrderDate", "ProductID",
         "OrderAmount", "Quantity", "DiscountUsed", "ReturnFlag"]
    ].copy()

    # Add DateKey (FK to Dim_Date)
    fact["DateKey"] = pd.to_datetime(fact["OrderDate"]).dt.strftime("%Y%m%d").astype(int)
    fact = fact.drop(columns=["OrderDate"])

    # --- Merge churn scores ---
    score_cols = scores_df[["ChurnProbability", "ChurnPrediction"]].copy()
    score_cols = score_cols.rename(columns={"ChurnPrediction": "ChurnFlag"})
    if score_cols.index.name != "CustomerID":
        score_cols.index.name = "CustomerID"
    score_cols = score_cols.reset_index()

    fact = fact.merge(score_cols, on="CustomerID", how="left")

    # --- Mark which rows have model scores ---
    # Customers without pre-cutoff orders were never scored by the model.
    # We add an IsScored flag so Power BI can filter dashboards to only
    # show model-backed metrics (churn rate, revenue at risk, CLTV).
    fact["IsScored"] = fact["ChurnProbability"].notna().astype(int)

    scored_count = fact["IsScored"].sum()
    unscored_count = len(fact) - scored_count
    logger.info(
        f"  Fact scoring: {scored_count:,} rows scored, "
        f"{unscored_count:,} rows unscored (no pre-cutoff orders)"
    )

    # --- Compute PredictedCLTV (only for scored customers) ---
    cltv_cols = ["Frequency", "AvgOrderValue", "DaysSinceSignup"]
    fm = feature_matrix[cltv_cols].copy()
    if fm.index.name != "CustomerID":
        fm.index.name = "CustomerID"
    fm = fm.reset_index()

    fact = fact.merge(fm, on="CustomerID", how="left")

    # PredictedFreq12m = Frequency * (365 / DaysSinceSignup), capped
    tenure_days = fact["DaysSinceSignup"].clip(lower=30)
    predicted_freq_12m = (fact["Frequency"] * (365.0 / tenure_days)).clip(upper=52)

    # CLTV = AvgOrderValue * PredictedFreq12m * (1 - ChurnProbability)
    fact["PredictedCLTV"] = (
        fact["AvgOrderValue"]
        * predicted_freq_12m
        * (1 - fact["ChurnProbability"])
    ).round(2)
    # PredictedCLTV stays NaN for unscored customers (intentional)

    # Drop helper columns
    fact = fact.drop(columns=["Frequency", "AvgOrderValue", "DaysSinceSignup"])

    # --- Fill defaults for unscored customers ---
    # ChurnProbability: NaN → -1 (sentinel; Power BI can filter on IsScored=1)
    # ChurnFlag: NaN → -1 (sentinel, meaning "unknown / not scored")
    # This avoids inflating churn rate with fake values.
    fact["ChurnProbability"] = fact["ChurnProbability"].fillna(-1).round(4)
    fact["ChurnFlag"] = fact["ChurnFlag"].fillna(-1).astype(int)
    fact["PredictedCLTV"] = fact["PredictedCLTV"].fillna(-1).round(2)

    # Ensure boolean columns are int for Power BI
    fact["DiscountUsed"] = fact["DiscountUsed"].astype(int)
    fact["ReturnFlag"] = fact["ReturnFlag"].astype(int)

    logger.info(f"Fact_Orders: {len(fact)} rows")
    return fact


# ---------------------------------------------------------------------------
# FK Validation
# ---------------------------------------------------------------------------

def validate_star_schema(
    fact: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_date: pd.DataFrame,
    dim_product: pd.DataFrame,
) -> dict:
    """
    Validate foreign-key relationships in the star schema.

    Checks
    ------
    1. All CustomerID in Fact → exist in Dim_Customer
    2. All DateKey in Fact → exist in Dim_Date
    3. All ProductID in Fact → exist in Dim_Product
    4. No null primary/foreign keys

    Returns
    -------
    dict with validation results and a boolean ``all_passed``.
    """
    results = {}

    # --- CustomerID ---
    fact_cids = set(fact["CustomerID"].dropna().unique())
    dim_cids = set(dim_customer["CustomerID"].unique())
    orphan_cids = fact_cids - dim_cids
    results["CustomerID_orphans"] = len(orphan_cids)
    results["CustomerID_check"] = len(orphan_cids) == 0

    # --- DateKey ---
    fact_dks = set(fact["DateKey"].dropna().unique())
    dim_dks = set(dim_date["DateKey"].unique())
    orphan_dks = fact_dks - dim_dks
    results["DateKey_orphans"] = len(orphan_dks)
    results["DateKey_check"] = len(orphan_dks) == 0

    # --- ProductID ---
    fact_pids = set(fact["ProductID"].dropna().unique())
    dim_pids = set(dim_product["ProductID"].unique())
    orphan_pids = fact_pids - dim_pids
    results["ProductID_orphans"] = len(orphan_pids)
    results["ProductID_check"] = len(orphan_pids) == 0

    # --- Null PKs/FKs ---
    null_checks = {
        "Fact_OrderID_nulls": fact["OrderID"].isna().sum(),
        "Fact_CustomerID_nulls": fact["CustomerID"].isna().sum(),
        "Fact_DateKey_nulls": fact["DateKey"].isna().sum(),
        "Fact_ProductID_nulls": fact["ProductID"].isna().sum(),
        "Dim_Customer_PK_nulls": dim_customer["CustomerID"].isna().sum(),
        "Dim_Date_PK_nulls": dim_date["DateKey"].isna().sum(),
        "Dim_Product_PK_nulls": dim_product["ProductID"].isna().sum(),
    }
    results.update(null_checks)
    results["no_null_keys"] = all(v == 0 for v in null_checks.values())

    results["all_passed"] = (
        results["CustomerID_check"]
        and results["DateKey_check"]
        and results["ProductID_check"]
        and results["no_null_keys"]
    )

    status = "PASSED ✓" if results["all_passed"] else "FAILED ✗"
    logger.info(f"Star schema validation: {status}")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_star_schema(
    tables: dict[str, pd.DataFrame],
    output_dir: str = "data/star_schema",
) -> None:
    """
    Export all star-schema tables as clean CSVs.

    Parameters
    ----------
    tables : dict
        Mapping of table name → DataFrame. Expected keys:
        Dim_Customer, Dim_Date, Dim_Product, Dim_SupportTicket, Fact_Orders.
    output_dir : str
        Directory to write CSVs into.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, df in tables.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        logger.info(f"Exported {name}: {len(df)} rows → {path}")

    logger.info(f"All {len(tables)} tables exported to {output_dir}")


# ---------------------------------------------------------------------------
# Convenience: build everything
# ---------------------------------------------------------------------------

def build_and_export_star_schema(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    behaviour_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    scores_df: pd.DataFrame,
    output_dir: str = "data/star_schema",
) -> dict:
    """
    One-call function to build all star-schema tables, validate, and export.

    Returns
    -------
    dict with keys: tables (dict of DataFrames), validation (dict).
    """
    dim_customer = build_dim_customer(customers_df, feature_matrix)
    dim_date = build_dim_date(orders_df)
    dim_product = build_dim_product(orders_df)
    dim_support = build_dim_support_ticket(behaviour_df)
    fact_orders = build_fact_orders(orders_df, scores_df, feature_matrix)

    tables = {
        "Dim_Customer": dim_customer,
        "Dim_Date": dim_date,
        "Dim_Product": dim_product,
        "Dim_SupportTicket": dim_support,
        "Fact_Orders": fact_orders,
    }

    validation = validate_star_schema(fact_orders, dim_customer, dim_date, dim_product)

    export_star_schema(tables, output_dir)

    return {"tables": tables, "validation": validation}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("star_schema.py — run via notebook 05_star_schema_export.")
    print("Key functions: build_and_export_star_schema(), validate_star_schema()")
