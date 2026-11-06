"""
Churn labelling using the snapshot/cutoff design.

Design
------
- Cutoff date: July 1, 2023 (month 12 of the 18-month window)
- All features are computed from orders BEFORE the cutoff
- ChurnFlag is computed from orders AFTER the cutoff
- Each customer's churn window = cutoff + (category_cycle × multiplier)
- Customer is churned if they have ZERO orders in their churn window

Category expected repurchase cycles (days):
  Electronics Accessories / Groceries : 60
  Clothing / Beauty / Books           : 90
  Home & Garden / Sports              : 105
  Furniture                           : 120

Multiplier: 1.5 (default) → actual windows are 90, 135, 157, 180 days

Author: Generated for Minor Project (Sem 5)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Category → expected repurchase cycle (days)
# ---------------------------------------------------------------------------

def get_category_cycles() -> dict:
    """
    Return a mapping of product category → expected repurchase cycle in days.

    These values represent how often a typical customer re-orders items
    in each category. They are used with a multiplier (default 1.5×) to
    define the churn window per customer.
    """
    return {
        "Electronics Accessories": 60,
        "Groceries": 60,
        "Clothing": 90,
        "Beauty": 90,
        "Books": 90,
        "Home & Garden": 105,
        "Sports": 105,
        "Furniture": 120,
    }


# ---------------------------------------------------------------------------
# Churn label computation (vectorised)
# ---------------------------------------------------------------------------

def compute_churn_labels(
    orders_df: pd.DataFrame,
    feature_matrix_df: pd.DataFrame,
    cutoff_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
    multiplier: float = 1.5,
) -> pd.Series:
    """
    Compute binary churn labels using the snapshot/cutoff design.

    For each customer in *feature_matrix_df*:

    1. Look up their **DominantCategory** (most-purchased category before
       the cutoff, computed during feature engineering).
    2. Map it to the expected repurchase cycle via ``get_category_cycles()``.
    3. Define the customer's personal churn window:
       ``cutoff_date`` → ``cutoff_date + cycle × multiplier``.
    4. Check whether the customer placed **any** order in ``orders_df``
       strictly after ``cutoff_date`` and on or before the window end.
    5. No orders → ``ChurnFlag = 1`` (churned).
       ≥1 order  → ``ChurnFlag = 0`` (retained).

    Parameters
    ----------
    orders_df : pd.DataFrame
        Full order history (must include OrderDate and CustomerID).
    feature_matrix_df : pd.DataFrame
        Feature matrix indexed by CustomerID, must contain
        ``DominantCategory``.
    cutoff_date : pd.Timestamp
        The snapshot date (default 2023-07-01).
    multiplier : float
        Multiplier applied to the category cycle (default 1.5).

    Returns
    -------
    pd.Series
        Binary churn labels (0/1) indexed by CustomerID, named
        ``ChurnFlag``.
    """
    cycles = get_category_cycles()
    default_cycle = 90  # fallback if category is missing

    # --- Build per-customer churn window end dates (vectorised) ---
    dom_cat = feature_matrix_df["DominantCategory"].copy()
    cycle_days = dom_cat.map(cycles).fillna(default_cycle)
    churn_window_end = cutoff_date + pd.to_timedelta(
        cycle_days * multiplier, unit="D"
    )  # Series indexed by CustomerID

    # --- Filter orders to post-cutoff only ---
    orders_dt = pd.to_datetime(orders_df["OrderDate"])
    post = orders_df.loc[orders_dt > cutoff_date, ["CustomerID", "OrderDate"]].copy()
    post["OrderDate"] = pd.to_datetime(post["OrderDate"])

    # --- For each customer, find their earliest post-cutoff order ---
    if not post.empty:
        first_post = post.groupby("CustomerID")["OrderDate"].min()
    else:
        first_post = pd.Series(dtype="datetime64[ns]", name="OrderDate")

    # --- Determine churn: no order within window → churned ---
    labels = pd.Series(
        index=feature_matrix_df.index, dtype=int, name="ChurnFlag"
    )
    for cid in feature_matrix_df.index:
        window_end = churn_window_end.loc[cid]
        if cid in first_post.index and first_post.loc[cid] <= window_end:
            labels.loc[cid] = 0  # retained — ordered within window
        else:
            labels.loc[cid] = 1  # churned — no order in window

    logger.info(
        f"Churn labels computed: {labels.sum()} churned / "
        f"{len(labels)} total ({labels.mean():.1%} churn rate)"
    )
    return labels


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize_churn(labels: pd.Series) -> dict:
    """
    Return a summary dict of churn statistics.

    Returns
    -------
    dict with keys: total_customers, churned_count, retained_count,
                    churn_rate (float 0–1).
    """
    total = len(labels)
    churned = int(labels.sum())
    retained = total - churned
    rate = churned / total if total > 0 else 0.0

    summary = {
        "total_customers": total,
        "churned_count": churned,
        "retained_count": retained,
        "churn_rate": round(rate, 4),
    }
    logger.info(f"Churn summary: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("churn_label.py — run via notebook 02_feature_engineering.")
    print("Key functions: get_category_cycles(), compute_churn_labels(), summarize_churn()")
