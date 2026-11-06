# Maintenance & Utility Scripts

This directory contains standalone data fix and maintenance utilities used during dataset preparation and star schema post-processing.

---

## Script Manifest

| Script | Purpose |
|---|---|
| `fix_churn_flag.py` | Adjusts or standardizes the classification threshold (0.40) and churn binary flag on `Fact_Orders.csv` for scored records while preserving `-1` unscored placeholders. |
| `fix_date_gaps.py` | Rebuilds `Dim_Date.csv` across the entire contiguous temporal range spanned by `Fact_Orders` to ensure continuous date dimension without missing days. |
| `fix_dimCustomer.py` | Denormalizes customer-level churn scores and predicted CLTV from `Fact_Orders` back into `Dim_Customer` for direct slice-and-dice customer analytics in Power BI. |
| `fix_dim_customer_duplicates.ipynb` | Jupyter notebook documenting verification and cleanup of duplicate columns during merge operations on `Dim_Customer.csv`. |

---

## Execution

All scripts can be executed directly from either the project root or the `scripts/` directory:

```bash
# Example: Running a fix script from repository root
python scripts/fix_date_gaps.py
```
