# Power BI Dashboard & Business Intelligence

This folder contains the Power BI dashboard file (`.pbix`) designed to provide executive and operational visibility into customer churn risk, revenue at risk, and customer lifetime value (CLTV).

---

## Files

- `minor_project_dashboard.pbix`: Power BI Desktop dashboard file containing interactive data visualizations, drill-through pages, and KPI cards.

---

## Star Schema Data Model

The dashboard is built upon a Kimball star schema imported from `../data/star_schema/`:

- **Fact Table:** `Fact_Orders.csv` (Key measures: `OrderAmount`, `ChurnProbability`, `ChurnFlag`, `PredictedCLTV`, `IsScored`)
- **Dimension Tables:**
  - `Dim_Customer.csv` (`CustomerID`, `Age`, `Gender`, `Region`, `RFM_Score`, `CustomerSegment`, `Tenure`)
  - `Dim_Date.csv` (`DateKey`, `FullDate`, `Year`, `Quarter`, `Month`, `MonthName`, `DayOfWeek`, `DayName`, `IsWeekend`)
  - `Dim_Product.csv` (`ProductID`, `ProductName`, `ProductCategory`, `BasePrice`)
  - `Dim_SupportTicket.csv` (`CustomerID`, `SupportTicketCount`, `CartAbandonmentRate`)

---

## How to Open and Refresh

1. Install [Power BI Desktop](https://powerbi.microsoft.com/desktop/).
2. Open `minor_project_dashboard.pbix`.
3. If prompt appears indicating source path changes:
   - Click **Transform Data** → **Data source settings**.
   - Select the CSV data source and update file paths to point to your local `data/star_schema/` directory.
   - Click **Close & Apply**.

---

## DAX Measures

All core metrics (e.g. `Churn Rate`, `Revenue at Risk`, `Retention Rate`, `Average Predicted CLTV`) are implemented using custom DAX measures.
Refer to [`../reports/dax_measures.md`](../reports/dax_measures.md) for full DAX formulas, calculation logic, and visualization guidance.
