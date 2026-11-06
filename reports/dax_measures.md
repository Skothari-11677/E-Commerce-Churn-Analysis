# DAX Measures Reference — Power BI Dashboard

> **Important**: The `Fact_Orders` table contains an `IsScored` column (0 or 1).
> Only customers with `IsScored = 1` have valid model-backed churn predictions.
> All churn-related measures below filter on `IsScored = 1` to avoid inflating
> metrics with unscored customers.

---

### 1. Churn Rate

```dax
Churn Rate =
VAR ScoredCustomers =
    CALCULATE(
        DISTINCTCOUNT(Fact_Orders[CustomerID]),
        Fact_Orders[IsScored] = 1
    )
VAR ChurnedCustomers =
    CALCULATE(
        DISTINCTCOUNT(Fact_Orders[CustomerID]),
        Fact_Orders[IsScored] = 1,
        Fact_Orders[ChurnFlag] = 1
    )
RETURN
    DIVIDE(ChurnedCustomers, ScoredCustomers, 0)
```

**Explanation:** Percentage of scored customers predicted to churn. Only counts customers whose orders were modelled (pre-cutoff activity exists).
**Visual:** Card, KPI, Line Chart (over time by Dim_Date)

---

### 2. Revenue at Risk

```dax
Revenue at Risk =
    CALCULATE(
        SUM(Fact_Orders[OrderAmount]),
        Fact_Orders[IsScored] = 1,
        Fact_Orders[ChurnFlag] = 1
    )
```

**Explanation:** Total historical revenue from customers predicted to churn. Filtered to scored customers only.
**Visual:** Card, Bar Chart (by Region or ProductCategory)

---

### 3. Retention Rate

```dax
Retention Rate = 1 - [Churn Rate]
```

**Explanation:** Complement of churn rate — proportion of scored customers predicted to stay.
**Visual:** Card, KPI, Gauge

---

### 4. Customer Lifetime Value (Average)

```dax
Avg Customer Lifetime Value =
    CALCULATE(
        AVERAGE(Fact_Orders[PredictedCLTV]),
        Fact_Orders[IsScored] = 1,
        Fact_Orders[PredictedCLTV] > 0
    )
```

**Explanation:** Average predicted CLV across scored customers. Excludes unscored rows (PredictedCLTV = -1).
**Visual:** Card, Bar Chart (by CustomerSegment via Dim_Customer slicer)

---

### 5. RFM Customer Segment

```dax
-- Not a measure; use as a column reference in visuals
-- Access via: Dim_Customer[CustomerSegment]
-- Or in a calculated column on Fact_Orders:
CustomerSegment = RELATED(Dim_Customer[CustomerSegment])
```

**Explanation:** Pull the RFM-based segment (Champion, Loyal, Potential Loyalist, At Risk, Lost) from the dimension table for slicing.
**Visual:** Slicer, Matrix row header, Donut Chart

---

### 6. Churn Rate MoM Change

```dax
Churn Rate MoM Change =
VAR CurrentChurnRate = [Churn Rate]
VAR PriorChurnRate =
    CALCULATE(
        [Churn Rate],
        DATEADD(Dim_Date[FullDate], -1, MONTH)
    )
RETURN
    CurrentChurnRate - PriorChurnRate
```

**Explanation:** Difference between current month's churn rate and the previous month's. Positive = churn increased.
**Visual:** KPI (with trend arrow), Line Chart

---

### 7. Total Revenue

```dax
Total Revenue = SUM(Fact_Orders[OrderAmount])
```

**Explanation:** Sum of all order amounts across all transactions (scored and unscored).
**Visual:** Card, Bar Chart, Line Chart

---

### 8. Total Customers

```dax
Total Customers = DISTINCTCOUNT(Fact_Orders[CustomerID])
```

**Explanation:** Count of all distinct customers with at least one order.
**Visual:** Card, Donut Chart (sliced by segment)

---

### 9. Average Order Value

```dax
Avg Order Value =
    DIVIDE(
        SUM(Fact_Orders[OrderAmount]),
        COUNTROWS(Fact_Orders),
        0
    )
```

**Explanation:** Average revenue per order across all transactions.
**Visual:** Card, Column Chart

---

### 10. High Risk Customer Count

```dax
High Risk Customer Count =
    CALCULATE(
        DISTINCTCOUNT(Fact_Orders[CustomerID]),
        Fact_Orders[IsScored] = 1,
        Fact_Orders[ChurnProbability] > 0.7
    )
```

**Explanation:** Number of distinct customers with churn probability above 70%. Filtered to scored customers only.
**Visual:** Card, Table (with customer-level detail drill-through)

---

### 11. Scored Customer Count

```dax
Scored Customer Count =
    CALCULATE(
        DISTINCTCOUNT(Fact_Orders[CustomerID]),
        Fact_Orders[IsScored] = 1
    )
```

**Explanation:** Number of customers with model-backed churn predictions. Use this as context for churn rate.
**Visual:** Card (alongside Churn Rate for context)

---

## Power BI Setup Notes

1. **Mark Dim_Date as a Date Table** using the `FullDate` column
2. **Set cross-filter direction** to Single for all relationships
3. **Create a page-level filter** `IsScored = 1` on any dashboard page that shows churn metrics
4. **Revenue and order count visuals** can show all customers (no IsScored filter needed)
5. The `ChurnFlag` column uses: `1` = churned, `0` = retained, `-1` = not scored
