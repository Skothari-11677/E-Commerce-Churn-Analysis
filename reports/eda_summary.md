# Exploratory Data Analysis (EDA) Summary

This report summarizes the findings from the exploratory data analysis (EDA) conducted on the e-commerce customer churn synthetic dataset. The dataset represents an 18-month window (Jan 2023 – Jun 2024), containing approximately 10,000 customers and 50,000 orders.

## 1. Data Quality & Completeness
The synthetic dataset exhibits high quality with complete records across all core tables (Customers, Orders). Missing values are negligible, and referential integrity is maintained. The 18-month timeframe provides sufficient historical depth for churn modeling, with a clear 12-month feature window and 6-month target window.

## 2. Customer Demographics
Customer demographics are well-balanced. The average age is approximately 35 years, with an even gender distribution. Customers are distributed across 6 distinct geographic regions, providing a diverse customer base for analysis.

![Univariate Distributions](univariate_distributions.png)

## 3. Order Patterns
Customers place an average of around 5 orders over the 18-month period. Pricing and purchase frequency vary significantly across product categories. The category-specific repurchase cycles (e.g., Electronics/Groceries at 60 days vs. Furniture/Luxury at 120 days) strongly influence expected order patterns.

![Bivariate Analysis](bivariate_analysis.png)
![Correlation Heatmap](correlation_heatmap.png)

## 4. Revenue Concentration
Revenue generation follows a Pareto-like distribution (80-20 rule), where a relatively small segment of high-value customers accounts for a significant portion of the total revenue. This highlights the importance of retaining high-CLTV customer segments.

![Order Frequency Pareto](order_frequency_pareto.png)

## 5. Cohort Retention Trends
Cohort analysis reveals distinct retention curves. Initial retention drop-offs are consistent with typical e-commerce behavior, followed by stabilization for loyal cohorts. Category-adjusted churn logic accurately identifies true churners based on expected purchase intervals.

![Cohort Retention Heatmap](cohort_retention_heatmap.png)

## 6. Temporal Patterns
Overall monthly transaction volume remains relatively stable throughout the 18-month period. However, some seasonal variations are present, particularly within specific product categories (e.g., Clothing, Home & Garden).

![Temporal Trends](temporal_trends.png)

## 7. Behavioural Metrics
Behavioral indicators such as web session counts and cart abandonment rates provide strong signals for churn prediction. High cart abandonment and decreasing session frequency are strongly correlated with an increased likelihood of churn.

![Behavioural Distributions](behavioural_distributions.png)
