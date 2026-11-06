# Project Documentation & Academic Reports

This directory houses the formal documentation and academic report artifacts for the **E-Commerce Customer Churn Analysis & Prediction** project.

---

## Files in this Directory

| File | Description | Format |
|---|---|---|
| `Churn_Project_Report_IET_Format.pdf` | Complete academic minor project report formatted according to IET (Institute of Engineering and Technology) project guidelines. | PDF (Print-Ready) |

---

## Report Summary & Key Sections

The project report covers:
1. **Introduction & Motivation:** The cost of customer acquisition vs. retention in modern e-commerce, and the role of predictive ML pipelines in proactive retention.
2. **Literature Survey & Problem Formulation:** Analysis of classic churn modeling techniques, snapshot/cutoff design principles to prevent data leakage, and category-dependent customer repurchase cycles.
3. **Data Generation & Architecture:** Design of synthetic e-commerce transactional schemas reflecting real-world distributions (Pareto order values, Poisson order frequencies, correlated returns, and customer support tickets).
4. **Data Cleaning & Feature Engineering:** Extraction of RFM scores (Recency, Frequency, Monetary), behavioral metrics (cart abandonment, session frequency), and temporal aggregation.
5. **Model Development & Hyperparameter Tuning:** Comparative benchmarking between **Logistic Regression**, **Random Forest Classifier**, and **XGBoost Classifier**, with SMOTE (Synthetic Minority Over-sampling Technique) applied to resolve class imbalance.
6. **Model Explainability (SHAP):** Global and local interpretability using TreeSHAP to uncover feature importance and individual decision waterfalls.
7. **Business Intelligence Integration (Star Schema & Power BI):** Kimball-dimensional modeling (`Fact_Orders`, `Dim_Customer`, `Dim_Date`, `Dim_Product`, `Dim_SupportTicket`) and DAX-driven KPI reporting for executive dashboards.
8. **Conclusion & Future Scope:** Real-world deployment pathways, streaming inference architecture, and dynamic discount intervention recommendations.
