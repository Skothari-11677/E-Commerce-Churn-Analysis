# %% [markdown]
# # 01 — Exploratory Data Analysis
# Synthetic e-commerce dataset: ~10,000 customers, ~50,000 orders (Jan 2023 – Jun 2024)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in globals() else os.path.abspath(os.path.join(os.getcwd(), '..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())
sys.path.insert(0, PROJECT_ROOT)
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)
print(f"Reports directory set to: {REPORTS_DIR}")

# %% [markdown]
# ## 2: Load Data

# %%
print("Loading data...")
customers_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'customers.csv')
orders_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'orders.csv')
behaviour_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'behaviour_support.csv')

customers = pd.read_csv(customers_path, parse_dates=['SignupDate'])
orders = pd.read_csv(orders_path, parse_dates=['OrderDate'])
behaviour = pd.read_csv(behaviour_path)

print(f"Customers shape: {customers.shape}")
print(f"Orders shape: {orders.shape}")
print(f"Behaviour shape: {behaviour.shape}")

print("\n--- Customers Info ---")
customers.info()
print("\n--- Orders Info ---")
orders.info()
print("\n--- Behaviour Info ---")
behaviour.info()

print("\n--- Customers Head ---")
print(customers.head())
print("\n--- Orders Head ---")
print(orders.head())

print("\n--- Customers Describe ---")
print(customers.describe(include='all'))

# %% [markdown]
# ## 3: Missing Values & Data Quality

# %%
print("Checking for missing values and data quality...")

print("\nMissing values in Customers:")
print(customers.isnull().sum())
print("\nMissing values in Orders:")
print(orders.isnull().sum())
print("\nMissing values in Behaviour:")
print(behaviour.isnull().sum())

print("\nDuplicates in Customers:", customers.duplicated().sum())
print("Duplicates in Orders:", orders.duplicated().sum())
print("Duplicates in Behaviour:", behaviour.duplicated().sum())

# Plot missing values heatmap for all
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.heatmap(customers.isnull(), cbar=False, ax=axes[0], yticklabels=False, cmap='viridis')
axes[0].set_title('Missing Values - Customers')
sns.heatmap(orders.isnull(), cbar=False, ax=axes[1], yticklabels=False, cmap='viridis')
axes[1].set_title('Missing Values - Orders')
sns.heatmap(behaviour.isnull(), cbar=False, ax=axes[2], yticklabels=False, cmap='viridis')
axes[2].set_title('Missing Values - Behaviour')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'missing_values_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 4: Univariate Distributions

# %%
print("Plotting univariate distributions...")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

# Age
sns.histplot(customers['Age'], kde=True, ax=axes[0])
axes[0].set_title('Customer Age Distribution')

# Gender
sns.countplot(data=customers, x='Gender', ax=axes[1])
axes[1].set_title('Gender Distribution')

# Region
sns.countplot(data=customers, x='Region', ax=axes[2])
axes[2].set_title('Region Distribution')
axes[2].tick_params(axis='x', rotation=45)

# AcquisitionChannel
sns.countplot(data=customers, x='AcquisitionChannel', ax=axes[3])
axes[3].set_title('Acquisition Channel')
axes[3].tick_params(axis='x', rotation=45)

# OrderAmount
sns.histplot(orders['OrderAmount'], bins=50, log_scale=True, ax=axes[4])
axes[4].set_title('Order Amount (Log Scale)')

# Quantity
sns.histplot(orders['Quantity'], discrete=True, ax=axes[5])
axes[5].set_title('Quantity per Order')

# ProductCategory
sns.countplot(data=orders, y='ProductCategory', ax=axes[6], order=orders['ProductCategory'].value_counts().index)
axes[6].set_title('Order Counts by Product Category')

# Remove empty subplots
for i in range(7, 9):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'univariate_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 5: Bivariate Analysis

# %%
print("Plotting bivariate analysis...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot of OrderAmount by ProductCategory
order_amt_cat = orders.groupby('ProductCategory')['OrderAmount'].median().sort_values().index
sns.boxplot(data=orders, x='OrderAmount', y='ProductCategory', order=order_amt_cat, ax=axes[0])
axes[0].set_title('Order Amount by Product Category')
axes[0].set_xscale('log')

# Avg OrderAmount by Region
merged_co = pd.merge(orders, customers, on='CustomerID')
avg_order_region = merged_co.groupby('Region')['OrderAmount'].mean().reset_index()
sns.barplot(data=avg_order_region, x='Region', y='OrderAmount', ax=axes[1])
axes[1].set_title('Average Order Amount by Region')
axes[1].tick_params(axis='x', rotation=45)

# Scatter of Age vs Total Spending
cust_spending = merged_co.groupby(['CustomerID', 'Age'])['OrderAmount'].sum().reset_index()
sns.scatterplot(data=cust_spending, x='Age', y='OrderAmount', alpha=0.3, ax=axes[2])
axes[2].set_title('Age vs Total Spending')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'bivariate_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 6: Correlation Heatmap

# %%
print("Creating correlation heatmap...")
# Aggregate orders per customer
cust_orders_agg = orders.groupby('CustomerID').agg({
    'OrderID': 'count',
    'OrderAmount': 'sum',
    'Quantity': 'sum'
}).rename(columns={'OrderID': 'TotalOrders', 'OrderAmount': 'TotalSpend', 'Quantity': 'TotalItems'})

# Merge all
merged_all = customers.merge(cust_orders_agg, on='CustomerID', how='left').merge(behaviour, on='CustomerID', how='left')

# Select numerical columns
num_cols = merged_all.select_dtypes(include=[np.number])
corr_matrix = num_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap - Customer Level')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 7: Order Frequency Distribution

# %%
print("Plotting order frequency and Pareto analysis...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of orders-per-customer
sns.histplot(cust_orders_agg['TotalOrders'], bins=30, ax=axes[0])
axes[0].set_title('Orders per Customer')
axes[0].set_xlabel('Number of Orders')

# Pareto Analysis
spend_sorted = cust_orders_agg['TotalSpend'].sort_values(ascending=False).values
cum_spend = np.cumsum(spend_sorted)
cum_spend_pct = cum_spend / cum_spend[-1] * 100
cum_cust_pct = np.arange(1, len(spend_sorted) + 1) / len(spend_sorted) * 100

axes[1].plot(cum_cust_pct, cum_spend_pct, marker='', linewidth=2)
axes[1].set_title('Revenue Pareto Analysis (Lorenz Curve)')
axes[1].set_xlabel('Cumulative % of Customers')
axes[1].set_ylabel('Cumulative % of Revenue')
axes[1].grid(True)

# Annotate 80/20 point (or approximate)
idx_80 = np.abs(cum_spend_pct - 80).argmin()
axes[1].axvline(cum_cust_pct[idx_80], color='r', linestyle='--')
axes[1].axhline(cum_spend_pct[idx_80], color='r', linestyle='--')
axes[1].annotate(f'Top {cum_cust_pct[idx_80]:.1f}% customers\ngenerate 80% revenue', 
                 xy=(cum_cust_pct[idx_80], 80), xytext=(cum_cust_pct[idx_80]+5, 70),
                 arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'order_frequency_pareto.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 8: Cohort Retention Analysis

# %%
print("Performing cohort retention analysis...")
# Cohort Month: SignupDate month
merged_co['CohortMonth'] = merged_co['SignupDate'].dt.to_period('M')
# Order Month
merged_co['OrderMonth'] = merged_co['OrderDate'].dt.to_period('M')

# Group by CohortMonth and OrderMonth
cohort_data = merged_co.groupby(['CohortMonth', 'OrderMonth'])['CustomerID'].nunique().reset_index()

# Calculate months since signup
cohort_data['CohortIndex'] = (cohort_data['OrderMonth'] - cohort_data['CohortMonth']).apply(lambda x: x.n)

# Pivot table
cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
cohort_sizes = customers['SignupDate'].dt.to_period('M').value_counts().sort_index()

retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100

plt.figure(figsize=(12, 8))
sns.heatmap(retention_matrix, annot=True, fmt='.1f', cmap='Blues', vmin=0.0, vmax=100)
plt.title('Monthly Cohort Retention Rate (%)')
plt.ylabel('Cohort Month (Signup)')
plt.xlabel('Months Since Signup')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'cohort_retention_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 9: Temporal Trends

# %%
print("Plotting temporal trends...")
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

monthly_orders = orders.set_index('OrderDate').resample('ME').size()
monthly_revenue = orders.set_index('OrderDate').resample('ME')['OrderAmount'].sum()

# Monthly order count
axes[0].plot(monthly_orders.index, monthly_orders.values, marker='o')
axes[0].set_title('Monthly Order Count')
axes[0].set_ylabel('Number of Orders')
axes[0].axvline(pd.to_datetime('2023-07-01'), color='r', linestyle='--', label='Cutoff Date (Jul 1, 2023)')
axes[0].legend()

# Monthly revenue
axes[1].plot(monthly_revenue.index, monthly_revenue.values, marker='o', color='green')
axes[1].set_title('Monthly Revenue')
axes[1].set_ylabel('Revenue')
axes[1].axvline(pd.to_datetime('2023-07-01'), color='r', linestyle='--', label='Cutoff Date (Jul 1, 2023)')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'temporal_trends.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 10: Behavioural Data Distribution

# %%
print("Plotting behavioural data distributions...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(behaviour['SessionCount_30d'], bins=30, ax=axes[0])
axes[0].set_title('Session Count (Last 30 Days)')

sns.histplot(behaviour['CartAbandonmentRate'], bins=20, ax=axes[1])
axes[1].set_title('Cart Abandonment Rate')

sns.histplot(behaviour['SupportTicketCount'], discrete=True, ax=axes[2])
axes[2].set_title('Support Ticket Count')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'behavioural_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 11: Summary

# %%
print("\n" + "="*50)
print("EDA SUMMARY OF KEY FINDINGS:")
print("="*50)
print('''
1. Data Quality: Verified presence of ~10k customers and ~50k orders. Checked for nulls and duplicates.
2. Demographics: Visualized distributions of age, gender, and region.
3. Spending Behavior: Evaluated order amounts and identified top-performing categories and regions.
4. Retention: Tracked monthly cohort retention across the 18-month timeline.
5. Trends: Plotted monthly revenue and orders, highlighting the July 1, 2023 cutoff date.
6. Customer Value: Conducted Pareto analysis to show the concentration of revenue among top customers.
7. Engagement: Analyzed recent sessions, support tickets, and cart abandonment rates.
''')
print("Plots saved in:", REPORTS_DIR)
