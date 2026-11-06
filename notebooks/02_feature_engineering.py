# %% [markdown]
# # 02 — Data Cleaning, Feature Engineering & Churn Labelling
# Cutoff design: features from orders before Jul 1, 2023; churn from orders after.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in globals() else os.path.abspath(os.path.join(os.getcwd(), '..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

from src.cleaning import run_full_cleaning
from src.features import build_feature_matrix
from src.churn_label import compute_churn_labels, summarize_churn, get_category_cycles

CUTOFF_DATE = pd.Timestamp('2024-01-01')

# Ensure reports directory exists
reports_dir = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(reports_dir, exist_ok=True)

# %% [markdown]
# ## Run Cleaning Pipeline

# %%
print("Running full cleaning pipeline...")
raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
run_full_cleaning(raw_dir, processed_dir)

# %% [markdown]
# ## Build Feature Matrix

# %%
print("Building feature matrix...")
customers = pd.read_csv(os.path.join(processed_dir, 'customers_cleaned.csv'), parse_dates=['SignupDate'])
orders = pd.read_csv(os.path.join(processed_dir, 'orders_cleaned.csv'), parse_dates=['OrderDate'])
behaviour = pd.read_csv(os.path.join(processed_dir, 'behaviour_support_cleaned.csv'))

feature_matrix = build_feature_matrix(customers, orders, behaviour, CUTOFF_DATE)
print(f"Feature matrix shape: {feature_matrix.shape}")
print("Columns:", feature_matrix.columns.tolist())
print("\nFeature matrix statistics:")
print(feature_matrix.describe())

# %% [markdown]
# ## Visualize RFM Distributions

# %%
print("Visualizing RFM Distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(feature_matrix['Recency'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Recency Distribution')

sns.histplot(feature_matrix['Frequency'], bins=30, kde=True, ax=axes[1])
axes[1].set_title('Frequency Distribution')

sns.histplot(feature_matrix['Monetary'], bins=30, kde=True, ax=axes[2])
axes[2].set_title('Monetary Distribution')

plt.tight_layout()
rfm_dist_path = os.path.join(reports_dir, 'rfm_distributions.png')
plt.savefig(rfm_dist_path)
plt.close()
print(f"Saved RFM distributions to {rfm_dist_path}")

plt.figure(figsize=(10, 6))
sns.countplot(data=feature_matrix, x='CustomerSegment', order=feature_matrix['CustomerSegment'].value_counts().index)
plt.title('Customer Segment Counts')
plt.xticks(rotation=45)
plt.tight_layout()
rfm_seg_path = os.path.join(reports_dir, 'rfm_segment_counts.png')
plt.savefig(rfm_seg_path)
plt.close()

# %% [markdown]
# ## Compute Churn Labels

# %%
print("Computing Churn Labels...")
churn_labels = compute_churn_labels(orders, feature_matrix, CUTOFF_DATE)
churn_summary = summarize_churn(churn_labels)
print("\nChurn Summary:")
for k, v in churn_summary.items():
    print(f"  {k}: {v}")

print("\nCategory Cycles:")
cycles = get_category_cycles()
for cat, days in cycles.items():
    print(f"  {cat}: {days} days")

plt.figure(figsize=(6, 6))
churn_counts = churn_labels.value_counts()
plt.pie(churn_counts, labels=['Retained' if x == 0 else 'Churned' for x in churn_counts.index], 
        autopct='%1.1f%%', startangle=90, colors=['#2ca02c', '#d62728'])
plt.title('Churn Distribution')
churn_dist_path = os.path.join(reports_dir, 'churn_distribution.png')
plt.savefig(churn_dist_path)
plt.close()
print(f"Saved churn distribution to {churn_dist_path}")

# %% [markdown]
# ## Add Churn to Feature Matrix & Save

# %%
print("Adding Churn to Feature Matrix...")
# churn_labels is a Series indexed by CustomerID
feature_matrix['ChurnFlag'] = churn_labels

# Handle SignupDate if present
if 'SignupDate' in feature_matrix.columns:
    if pd.api.types.is_datetime64_any_dtype(feature_matrix['SignupDate']):
        feature_matrix['SignupDate'] = feature_matrix['SignupDate'].dt.strftime('%Y-%m-%d')

# Fill NaN in numeric columns
numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)

# Verify no NaN in critical columns
print("Missing values in critical columns:", feature_matrix[['Recency', 'Frequency', 'Monetary', 'ChurnFlag']].isna().sum().to_dict())

save_path = os.path.join(processed_dir, 'customer_features.csv')
feature_matrix.to_csv(save_path, index=True)  # keep CustomerID as index
print(f"Saved complete feature matrix to {save_path}")
print(f"Final shape: {feature_matrix.shape}")

print("\nClass Distribution:")
print(feature_matrix['ChurnFlag'].value_counts(normalize=True))

# %% [markdown]
# ## Feature Distributions by Churn Status

# %%
print("Generating feature distributions by churn status...")
key_features = ['Recency', 'Frequency', 'Monetary', 'SessionCount_30d', 'CartAbandonmentRate']
# Only plot features that exist
plot_features = [f for f in key_features if f in feature_matrix.columns]

if plot_features:
    fig, axes = plt.subplots(len(plot_features), 1, figsize=(10, 4 * len(plot_features)))
    if len(plot_features) == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, plot_features):
        sns.violinplot(data=feature_matrix, x='ChurnFlag', y=feature, ax=ax)
        ax.set_title(f'{feature} by Churn Status')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Retained (0)', 'Churned (1)'])
        
    plt.tight_layout()
    feat_churn_path = os.path.join(reports_dir, 'features_by_churn.png')
    plt.savefig(feat_churn_path)
    plt.close()
    print(f"Saved feature distributions to {feat_churn_path}")
else:
    print("None of the key features found in the dataset for plotting.")

# %% [markdown]
# ## Summary Statistics

# %%
print("Summary Statistics (Churned vs Retained):")
if plot_features:
    summary_stats = feature_matrix.groupby('ChurnFlag')[plot_features].mean().T
    summary_stats.columns = ['Retained', 'Churned']
    summary_stats['% Difference'] = ((summary_stats['Churned'] - summary_stats['Retained']) / summary_stats['Retained'] * 100).round(2)
    
    # Print markdown table manually
    print("| Feature | Retained Mean | Churned Mean | % Difference |")
    print("|---|---|---|---|")
    for idx, row in summary_stats.iterrows():
        print(f"| {idx} | {row['Retained']:.2f} | {row['Churned']:.2f} | {row['% Difference']}% |")
        
    print("\nFeatures differing most between groups:")
    print(summary_stats.sort_values(by='% Difference', key=abs, ascending=False))
