# %% [markdown]
# # 04 — Model Explainability (SHAP Analysis)
# Extract SHAP values from the best model to explain churn predictions.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in globals() else os.path.abspath(os.path.join(os.getcwd(), '..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(REPORTS_DIR, exist_ok=True)

print("✓ Imports configured")

# %% [markdown]
# ## Load Best Model & Data

# %%
# Load the saved model artifact
artifact = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
model = artifact['model']
model_name = artifact['model_name']
scaler = artifact['scaler']
label_encoders = artifact['label_encoders']
print(f"Loaded model: {model_name}")

# Load feature matrix
features_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_features.csv')
df = pd.read_csv(features_path, index_col='CustomerID')

# Prepare features (same as notebook 03)
cols_to_drop = ['ChurnFlag', 'SignupDate', 'CustomerSegment']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).fillna(0)

# Encode categoricals
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    if col in label_encoders:
        le = label_encoders[col]
        X[col] = X[col].astype(str).map(
            lambda val, _le=le: (
                _le.transform([val])[0] if val in _le.classes_ else 0
            )
        )

# Use the scaler's own feature names in the exact order it was fitted on
feature_names = list(scaler.feature_names_in_)

# Drop any remaining object columns not in feature_names (unencoded categoricals)
remaining_obj = X.select_dtypes(include=['object', 'category']).columns
for col in remaining_obj:
    if col not in feature_names:
        X = X.drop(columns=[col])

# Ensure only the training features are used, in the exact order the scaler expects
X = X.reindex(columns=feature_names, fill_value=0)

# Scale
X_scaled = pd.DataFrame(
    scaler.transform(X), columns=feature_names, index=X.index
)

print(f"Feature matrix: {X_scaled.shape}")

# %% [markdown]
# ## SHAP Explainer Setup
# Using TreeExplainer for tree-based models (Random Forest, XGBoost)
# or KernelExplainer for Logistic Regression.

# %%
# Use a sample for SHAP (full dataset can be slow for KernelExplainer)
SAMPLE_SIZE = min(1000, len(X_scaled))
np.random.seed(42)
sample_idx = np.random.choice(X_scaled.index, size=SAMPLE_SIZE, replace=False)
X_sample = X_scaled.loc[sample_idx]

print(f"Computing SHAP values for {SAMPLE_SIZE} customers...")

if model_name in ['Random Forest', 'XGBoost']:
    # TreeExplainer is fast and exact for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # TreeExplainer output varies by SHAP version:
    #   - list of [class_0_array, class_1_array]  (older)
    #   - 3D array (n_samples, n_features, n_classes) (newer)
    #   - 2D array (n_samples, n_features) for single-output
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_churn = shap_values[:, :, 1]  # class 1
    else:
        shap_values_churn = shap_values
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])  # class 1
    print(f"TreeExplainer computed (shape: {shap_values_churn.shape})")
else:
    # KernelExplainer for Logistic Regression
    # Use predict_proba and extract class-1 column
    background = shap.sample(X_scaled, 100)
    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, 1],  # Only class 1 probability
        background
    )
    shap_values_churn = explainer.shap_values(X_sample, nsamples=200)
    # Now shap_values_churn is 2D: (n_samples, n_features) — no class axis
    if shap_values_churn.ndim == 3:
        shap_values_churn = shap_values_churn[:, :, 1]  # fallback if still 3D
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0])
    print(f"KernelExplainer computed (shape: {shap_values_churn.shape})")

# %% [markdown]
# ## SHAP Summary Plot (Beeswarm)
# Shows the impact of each feature on churn prediction.
# Red = high feature value, Blue = low feature value.

# %%
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values_churn, X_sample,
    feature_names=feature_names,
    show=False, max_display=15
)
plt.title(f"SHAP Summary — {model_name}", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, 'shap_summary_beeswarm.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("✓ SHAP beeswarm plot saved to reports/shap_summary_beeswarm.png")

# %% [markdown]
# ## SHAP Bar Plot (Mean |SHAP| values)
# Shows which features have the largest overall impact.

# %%
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values_churn, X_sample,
    feature_names=feature_names,
    plot_type='bar',
    show=False, max_display=15
)
plt.title(f"Mean |SHAP| Feature Importance — {model_name}", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, 'shap_bar_importance.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("✓ SHAP bar plot saved to reports/shap_bar_importance.png")

# %% [markdown]
# ## SHAP Waterfall Plots (Individual Customers)
# Show how each feature contributes to a specific customer's churn prediction.

# %%
# Find 3 high-risk customers (highest churn probability)
scores_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_scores.csv')
scores = pd.read_csv(scores_path, index_col='CustomerID')

# Get high-risk customers that are in our SHAP sample
high_risk = scores.loc[scores.index.isin(sample_idx)].nlargest(3, 'ChurnProbability')

for i, (cid, _) in enumerate(high_risk.iterrows()):
    idx_in_sample = list(X_sample.index).index(cid)

    explanation = shap.Explanation(
        values=shap_values_churn[idx_in_sample],
        base_values=base_value,
        data=X_sample.iloc[idx_in_sample].values,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False, max_display=10)
    plt.title(f"Customer {cid} (P(churn)={scores.loc[cid, 'ChurnProbability']:.2f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        os.path.join(REPORTS_DIR, f'shap_waterfall_{i+1}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f"  Saved waterfall for customer {cid}")

print("Waterfall plots saved to reports/shap_waterfall_*.png")

# %% [markdown]
# ## Save SHAP Feature Importance

# %%
# Mean absolute SHAP values per feature
mean_shap = np.abs(shap_values_churn).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_SHAP': mean_shap
}).sort_values('Mean_SHAP', ascending=False)

shap_importance.to_csv(
    os.path.join(MODELS_DIR, 'shap_feature_importance.csv'), index=False
)
print("✓ SHAP feature importance saved to models/shap_feature_importance.csv")
print(f"\nTop 10 features by mean |SHAP|:")
print(shap_importance.head(10).to_string(index=False))

# %% [markdown]
# ## Comparison: Built-in Importance vs SHAP

# %%
builtin_imp = pd.read_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'))
builtin_imp = builtin_imp.rename(columns={'Importance': 'BuiltIn_Importance'})

comparison = shap_importance.merge(builtin_imp, on='Feature')
comparison['SHAP_Rank'] = range(1, len(comparison) + 1)
comparison = comparison.sort_values('BuiltIn_Importance', ascending=False)
comparison['BuiltIn_Rank'] = range(1, len(comparison) + 1)
comparison = comparison.sort_values('SHAP_Rank')

print("\nFeature Importance Comparison (SHAP vs Built-in):")
print(comparison[['Feature', 'SHAP_Rank', 'Mean_SHAP', 'BuiltIn_Rank', 'BuiltIn_Importance']].head(15).to_string(index=False))

print("\n" + "="*50)
print("NOTEBOOK 04 COMPLETE")
print("="*50)
