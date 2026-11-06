# %% [markdown]
# # 03 — Model Training & Evaluation
# Train Logistic Regression, Random Forest, and XGBoost classifiers
# for customer churn prediction. Uses SMOTE for class imbalance.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in globals() else os.path.abspath(os.path.join(os.getcwd(), '..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

from src.model import (
    prepare_data, train_models, evaluate_all_models,
    cross_validate_models, select_best_model, save_model,
    score_all_customers, plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, generate_metrics_table,
)

REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("✓ Imports and paths configured")

# %% [markdown]
# ## Load Feature Matrix with Churn Labels

# %%
features_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_features.csv')
df = pd.read_csv(features_path, index_col='CustomerID')
print(f"Feature matrix loaded: {df.shape}")
print(f"Churn distribution:\n{df['ChurnFlag'].value_counts()}")
print(f"Churn rate: {df['ChurnFlag'].mean():.2%}")

# %% [markdown]
# ## Prepare Features and Target

# %%
# Separate target
y = df['ChurnFlag'].astype(int)

# Drop columns that shouldn't be features
cols_to_drop = ['ChurnFlag', 'SignupDate', 'CustomerSegment']
# Keep DominantCategory for encoding but we need it later for star schema
# so we'll also save a copy
dominant_category = df['DominantCategory'].copy() if 'DominantCategory' in df.columns else None

X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Fill any remaining NaN
X = X.fillna(0)

print(f"Features: {X.shape[1]} columns")
print(f"Target: {len(y)} samples, {y.sum()} churned ({y.mean():.2%})")
print(f"\nFeature columns:\n{list(X.columns)}")

# %% [markdown]
# ## Train/Test Split + SMOTE

# %%
data = prepare_data(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {len(data['X_train'])} samples")
print(f"Test set:  {len(data['X_test'])} samples")
print(f"After SMOTE: {len(data['X_train_smote'])} training samples")

# %% [markdown]
# ## Train All Models

# %%
fitted_models = train_models(data, random_state=42)
print(f"\n✓ Trained {len(fitted_models)} models: {list(fitted_models.keys())}")

# %% [markdown]
# ## Evaluate on Test Set

# %%
results = evaluate_all_models(fitted_models, data)

# Print metrics comparison table
metrics_df = generate_metrics_table(results)
print("\n" + "="*70)
print("MODEL COMPARISON — TEST SET METRICS")
print("="*70)
print(metrics_df.to_string(index=False))
print("="*70)

# %% [markdown]
# ## Cross-Validation (5-Fold Stratified)

# %%
cv_results = cross_validate_models(fitted_models, data, cv=5)
print("\n5-Fold Cross-Validation Results:")
print(cv_results.to_string(index=False))

# %% [markdown]
# ## ROC Curve Comparison

# %%
fig_roc = plot_roc_curves(
    results, save_path=os.path.join(REPORTS_DIR, 'roc_curves.png')
)
plt.close(fig_roc)
print("✓ ROC curves saved to reports/roc_curves.png")

# %% [markdown]
# ## Confusion Matrices

# %%
fig_cm = plot_confusion_matrices(
    results, save_path=os.path.join(REPORTS_DIR, 'confusion_matrices.png')
)
plt.close(fig_cm)
print("✓ Confusion matrices saved to reports/confusion_matrices.png")

# %% [markdown]
# ## Select Best Model

# %%
best_result = select_best_model(results)
best_name = best_result['model_name']
best_model = fitted_models[best_name]

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_name}")
print(f"  AUC-ROC:   {best_result['auc_roc']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall:    {best_result['recall']:.4f}")
print(f"  F1 Score:  {best_result['f1']:.4f}")
print(f"{'='*50}")

# %% [markdown]
# ## Feature Importance (Best Model)

# %%
fig_fi = plot_feature_importance(
    best_model, data['feature_names'], best_name, top_n=15,
    save_path=os.path.join(REPORTS_DIR, 'feature_importance.png')
)
if fig_fi:
    plt.close(fig_fi)
    print("✓ Feature importance plot saved to reports/feature_importance.png")

# Save feature importances to CSV
if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    imp = np.abs(best_model.coef_[0])
else:
    imp = np.zeros(len(data['feature_names']))

fi_df = pd.DataFrame({
    'Feature': data['feature_names'],
    'Importance': imp
}).sort_values('Importance', ascending=False)
fi_df.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)
print("✓ Feature importances saved to models/feature_importance.csv")

# %% [markdown]
# ## Save Best Model

# %%
model_path = save_model(
    best_model, best_name, data['scaler'], data['label_encoders'],
    output_dir=MODELS_DIR
)
print(f"✓ Best model saved to {model_path}")

# %% [markdown]
# ## Score ALL Customers (for Star Schema)

# %%
# Score all customers in the feature matrix
X_all = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).fillna(0)

scores_df = score_all_customers(
    best_model, data['scaler'], data['label_encoders'],
    X_all, data['feature_names']
)
scores_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_scores.csv'))
print(f"✓ Scores saved for {len(scores_df)} customers")
print(f"  Mean churn probability: {scores_df['ChurnProbability'].mean():.3f}")
print(f"  Predicted churn rate:   {scores_df['ChurnPrediction'].mean():.2%}")

# %% [markdown]
# ## Save Evaluation Metrics

# %%
# Save metrics to JSON for later use
import json

eval_metrics = {
    'best_model': best_name,
    'test_metrics': {
        'auc_roc': best_result['auc_roc'],
        'precision': best_result['precision'],
        'recall': best_result['recall'],
        'f1': best_result['f1'],
        'accuracy': best_result['accuracy'],
    },
    'all_models': metrics_df.to_dict(orient='records'),
    'cv_results': cv_results.to_dict(orient='records'),
}
with open(os.path.join(MODELS_DIR, 'evaluation_metrics.json'), 'w') as f:
    json.dump(eval_metrics, f, indent=2)
print("✓ Evaluation metrics saved to models/evaluation_metrics.json")

print("\n" + "="*50)
print("NOTEBOOK 03 COMPLETE")
print("="*50)
