"""
Model training, evaluation, and selection for churn prediction.

Trains and compares Logistic Regression, Random Forest, and XGBoost
classifiers. Uses SMOTE for class imbalance handling (applied only to
training data to prevent data leakage). Selects the best model by AUC-ROC.

Author: Generated for Minor Project (Sem 5)
"""

import logging
import os
import warnings
from typing import Any

import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------------------------

def prepare_data(
    feature_matrix: pd.DataFrame,
    churn_labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Prepare train/test splits with SMOTE applied ONLY to training data.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Customer-level feature matrix (output of features.build_feature_matrix).
    churn_labels : pd.Series
        Binary churn labels indexed by CustomerID.
    test_size : float
        Fraction of data reserved for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test        — raw splits
        X_train_smote, y_train_smote             — SMOTE-resampled training set
        feature_names                            — list of feature column names
        scaler                                   — fitted StandardScaler
        label_encoders                           — dict of fitted LabelEncoders
    """
    # Align indices: only keep customers present in both
    common_idx = feature_matrix.index.intersection(churn_labels.index)
    X = feature_matrix.loc[common_idx].copy()
    y = churn_labels.loc[common_idx].copy()

    logger.info(f"Preparing data: {len(X)} customers, churn rate = {y.mean():.2%}")

    # --- Encode categorical columns ---
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        logger.info(f"  Label-encoded '{col}' → {len(le.classes_)} classes")

    # --- Store feature names before any array conversion ---
    feature_names = X.columns.tolist()

    # --- Train / test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"  Train: {len(X_train)} samples (churn={y_train.mean():.2%}), "
        f"Test: {len(X_test)} samples (churn={y_test.mean():.2%})"
    )

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    # --- SMOTE on training data ONLY ---
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    logger.info(
        f"  After SMOTE: {len(X_train_smote)} training samples "
        f"(churn={y_train_smote.mean():.2%})"
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "feature_names": feature_names,
        "scaler": scaler,
        "label_encoders": label_encoders,
    }


# ---------------------------------------------------------------------------
# 2. Model Training
# ---------------------------------------------------------------------------

def get_models(random_state: int = 42) -> dict:
    """
    Return a dictionary of model name → unfitted estimator.

    Models
    ------
    - Logistic Regression : linear baseline with L2 regularisation
    - Random Forest       : ensemble of 200 decision trees
    - XGBoost             : gradient-boosted trees with scale_pos_weight
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            class_weight="balanced",   # secondary imbalance handling
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            # scale_pos_weight will be set dynamically in train_models()
        ),
    }


def train_models(data: dict, random_state: int = 42) -> dict:
    """
    Train all three classifiers on the SMOTE-resampled training set.

    Parameters
    ----------
    data : dict
        Output of prepare_data().
    random_state : int
        Random seed.

    Returns
    -------
    dict : model_name → fitted estimator
    """
    models = get_models(random_state)
    fitted = {}

    # Set XGBoost scale_pos_weight from original (un-SMOTE'd) class ratio
    neg = (data["y_train"] == 0).sum()
    pos = (data["y_train"] == 1).sum()
    if pos > 0:
        models["XGBoost"].set_params(scale_pos_weight=neg / pos)
        logger.info(f"  XGBoost scale_pos_weight = {neg/pos:.2f}")

    X_train = data["X_train_smote"]
    y_train = data["y_train_smote"]

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        logger.info(f"  {name} trained successfully.")

    return fitted


# ---------------------------------------------------------------------------
# 3. Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"
) -> dict:
    """
    Evaluate a single model on the test set.

    Returns
    -------
    dict with keys: model_name, accuracy, precision, recall, f1, auc_roc,
                    confusion_matrix, y_pred, y_prob, fpr, tpr
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    logger.info(
        f"  {model_name}: AUC={auc_val:.4f}, Precision={prec:.4f}, "
        f"Recall={rec:.4f}, F1={f1:.4f}, Accuracy={acc:.4f}"
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_val,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_all_models(fitted_models: dict, data: dict) -> list[dict]:
    """
    Evaluate all fitted models and return a list of result dicts.
    """
    results = []
    for name, model in fitted_models.items():
        result = evaluate_model(model, data["X_test"], data["y_test"], name)
        results.append(result)
    return results


def cross_validate_models(
    fitted_models: dict, data: dict, cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation on the UN-SMOTE'd training data.

    This gives a more honest estimate of generalisation performance than
    evaluating on SMOTE'd data.

    Returns
    -------
    pd.DataFrame with columns: Model, Mean_AUC, Std_AUC
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_results = []

    for name, model in fitted_models.items():
        scores = cross_val_score(
            model, data["X_train"], data["y_train"],
            cv=skf, scoring="roc_auc", n_jobs=-1,
        )
        cv_results.append({
            "Model": name,
            "Mean_AUC": scores.mean(),
            "Std_AUC": scores.std(),
        })
        logger.info(f"  {name} CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(cv_results)


# ---------------------------------------------------------------------------
# 4. Model Selection & Persistence
# ---------------------------------------------------------------------------

def select_best_model(results: list[dict]) -> dict:
    """
    Select the model with the highest AUC-ROC on the test set.

    Returns the result dict for the winning model.
    """
    best = max(results, key=lambda r: r["auc_roc"])
    logger.info(
        f"Best model: {best['model_name']} (AUC-ROC = {best['auc_roc']:.4f})"
    )
    return best


def save_model(
    model, model_name: str, scaler, label_encoders: dict,
    output_dir: str = "models",
) -> str:
    """
    Persist the trained model, scaler, and label encoders using joblib.

    Saves to output_dir/best_model.joblib as a dict containing:
        - model: the fitted estimator
        - model_name: string identifier
        - scaler: fitted StandardScaler
        - label_encoders: dict of fitted LabelEncoders

    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "best_model.joblib")

    artifact = {
        "model": model,
        "model_name": model_name,
        "scaler": scaler,
        "label_encoders": label_encoders,
    }
    joblib.dump(artifact, save_path)
    logger.info(f"Model saved to {save_path}")
    return save_path


def load_model(model_path: str = "models/best_model.joblib") -> dict:
    """
    Load a previously saved model artifact.

    Returns dict with keys: model, model_name, scaler, label_encoders.
    """
    artifact = joblib.load(model_path)
    logger.info(f"Loaded model '{artifact['model_name']}' from {model_path}")
    return artifact


# ---------------------------------------------------------------------------
# 5. Scoring (generate predictions for all customers)
# ---------------------------------------------------------------------------

def score_all_customers(
    model, scaler, label_encoders: dict,
    feature_matrix: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Generate churn predictions and probabilities for ALL customers.

    Parameters
    ----------
    model : fitted estimator
    scaler : fitted StandardScaler
    label_encoders : dict of fitted LabelEncoders
    feature_matrix : the full (not split) feature matrix
    feature_names : list of feature columns used during training

    Returns
    -------
    pd.DataFrame with columns: CustomerID, ChurnProbability, ChurnPrediction
    """
    X = feature_matrix.copy()

    # Encode categoricals using the same encoders
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels by mapping to the most frequent class
            X[col] = X[col].astype(str).map(
                lambda val, _le=le: (
                    _le.transform([val])[0]
                    if val in _le.classes_
                    else 0
                )
            )

    # Ensure we use only the features the model was trained on
    X = X[feature_names]

    # Scale
    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=feature_names, index=X.index
    )

    # Predict
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    scores_df = pd.DataFrame({
        "CustomerID": X.index,
        "ChurnProbability": probs,
        "ChurnPrediction": preds,
    })
    scores_df = scores_df.set_index("CustomerID")

    logger.info(
        f"Scored {len(scores_df)} customers. "
        f"Predicted churn rate = {preds.mean():.2%}"
    )
    return scores_df


# ---------------------------------------------------------------------------
# 6. Plotting Utilities
# ---------------------------------------------------------------------------

def plot_roc_curves(results: list[dict], save_path: str = None) -> plt.Figure:
    """
    Plot ROC curves for all models on the same axes.

    Parameters
    ----------
    results : list of evaluation result dicts
    save_path : optional file path to save the figure

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for i, res in enumerate(results):
        ax.plot(
            res["fpr"], res["tpr"],
            color=colors[i % len(colors)],
            lw=2,
            label=f"{res['model_name']} (AUC = {res['auc_roc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_confusion_matrices(
    results: list[dict], save_path: str = None
) -> plt.Figure:
    """
    Plot confusion matrices for all models side by side.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
        )
        ax.set_title(f"{res['model_name']}\nAUC={res['auc_roc']:.3f}", fontsize=11)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrix Comparison", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrices saved to {save_path}")

    return fig


def plot_feature_importance(
    model, feature_names: list[str], model_name: str = "Model",
    top_n: int = 15, save_path: str = None,
) -> plt.Figure:
    """
    Plot top-N feature importances from a tree-based model.

    For Logistic Regression, uses absolute coefficient values.
    For tree models, uses the built-in feature_importances_.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"Cannot extract feature importance from {model_name}")
        return None

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    feat_imp_top = feat_imp.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp_top.plot(kind="barh", ax=ax, color="#2196F3", edgecolor="black")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13)
    ax.set_xlabel("Importance", fontsize=11)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def generate_metrics_table(results: list[dict]) -> pd.DataFrame:
    """
    Create a summary DataFrame comparing all models' metrics.

    Returns
    -------
    pd.DataFrame with columns: Model, AUC-ROC, Precision, Recall, F1, Accuracy
    """
    rows = []
    for res in results:
        rows.append({
            "Model": res["model_name"],
            "AUC-ROC": round(res["auc_roc"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1": round(res["f1"], 4),
            "Accuracy": round(res["accuracy"], 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main (for standalone testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("model.py — run via notebook 03_model_training for full pipeline.")
    print("Available functions:")
    for name in [
        "prepare_data", "train_models", "evaluate_all_models",
        "select_best_model", "save_model", "score_all_customers",
        "plot_roc_curves", "plot_confusion_matrices", "plot_feature_importance",
    ]:
        print(f"  - {name}")
