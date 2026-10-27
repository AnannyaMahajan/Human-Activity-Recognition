"""
Evaluation and plotting module.
Calculates metrics (accuracy, macro/weighted precision/recall/F1),
plots high-resolution (300 DPI) confusion matrices, OvR ROC curves,
and formats evaluation tables.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import config
from src.utils import logger, save_table

# Matplotlib global style settings for publication quality
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0


def calculate_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Computes summary classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Macro_Precision": p_macro,
        "Macro_Recall": r_macro,
        "Macro_F1": f1_macro,
        "Weighted_Precision": p_weighted,
        "Weighted_Recall": r_weighted,
        "Weighted_F1": f1_weighted
    }


def compute_comprehensive_evaluation(model, X_test: pd.DataFrame, y_true: pd.Series, model_name: str = "Model") -> dict:
    """
    Performs on-demand live evaluation of a model pipeline on a test set.
    Returns summary metrics, confusion matrices (counts & normalized),
    and per-class classification report.
    """
    import time
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    total_time_ms = (time.perf_counter() - t0) * 1000.0
    latency_ms_per_sample = total_time_ms / max(1, len(X_test))

    # Metrics
    summary = calculate_metrics(y_true, y_pred, model_name=model_name)
    summary["Inference_Latency_ms_per_sample"] = latency_ms_per_sample
    summary["Total_Evaluation_Time_ms"] = total_time_ms
    summary["Sample_Count"] = len(X_test)

    # Confusion Matrices
    cm_count = confusion_matrix(y_true, y_pred, labels=config.ACTIVITY_NAMES)
    cm_norm = confusion_matrix(y_true, y_pred, labels=config.ACTIVITY_NAMES, normalize='true')

    df_cm_count = pd.DataFrame(cm_count, index=config.ACTIVITY_NAMES, columns=config.ACTIVITY_NAMES)
    df_cm_norm = pd.DataFrame(cm_norm, index=config.ACTIVITY_NAMES, columns=config.ACTIVITY_NAMES)

    # Per-Class Precision, Recall, F1
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=config.ACTIVITY_NAMES, zero_division=0
    )
    df_per_class = pd.DataFrame({
        "Activity": config.ACTIVITY_NAMES,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Support": supp
    })

    return {
        "summary": summary,
        "confusion_matrix_counts": df_cm_count,
        "confusion_matrix_normalized": df_cm_norm,
        "per_class_report": df_per_class,
        "y_pred": y_pred,
        "classes": config.ACTIVITY_NAMES
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: Path = config.FIGURES_DIR):
    """
    Plots side-by-side count and normalized confusion matrices and saves at 300 DPI.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    cm_count = confusion_matrix(y_true, y_pred, labels=config.ACTIVITY_NAMES)
    cm_norm = confusion_matrix(y_true, y_pred, labels=config.ACTIVITY_NAMES, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(cm_count, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=config.ACTIVITY_NAMES, yticklabels=config.ACTIVITY_NAMES, ax=axes[0])
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis='x', rotation=45)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', cbar=False,
                xticklabels=config.ACTIVITY_NAMES, yticklabels=config.ACTIVITY_NAMES, ax=axes[1])
    axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    filename = save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix plot to {filename}")


def plot_roc_curves(model, X_test, y_true, model_name: str, save_dir: Path = config.FIGURES_DIR):
    """
    Plots One-vs-Rest ROC curves with per-class and macro AUC.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "predict_proba"):
        logger.warning(f"Model {model_name} does not support predict_proba. Skipping ROC curve.")
        return

    y_probs = model.predict_proba(X_test)
    y_bin = label_binarize(y_true, classes=config.ACTIVITY_NAMES)
    n_classes = len(config.ACTIVITY_NAMES)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    plt.figure(figsize=(9, 7))
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=3)

    for i, activity in enumerate(config.ACTIVITY_NAMES):
        color = config.ACTIVITY_COLORS.get(activity, '#333333')
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{activity} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title(f'{model_name} - One-vs-Rest ROC Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)

    filename = save_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {filename}")


def generate_class_distribution_plot(y_train, y_test, save_dir: Path = config.FIGURES_DIR):
    """
    Plots class distribution bar chart across train and test sets.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    df_train = y_train.value_counts().reset_index()
    df_train.columns = ['Activity', 'Train Count']
    
    df_test = y_test.value_counts().reset_index()
    df_test.columns = ['Activity', 'Test Count']

    df_combined = pd.merge(df_train, df_test, on='Activity')
    df_melted = pd.melt(df_combined, id_vars=['Activity'], value_vars=['Train Count', 'Test Count'],
                        var_name='Split', value_name='Sample Count')

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_melted, x='Activity', y='Sample Count', hue='Split', palette=['#1f77b4', '#ff7f0e'])
    plt.title("UCI HAR Class Distribution across Train and Test Splits", fontsize=13, fontweight='bold')
    plt.xlabel("Activity", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.xticks(rotation=30)
    plt.grid(axis='y', alpha=0.3)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    filename = save_dir / "class_distribution.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved class distribution plot to {filename}")
