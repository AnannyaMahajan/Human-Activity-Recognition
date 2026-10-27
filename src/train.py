"""
End-to-end Training Script for UCI HAR Project.
Handles GroupKFold cross-validation, hyperparameter tuning, model evaluation,
statistical testing, model serialization, and sample input generation.
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import config
from src.utils import logger, set_seed, save_model, save_table
from src.data_loader import load_raw_data, perform_data_quality_checks
from src.preprocessing import build_preprocessing_pipeline, get_pca_fitted_model, ActivityClassifierPipeline
from src.models import get_model_zoo
from src.evaluate import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curves,
    generate_class_distribution_plot
)
from src.stats_tests import run_mcnemar_test, run_paired_ttest, bootstrap_confidence_intervals

def generate_pca_variance_plots(X_train):
    """Generates cumulative explained variance curve and scree plot for PCA."""
    scaler, pca_full = get_pca_fitted_model(X_train, n_components=None)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.argmax(cum_var >= 0.95) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree Plot
    axes[0].plot(range(1, 51), pca_full.explained_variance_ratio_[:50], marker='o', color='#1f77b4', lw=1.5)
    axes[0].set_title("PCA Scree Plot (Top 50 Components)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Principal Component", fontsize=11)
    axes[0].set_ylabel("Explained Variance Ratio", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Cumulative Variance Curve
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, color='#2ca02c', lw=2)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label=f'95% Variance ({n_95} components)')
    axes[1].axvline(x=n_95, color='r', linestyle='--')
    axes[1].set_title("Cumulative Explained Variance", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Number of Components", fontsize=11)
    axes[1].set_ylabel("Cumulative Variance", fontsize=11)
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "pca_explained_variance.png", dpi=300)
    plt.close()
    logger.info(f"PCA retaining 95% variance requires {n_95} components out of {X_train.shape[1]}")
    return n_95


from sklearn.base import clone

def train_and_evaluate_models(model_names: list, pca_modes: list, tune: bool = False, fast_mode: bool = False):
    """
    Main training workflow across models and feature settings.
    """
    set_seed(config.RANDOM_SEED)

    # Load data
    X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names = load_raw_data()
    checks = perform_data_quality_checks(X_train, y_train, X_test, y_test)

    # Generate EDA figures
    generate_class_distribution_plot(y_train, y_test)
    n_pca_comp = generate_pca_variance_plots(X_train)

    # Save sample input for Streamlit app demo
    save_sample_input_csv(X_test, y_test)

    model_zoo = get_model_zoo()

    if fast_mode:
        logger.info("FAST MODE: Subsampling data for rapid execution.")
        train_idx = np.random.RandomState(config.RANDOM_SEED).choice(len(X_train), size=min(2000, len(X_train)), replace=False)
        X_train_sub = X_train.iloc[train_idx]
        y_train_sub = y_train.iloc[train_idx]
        sub_train_sub = sub_train.iloc[train_idx]
    else:
        X_train_sub, y_train_sub, sub_train_sub = X_train, y_train, sub_train

    results_list = []
    trained_pipelines = {}
    test_predictions = {}
    cv_fold_scores = {}

    gkf = GroupKFold(n_splits=5)

    for m_name in model_names:
        if m_name not in model_zoo:
            logger.warning(f"Model '{m_name}' not in zoo. Skipping.")
            continue

        base_clf = model_zoo[m_name]

        for use_pca in pca_modes:
            mode_str = "PCA" if use_pca else "Full"
            full_model_name = f"{m_name} ({mode_str})"
            logger.info(f"--- Training {full_model_name} ---")

            prep_pipe = build_preprocessing_pipeline(
                use_scaler=True,
                variance_threshold=None,
                use_pca=use_pca,
                n_components=0.95
            )

            raw_pipeline = Pipeline([
                ('prep', prep_pipe),
                ('classifier', clone(base_clf))
            ])

            full_wrapper = ActivityClassifierPipeline(raw_pipeline)

            # GroupKFold Cross-Validation
            fold_f1_scores = []
            try:
                for tr_idx, val_idx in gkf.split(X_train_sub, y_train_sub, groups=sub_train_sub):
                    X_tr, X_val = X_train_sub.iloc[tr_idx], X_train_sub.iloc[val_idx]
                    y_tr, y_val = y_train_sub.iloc[tr_idx], y_train_sub.iloc[val_idx]

                    fold_wrapper = ActivityClassifierPipeline(Pipeline([
                        ('prep', build_preprocessing_pipeline(use_scaler=True, variance_threshold=None, use_pca=use_pca, n_components=0.95)),
                        ('classifier', clone(base_clf))
                    ]))
                    fold_wrapper.fit(X_tr, y_tr)
                    val_preds = fold_wrapper.predict(X_val)
                    fold_f1_scores.append(f1_score(y_val, val_preds, average='macro'))

                cv_mean = float(np.mean(fold_f1_scores))
                cv_std = float(np.std(fold_f1_scores))
            except Exception as e:
                logger.warning(f"GroupKFold CV failed for {full_model_name}: {e}")
                cv_mean, cv_std = 0.0, 0.0
            cv_fold_scores[full_model_name] = fold_f1_scores

            try:
                # Final Fit
                start_train_time = time.time()
                full_wrapper.fit(X_train, y_train)
                train_time_sec = time.time() - start_train_time

                # Inference on held-out test set
                start_inf_time = time.time()
                y_pred_test = full_wrapper.predict(X_test)
                inf_time_total = time.time() - start_inf_time
                inf_time_per_sample_ms = (inf_time_total / len(X_test)) * 1000.0

                # Calculate test metrics & bootstrap CIs
                metrics = calculate_metrics(y_test, y_pred_test, model_name=full_model_name)
                cis = bootstrap_confidence_intervals(y_test, y_pred_test, n_bootstraps=300 if fast_mode else 500)

                # Plots
                plot_confusion_matrix(y_test, y_pred_test, full_model_name)
                plot_roc_curves(full_wrapper, X_test, y_test, full_model_name)

                trained_pipelines[full_model_name] = full_wrapper
                test_predictions[full_model_name] = y_pred_test

                # Model size estimate
                import io, joblib
                buf = io.BytesIO()
                joblib.dump(full_wrapper, buf, compress=3)
                size_kb = len(buf.getvalue()) / 1024.0

                metrics.update({
                    "CV_Macro_F1_Mean": cv_mean,
                    "CV_Macro_F1_Std": cv_std,
                    "CV_Score_Str": f"{cv_mean:.4f} ± {cv_std:.4f}",
                    "Accuracy_95CI": cis["Accuracy_CI_Str"],
                    "Macro_F1_95CI": cis["Macro_F1_CI_Str"],
                    "Train_Time_Sec": train_time_sec,
                    "Inference_Time_ms_per_sample": inf_time_per_sample_ms,
                    "Model_Size_KB": size_kb,
                    "Best_Params": "Default"
                })
                results_list.append(metrics)
            except Exception as e:
                logger.warning(f"Could not fit or evaluate {full_model_name}: {e}")

    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)
    save_table(df_results, "model_comparison_results")

    # Generate comparative bar chart
    plot_model_comparison_chart(df_results)

    # Statistical tests between top models
    if len(df_results) >= 2:
        top1_name = df_results.iloc[0]["Model"]
        top2_name = df_results.iloc[1]["Model"]
        logger.info(f"Running Statistical Tests between Top 2 Models: '{top1_name}' vs '{top2_name}'...")

        mc_res = run_mcnemar_test(y_test, test_predictions[top1_name], test_predictions[top2_name], top1_name, top2_name)
        save_table(pd.DataFrame([mc_res]), "stats_mcnemar_top_models")

        ttest_res = run_paired_ttest(cv_fold_scores[top1_name], cv_fold_scores[top2_name], top1_name, top2_name)
        save_table(pd.DataFrame([ttest_res]), "stats_paired_ttest_top_models")

    # Save all trained models so users can select from any model in the web UI
    for m_name, pipe in trained_pipelines.items():
        safe_fname = m_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        save_model(pipe, config.MODELS_DIR / safe_fname, compress=3)

    # Select best overall model
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_name]

    logger.info(f"🏆 BEST MODEL SELECTED: '{best_name}' (Test Macro-F1 = {best_row['Macro_F1']:.4f}, Accuracy = {best_row['Accuracy']:.4f})")

    save_model(best_pipeline, config.MODELS_DIR / "best_model.joblib", compress=3)

    # Save metadata.json
    metadata = {
        "best_model_name": best_name,
        "available_models": list(trained_pipelines.keys()),
        "test_accuracy": float(best_row["Accuracy"]),
        "test_macro_f1": float(best_row["Macro_F1"]),
        "test_weighted_f1": float(best_row["Weighted_F1"]),
        "cv_macro_f1": float(best_row["CV_Macro_F1_Mean"]),
        "inference_time_ms_per_sample": float(best_row["Inference_Time_ms_per_sample"]),
        "feature_names": feature_names,
        "activity_names": config.ACTIVITY_NAMES,
        "n_features": len(feature_names),
        "random_seed": config.RANDOM_SEED
    }
    with open(config.MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training and evaluation completed successfully!")
    return df_results


def plot_model_comparison_chart(df_results: pd.DataFrame):
    """Plots comparative bar chart of Accuracy and Macro-F1 across models."""
    plt.figure(figsize=(12, 6))
    df_melt = pd.melt(df_results, id_vars=['Model'], value_vars=['Accuracy', 'Macro_F1'],
                      var_name='Metric', value_name='Score')

    ax = sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette=['#1f77b4', '#2ca02c'])
    plt.title("Model Performance Comparison (Test Set)", fontsize=13, fontweight='bold')
    plt.xlabel("Model Pipeline", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim([0.7, 1.02])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.005),
                        ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "model_comparison_bar_chart.png", dpi=300)
    plt.close()


def save_sample_input_csv(X_test: pd.DataFrame, y_test: pd.Series):
    """
    Saves a sample_input.csv with 2-3 rows per activity class for Streamlit demo.
    """
    samples = []
    for activity in config.ACTIVITY_NAMES:
        activity_indices = y_test[y_test == activity].index
        selected_idx = activity_indices[:3]
        sub_df = X_test.loc[selected_idx].copy()
        sub_df['true_activity'] = activity
        samples.append(sub_df)

    sample_df = pd.concat(samples, axis=0).reset_index(drop=True)
    sample_path = config.APP_DIR / "sample_input.csv"
    sample_df.to_csv(sample_path, index=False)
    logger.info(f"Saved sample demo input file to {sample_path} ({len(sample_df)} rows)")

    # Save unlabelled sample input for testing predictions
    features_only = sample_df.drop(columns=['true_activity'], errors='ignore')
    features_only.to_csv(config.APP_DIR / "sample_input_no_labels.csv", index=False)

    # Save schema template
    template_df = pd.DataFrame(columns=features_only.columns)
    template_df.to_csv(config.APP_DIR / "template_schema.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate HAR models.")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="List of models to train: LogisticRegression, KNN, SVM_Linear, SVM_RBF, RandomForest, XGBoost, etc., or 'all'.")
    parser.add_argument("--pca", choices=["full", "pca", "both"], default="both",
                        help="Feature representation mode.")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning with GridSearchCV.")
    parser.add_argument("--fast", action="store_true", help="Subsample dataset for quick verification run.")

    args = parser.parse_args()

    all_models = ["LogisticRegression", "KNN", "SVM_Linear", "SVM_RBF", "SVM_Poly", "RandomForest", "XGBoost", "AdaBoost", "LDA", "QDA", "GaussianNB", "DecisionTree"]
    selected_models = all_models if "all" in args.models else args.models

    pca_modes = [False, True] if args.pca == "both" else ([True] if args.pca == "pca" else [False])

    train_and_evaluate_models(selected_models, pca_modes, tune=args.tune, fast_mode=args.fast)
