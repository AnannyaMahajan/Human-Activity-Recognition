"""
Statistical tests module for model comparison.
Provides McNemar's test, paired t-test on cross-validation folds,
and bootstrapping for 95% confidence intervals on test set performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, f1_score

import config
from src.utils import logger, save_table

def run_mcnemar_test(y_true, y_pred1, y_pred2, model1_name: str = "Model 1", model2_name: str = "Model 2") -> dict:
    """
    Performs McNemar's test between two models based on their predictions.
    
    Contingency table format:
                    Model 2 Correct    Model 2 Incorrect
    Model 1 Correct        a                   b
    Model 1 Inc.           c                   d
    """
    correct1 = (np.array(y_pred1) == np.array(y_true))
    correct2 = (np.array(y_pred2) == np.array(y_true))

    a = np.sum(correct1 & correct2)
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    d = np.sum(~correct1 & ~correct2)

    table = [[a, b], [c, d]]
    # Use exact test if b + c < 25, else asymptotic with continuity correction
    exact = (b + c) < 25
    result = mcnemar(table, exact=exact, correction=True)

    p_val = float(result.pvalue)
    statistic = float(result.statistic)

    is_significant = p_val < 0.05
    conclusion = (
        f"Statistically significant difference (p = {p_val:.4e} < 0.05). "
        f"{model1_name if b > c else model2_name} performs significantly better."
        if is_significant else
        f"No statistically significant difference (p = {p_val:.4f} >= 0.05)."
    )

    return {
        "Model 1": model1_name,
        "Model 2": model2_name,
        "Both Correct (a)": int(a),
        "M1 Right / M2 Wrong (b)": int(b),
        "M1 Wrong / M2 Right (c)": int(c),
        "Both Wrong (d)": int(d),
        "Statistic": statistic,
        "p-value": p_val,
        "Significant (alpha=0.05)": is_significant,
        "Conclusion": conclusion
    }


def run_paired_ttest(cv_scores1: list, cv_scores2: list, model1_name: str = "Model 1", model2_name: str = "Model 2") -> dict:
    """
    Performs paired t-test across GroupKFold CV fold scores of two models.
    """
    scores1 = np.array(cv_scores1)
    scores2 = np.array(cv_scores2)
    diff = scores1 - scores2

    t_stat, p_val = stats.ttest_rel(scores1, scores2)
    p_val = float(p_val)
    t_stat = float(t_stat)

    is_significant = p_val < 0.05
    conclusion = (
        f"Significant difference across CV folds (p = {p_val:.4f} < 0.05)."
        if is_significant else
        f"No significant difference across CV folds (p = {p_val:.4f} >= 0.05)."
    )

    return {
        "Model 1": model1_name,
        "Model 2": model2_name,
        "M1 Mean CV": float(scores1.mean()),
        "M2 Mean CV": float(scores2.mean()),
        "Mean Diff": float(diff.mean()),
        "t-statistic": t_stat,
        "p-value": p_val,
        "Significant (alpha=0.05)": is_significant,
        "Conclusion": conclusion
    }


def bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps: int = 200, ci: float = 95.0, random_state: int = config.RANDOM_SEED) -> dict:
    """
    Computes percentile bootstrap confidence intervals for Accuracy and Macro-F1.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    boot_acc = []
    boot_f1 = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_true = y_true_arr[indices]
        sample_pred = y_pred_arr[indices]

        boot_acc.append(accuracy_score(sample_true, sample_pred))
        boot_f1.append(f1_score(sample_true, sample_pred, average='macro', zero_division=0))

    lower_percentile = (100.0 - ci) / 2.0
    upper_percentile = 100.0 - lower_percentile

    acc_ci_low, acc_ci_high = np.percentile(boot_acc, [lower_percentile, upper_percentile])
    f1_ci_low, f1_ci_high = np.percentile(boot_f1, [lower_percentile, upper_percentile])

    return {
        "Accuracy_Mean": float(np.mean(boot_acc)),
        "Accuracy_CI_Low": float(acc_ci_low),
        "Accuracy_CI_High": float(acc_ci_high),
        "Accuracy_CI_Str": f"{np.mean(boot_acc):.4f} [{acc_ci_low:.4f} - {acc_ci_high:.4f}]",
        "Macro_F1_Mean": float(np.mean(boot_f1)),
        "Macro_F1_CI_Low": float(f1_ci_low),
        "Macro_F1_CI_High": float(f1_ci_high),
        "Macro_F1_CI_Str": f"{np.mean(boot_f1):.4f} [{f1_ci_low:.4f} - {f1_ci_high:.4f}]"
    }
