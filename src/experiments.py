"""
Experiments module for syllabus-linked Machine Learning studies (E1 - E7).
Each experiment generates 300 DPI figures and exports CSV/Markdown tables.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score, f1_score
from sklearn.model_selection import learning_curve, validation_curve, GroupKFold
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import dendrogram, linkage

import config
from src.utils import logger, save_table
from src.data_loader import load_raw_data

# Set global figure style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_experiment_1_knn_curse_of_dimensionality(X_train, y_train, X_test, y_test):
    """
    E1. KNN Accuracy and Inference Time vs PCA components (5, 10, 20, 50, 100, 200, 561).
    """
    logger.info("Running Experiment 1: KNN Curse of Dimensionality...")
    components_list = [5, 10, 20, 50, 100, 200, 561]
    results = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for n in components_list:
        if n < 561:
            pca = PCA(n_components=n, random_state=config.RANDOM_SEED)
            X_tr = pca.fit_transform(X_train_scaled)
            X_te = pca.transform(X_test_scaled)
        else:
            X_tr, X_te = X_train_scaled, X_test_scaled

        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_tr, y_train)

        start_time = time.time()
        y_pred = knn.predict(X_te)
        inf_time = (time.time() - start_time) * 1000.0 / len(X_te) # ms per sample

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append({
            "Components": n,
            "Accuracy": acc,
            "Macro_F1": f1,
            "Inference_Time_ms_per_sample": inf_time
        })

    df_res = pd.DataFrame(results)
    save_table(df_res, "exp1_knn_curse_of_dimensionality")

    # Plot dual-axis
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = '#1f77b4'
    ax1.set_xlabel('Number of PCA Components', fontsize=11)
    ax1.set_ylabel('Test Accuracy', color=color, fontsize=11)
    ax1.plot(df_res['Components'], df_res['Accuracy'], marker='o', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Inference Time (ms/sample)', color=color, fontsize=11)
    ax2.plot(df_res['Components'], df_res['Inference_Time_ms_per_sample'], marker='s', color=color, linewidth=2, linestyle='--', label='Inference Time')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("E1: KNN Curse of Dimensionality & Speed vs. PCA Components", fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp1_knn_curse_of_dimensionality.png", dpi=300)
    plt.close()
    logger.info("Experiment 1 complete.")


def run_experiment_2_svm_kernels_and_c(X_train, y_train, X_test, y_test):
    """
    E2. SVM Accuracy vs C (log scale) for Linear, RBF, and Poly kernels; Gamma heatmap for RBF.
    """
    logger.info("Running Experiment 2: SVM Kernel & C Study...")
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    kernels = ['linear', 'rbf', 'poly']
    results = []

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    for k in kernels:
        for c in c_values:
            svm = SVC(kernel=k, C=c, max_iter=5000, random_state=config.RANDOM_SEED)
            svm.fit(X_tr, y_train)
            train_acc = accuracy_score(y_train, svm.predict(X_tr))
            test_acc = accuracy_score(y_test, svm.predict(X_te))
            results.append({
                "Kernel": k,
                "C": c,
                "Train_Accuracy": train_acc,
                "Test_Accuracy": test_acc
            })

    df_res = pd.DataFrame(results)
    save_table(df_res, "exp2_svm_c_kernel_study")

    # Plot Accuracy vs C for kernels
    plt.figure(figsize=(9, 5))
    for k in kernels:
        sub = df_res[df_res['Kernel'] == k]
        plt.plot(sub['C'], sub['Test_Accuracy'], marker='o', label=f'Kernel={k}', linewidth=2)

    plt.xscale('log')
    plt.xlabel('Regularization Parameter C (log scale)', fontsize=11)
    plt.ylabel('Test Set Accuracy', fontsize=11)
    plt.title("E2: SVM Test Accuracy vs. C for Linear, RBF, and Poly Kernels", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp2_svm_c_study.png", dpi=300)
    plt.close()

    # RBF Gamma & C Heatmap
    gamma_vals = [0.0001, 0.001, 0.01, 0.1]
    heatmap_data = np.zeros((len(c_values), len(gamma_vals)))
    for i, c in enumerate(c_values):
        for j, g in enumerate(gamma_vals):
            svm = SVC(kernel='rbf', C=c, gamma=g, max_iter=5000, random_state=config.RANDOM_SEED)
            svm.fit(X_tr, y_train)
            heatmap_data[i, j] = accuracy_score(y_test, svm.predict(X_te))

    df_heatmap = pd.DataFrame(heatmap_data, index=c_values, columns=gamma_vals)
    save_table(df_heatmap, "exp2_svm_rbf_gamma_heatmap")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_heatmap, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Test Accuracy'})
    plt.title("E2: SVM (RBF) Test Accuracy Heatmap: C vs. Gamma", fontsize=13, fontweight='bold')
    plt.xlabel("Gamma Parameter", fontsize=11)
    plt.ylabel("C Parameter", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp2_svm_rbf_gamma_heatmap.png", dpi=300)
    plt.close()
    logger.info("Experiment 2 complete.")


def run_experiment_3_bias_variance(X_train, y_train, subject_train):
    """
    E3. Learning curves & validation curves for Decision Tree (max_depth) and KNN (k).
    """
    logger.info("Running Experiment 3: Bias-Variance & Validation Curves...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)

    gkf = GroupKFold(n_splits=5)

    # Validation curve for Decision Tree max_depth
    depths = [1, 2, 3, 5, 8, 12, 18, 25]
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=config.RANDOM_SEED),
        X_tr, y_train,
        param_name="max_depth",
        param_range=depths,
        cv=gkf.split(X_tr, y_train, groups=subject_train),
        scoring="accuracy",
        n_jobs=-1
    )

    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    df_dt_val = pd.DataFrame({
        "max_depth": depths,
        "Train_Mean": tr_mean, "Train_Std": tr_std,
        "Val_Mean": val_mean, "Val_Std": val_std
    })
    save_table(df_dt_val, "exp3_dt_validation_curve")

    plt.figure(figsize=(9, 5))
    plt.plot(depths, tr_mean, label="Training Score", color="#1f77b4", marker="o", lw=2)
    plt.fill_between(depths, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#1f77b4")
    plt.plot(depths, val_mean, label="Validation Score (GroupKFold)", color="#ff7f0e", marker="s", lw=2)
    plt.fill_between(depths, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#ff7f0e")

    plt.xlabel("Decision Tree max_depth", fontsize=11)
    plt.ylabel("Accuracy Score", fontsize=11)
    plt.title("E3: Decision Tree Validation Curve (Underfitting to Overfitting)", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp3_dt_validation_curve.png", dpi=300)
    plt.close()

    # Learning Curve for Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_SEED, n_jobs=-1)
    train_sizes, train_lc, val_lc = learning_curve(
        rf, X_tr, y_train,
        cv=gkf.split(X_tr, y_train, groups=subject_train),
        train_sizes=np.linspace(0.1, 1.0, 6),
        scoring="accuracy",
        n_jobs=-1
    )

    tr_lc_mean = train_lc.mean(axis=1)
    val_lc_mean = val_lc.mean(axis=1)

    df_lc = pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_Acc": tr_lc_mean,
        "Val_Acc": val_lc_mean
    })
    save_table(df_lc, "exp3_learning_curve_rf")

    plt.figure(figsize=(9, 5))
    plt.plot(train_sizes, tr_lc_mean, 'o-', color='#2ca02c', label='Training Score', lw=2)
    plt.plot(train_sizes, val_lc_mean, 's-', color='#d62728', label='Validation Score', lw=2)
    plt.xlabel("Training Examples", fontsize=11)
    plt.ylabel("Accuracy Score", fontsize=11)
    plt.title("E3: Random Forest Learning Curve vs. Sample Size", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp3_learning_curve.png", dpi=300)
    plt.close()
    logger.info("Experiment 3 complete.")


def run_experiment_4_bagging_vs_boosting(X_train, y_train, X_test, y_test):
    """
    E4. Bagging vs Boosting: RF vs XGBoost vs AdaBoost across n_estimators.
    """
    logger.info("Running Experiment 4: Bagging vs Boosting...")
    n_estimators_list = [10, 25, 50, 100, 150, 200]
    results = []

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Encode target labels as integers for XGBoost
    label_mapping = {name: i for i, name in enumerate(config.ACTIVITY_NAMES)}
    y_train_enc = y_train.map(label_mapping)
    y_test_enc = y_test.map(label_mapping)

    for n in n_estimators_list:
        # Random Forest (Bagging)
        rf = RandomForestClassifier(n_estimators=n, random_state=config.RANDOM_SEED, n_jobs=-1)
        rf.fit(X_tr, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_te))

        # XGBoost (Boosting)
        xgb = XGBClassifier(n_estimators=n, learning_rate=0.1, random_state=config.RANDOM_SEED, eval_metric='mlogloss', n_jobs=-1)
        xgb.fit(X_tr, y_train_enc)
        xgb_acc = accuracy_score(y_test_enc, xgb.predict(X_te))

        # AdaBoost (Boosting)
        ada = AdaBoostClassifier(n_estimators=n, random_state=config.RANDOM_SEED)
        ada.fit(X_tr, y_train)
        ada_acc = accuracy_score(y_test, ada.predict(X_te))

        results.append({
            "n_estimators": n,
            "RandomForest_Bagging": rf_acc,
            "XGBoost_Boosting": xgb_acc,
            "AdaBoost_Boosting": ada_acc
        })

    df_res = pd.DataFrame(results)
    save_table(df_res, "exp4_bagging_vs_boosting")

    plt.figure(figsize=(9, 5))
    plt.plot(df_res['n_estimators'], df_res['RandomForest_Bagging'], marker='o', label='Random Forest (Bagging)', lw=2)
    plt.plot(df_res['n_estimators'], df_res['XGBoost_Boosting'], marker='s', label='XGBoost (Boosting)', lw=2)
    plt.plot(df_res['n_estimators'], df_res['AdaBoost_Boosting'], marker='^', label='AdaBoost (Boosting)', lw=2)

    plt.xlabel("Number of Estimators (Trees)", fontsize=11)
    plt.ylabel("Test Set Accuracy", fontsize=11)
    plt.title("E4: Bagging vs. Boosting Ensemble Accuracy vs. n_estimators", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp4_bagging_vs_boosting.png", dpi=300)
    plt.close()
    logger.info("Experiment 4 complete.")


def run_experiment_5_decision_tree_pruning(X_train, y_train, X_test, y_test):
    """
    E5. Decision Tree Pruning: Minimal cost-complexity pruning ccp_alpha vs Accuracy.
    """
    logger.info("Running Experiment 5: Decision Tree Pruning...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Get effective alphas
    dt_base = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    path = dt_base.cost_complexity_pruning_path(X_tr, y_train)
    ccp_alphas = path.ccp_alphas

    # Subsample alphas if too many
    if len(ccp_alphas) > 20:
        ccp_alphas = ccp_alphas[::len(ccp_alphas)//20]

    results = []
    for alpha in ccp_alphas:
        dt = DecisionTreeClassifier(random_state=config.RANDOM_SEED, ccp_alpha=alpha)
        dt.fit(X_tr, y_train)
        tr_acc = accuracy_score(y_train, dt.predict(X_tr))
        te_acc = accuracy_score(y_test, dt.predict(X_te))
        nodes = dt.tree_.node_count
        results.append({
            "ccp_alpha": alpha,
            "Node_Count": nodes,
            "Train_Accuracy": tr_acc,
            "Test_Accuracy": te_acc
        })

    df_res = pd.DataFrame(results)
    save_table(df_res, "exp5_dt_pruning")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df_res['ccp_alpha'], df_res['Train_Accuracy'], marker='o', label='Train Accuracy', color='#1f77b4', lw=2)
    ax1.plot(df_res['ccp_alpha'], df_res['Test_Accuracy'], marker='s', label='Test Accuracy', color='#ff7f0e', lw=2)
    ax1.set_xlabel("Cost Complexity Parameter (ccp_alpha)", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.title("E5: Decision Tree Post-Pruning (ccp_alpha vs. Accuracy)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp5_dt_pruning.png", dpi=300)
    plt.close()
    logger.info("Experiment 5 complete.")


def run_experiment_6_unsupervised_clustering(X_train, y_train):
    """
    E6. Unsupervised Clustering: K-Means (k=6) and GMM on PCA features,
    Adjusted Rand Index & Silhouette score + Hierarchical Dendrogram.
    """
    logger.info("Running Experiment 6: Unsupervised Clustering & Dendrogram...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=50, random_state=config.RANDOM_SEED)
    X_tr_pca = pca.fit_transform(X_tr_scaled)

    # K-Means (k=6)
    kmeans = KMeans(n_clusters=6, random_state=config.RANDOM_SEED, n_init=10)
    km_labels = kmeans.fit_predict(X_tr_pca)
    km_ari = adjusted_rand_score(y_train, km_labels)
    km_sil = silhouette_score(X_tr_pca[:2000], km_labels[:2000])

    # GMM (k=6)
    gmm = GaussianMixture(n_components=6, random_state=config.RANDOM_SEED)
    gmm_labels = gmm.fit_predict(X_tr_pca)
    gmm_ari = adjusted_rand_score(y_train, gmm_labels)
    gmm_sil = silhouette_score(X_tr_pca[:2000], gmm_labels[:2000])

    res_df = pd.DataFrame([
        {"Algorithm": "K-Means (k=6)", "ARI": km_ari, "Silhouette_Score": km_sil},
        {"Algorithm": "GMM (k=6)", "ARI": gmm_ari, "Silhouette_Score": gmm_sil}
    ])
    save_table(res_df, "exp6_clustering_metrics")

    # Hierarchical Clustering Dendrogram on subsample of 100 samples
    sub_indices = np.random.RandomState(config.RANDOM_SEED).choice(len(X_tr_pca), size=100, replace=False)
    X_sub = X_tr_pca[sub_indices]
    y_sub = y_train.iloc[sub_indices].values

    linked = linkage(X_sub, method='ward')

    plt.figure(figsize=(11, 6))
    dendrogram(linked, labels=y_sub, orientation='top', distance_sort='descending', show_leaf_counts=True, leaf_rotation=90)
    plt.title("E6: Hierarchical Clustering Dendrogram (Ward Linkage, 100 Samples)", fontsize=13, fontweight='bold')
    plt.xlabel("Sample True Activity Label", fontsize=11)
    plt.ylabel("Euclidean Distance", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp6_hierarchical_dendrogram.png", dpi=300)
    plt.close()
    logger.info("Experiment 6 complete.")


def run_experiment_7_feature_importance(X_train, y_train, X_test, y_test, feature_names):
    """
    E7. Feature Importance: Random Forest top-20 features + Permutation Importance.
    """
    logger.info("Running Experiment 7: Feature Importance Analysis...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, y_train)

    # MDI Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_features_mdi = [feature_names[i] for i in indices]
    top_scores_mdi = importances[indices]

    df_mdi = pd.DataFrame({"Feature": top_features_mdi, "MDI_Importance": top_scores_mdi})
    save_table(df_mdi, "exp7_top20_features_mdi")

    # Plot Top 20 MDI Features
    plt.figure(figsize=(10, 6))
    sns.barplot(x="MDI_Importance", y="Feature", data=df_mdi, palette="Blues_r")
    plt.title("E7: Top 20 Features by Mean Decrease in Impurity (Random Forest)", fontsize=13, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp7_feature_importance_rf.png", dpi=300)
    plt.close()

    # Permutation Importance on Test Set (using top 10 features for speed)
    perm_imp = permutation_importance(rf, X_te, y_test, n_repeats=5, random_state=config.RANDOM_SEED, n_jobs=-1)
    perm_indices = np.argsort(perm_imp.importances_mean)[::-1][:20]
    df_perm = pd.DataFrame({
        "Feature": [feature_names[i] for i in perm_indices],
        "Permutation_Importance_Mean": perm_imp.importances_mean[perm_indices],
        "Permutation_Importance_Std": perm_imp.importances_std[perm_indices]
    })
    save_table(df_perm, "exp7_permutation_importance")

    logger.info("Experiment 7 complete.")


def run_all_experiments():
    """Runs all 7 syllabus experiments."""
    X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names = load_raw_data()
    run_experiment_1_knn_curse_of_dimensionality(X_train, y_train, X_test, y_test)
    run_experiment_2_svm_kernels_and_c(X_train, y_train, X_test, y_test)
    run_experiment_3_bias_variance(X_train, y_train, sub_train)
    run_experiment_4_bagging_vs_boosting(X_train, y_train, X_test, y_test)
    run_experiment_5_decision_tree_pruning(X_train, y_train, X_test, y_test)
    run_experiment_6_unsupervised_clustering(X_train, y_train)
    run_experiment_7_feature_importance(X_train, y_train, X_test, y_test, feature_names)

if __name__ == "__main__":
    run_all_experiments()
