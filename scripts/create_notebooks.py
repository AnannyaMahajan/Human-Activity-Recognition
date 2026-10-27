"""
Script to programmatically create the four Jupyter Notebooks required for the project.
Uses nbformat to guarantee valid, clean notebook structure.
"""

from pathlib import Path
import nbformat as nbf
import config

def make_notebook(cells, filename):
    nb = nbf.v4.new_notebook()
    for cell_type, content in cells:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content))
            
    filepath = config.NOTEBOOKS_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created notebook: {filepath}")

def create_all_notebooks():
    # 01_EDA.ipynb
    eda_cells = [
        ("markdown", "# 01. Exploratory Data Analysis (EDA)\n## Human Activity Recognition using Smartphone Sensors"),
        ("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(str(Path.cwd().parent))
import config
from src.data_loader import load_raw_data, perform_data_quality_checks
from src.evaluate import generate_class_distribution_plot
"""),
        ("markdown", "### 1. Load UCI HAR Dataset and Check Quality"),
        ("code", """X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names = load_raw_data()
checks = perform_data_quality_checks(X_train, y_train, X_test, y_test)
print("Data Checks Summary:", checks)
"""),
        ("markdown", "### 2. Class Distribution and Subject Distribution"),
        ("code", """plt.figure(figsize=(10, 4))
sns.countplot(x=y_train, palette='viridis', order=config.ACTIVITY_NAMES)
plt.title("Training Set Class Distribution")
plt.xlabel("Activity")
plt.ylabel("Sample Count")
plt.show()

plt.figure(figsize=(12, 4))
sns.countplot(x=sub_train, palette='Blues_r')
plt.title("Samples per Subject (Training Set)")
plt.xlabel("Subject ID")
plt.ylabel("Sample Count")
plt.show()
"""),
        ("markdown", "### 3. Per-Activity Feature Mean Analysis"),
        ("code", """top_features = ['tBodyAcc-mean()-X', 'tGravityAcc-mean()-X', 'tBodyGyro-mean()-X', 'tBodyAccJerk-mean()-X']
df_top = X_train[top_features].copy()
df_top['Activity'] = y_train.values

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    sns.boxplot(data=df_top, x='Activity', y=feat, ax=axes[i], palette='Set2')
    axes[i].set_title(f"Distribution of {feat}")
    axes[i].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()
"""),
        ("markdown", "### 4. Dimensionality Reduction (PCA 2D Projection)"),
        ("code", """from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=config.RANDOM_SEED)
X_pca_2d = pca_2d.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
df_pca['Activity'] = y_train.values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Activity', palette=config.ACTIVITY_COLORS, alpha=0.7)
plt.title("2D PCA Projection: Dynamic vs Static Activity Separation")
plt.grid(True, alpha=0.3)
plt.show()
""")
    ]

    # 02_Preprocessing_and_Leakage_Check.ipynb
    preproc_cells = [
        ("markdown", "# 02. Preprocessing & Leakage Prevention Verification\n## Human Activity Recognition"),
        ("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

sys.path.append(str(Path.cwd().parent))
import config
from src.data_loader import load_raw_data
from src.preprocessing import build_preprocessing_pipeline
"""),
        ("markdown", "### 1. Proof of No Subject Leakage (Disjoint Sets)"),
        ("code", """X_train, y_train, sub_train, X_test, y_test, sub_test, _ = load_raw_data()
train_subs = set(sub_train.unique())
test_subs = set(sub_test.unique())

print("Train Subjects:", train_subs)
print("Test Subjects:", test_subs)
print("Intersection:", train_subs.intersection(test_subs))
assert train_subs.isdisjoint(test_subs), "Leakage detected!"
print("✅ Subject Disjoint Check PASSED!")
"""),
        ("markdown", "### 2. Sklearn Pipeline Fitting Verification"),
        ("code", """pipeline = build_preprocessing_pipeline(use_scaler=True, use_pca=True, n_components=0.95)
print("Pipeline Steps:", pipeline.steps)

# Fit on training data fold ONLY
pipeline.fit(X_train, y_train)
X_test_trans = pipeline.transform(X_test)
print("Transformed X_test shape:", X_test_trans.shape)
""")
    ]

    # 03_Model_Training_and_Tuning.ipynb
    train_cells = [
        ("markdown", "# 03. Model Training & Hyperparameter Tuning\n## Human Activity Recognition"),
        ("code", """import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd().parent))
import config
from src.train import train_and_evaluate_models
"""),
        ("markdown", "### Train All Models (Full vs PCA) with GroupKFold CV"),
        ("code", """results_df = train_and_evaluate_models(
    model_names=["LogisticRegression", "KNN", "SVM_Linear", "SVM_RBF", "RandomForest", "XGBoost", "AdaBoost", "LDA", "QDA", "GaussianNB", "DecisionTree"],
    pca_modes=[False, True],
    tune=False,
    fast_mode=False
)
display(results_df)
""")
    ]

    # 04_Evaluation_and_Statistical_Tests.ipynb
    eval_cells = [
        ("markdown", "# 04. Model Evaluation & Statistical Hypothesis Testing\n## Human Activity Recognition"),
        ("code", """import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd().parent))
import config
from src.stats_tests import run_mcnemar_test, run_paired_ttest, bootstrap_confidence_intervals
from src.experiments import run_all_experiments
"""),
        ("markdown", "### 1. Load Model Comparison Table"),
        ("code", """df_results = pd.read_csv(config.TABLES_DIR / "model_comparison_results.csv")
display(df_results)
"""),
        ("markdown", "### 2. Run Syllabus Experiments E1 - E7"),
        ("code", """run_all_experiments()
print("All 7 experiments completed!")
""")
    ]

    make_notebook(eda_cells, "01_EDA.ipynb")
    make_notebook(preproc_cells, "02_Preprocessing_and_Leakage_Check.ipynb")
    make_notebook(train_cells, "03_Model_Training_and_Tuning.ipynb")
    make_notebook(eval_cells, "04_Evaluation_and_Statistical_Tests.ipynb")

if __name__ == "__main__":
    create_all_notebooks()
