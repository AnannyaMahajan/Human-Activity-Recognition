# Human Activity Recognition (HAR) using Smartphone Sensor Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-quality, end-to-end Machine Learning comparative study and web application for **Human Activity Recognition (HAR)** built on the benchmark UCI HAR Dataset (30 subjects, 6 activities, 561 pre-engineered sensor features, 10,299 samples).

---

## 🚀 Key Highlights & Features

- **Zero Data Leakage:** Strict subject-wise partition (21 train subjects, 9 test subjects) verified by disjoint set assertion.
- **GroupKFold Cross-Validation:** 5-fold `GroupKFold` grouped by subject ID to test true out-of-subject generalization.
- **Pipeline Encapsulation:** All scalers, variance filters, and PCA transformers fit strictly on training folds inside scikit-learn `Pipeline` objects.
- **Comparative Model Zoo:** Evaluates 12+ machine learning algorithms across Full Feature (561 features) and PCA (102 features, 95% variance) modes.
- **Syllabus-Linked Experiments:** Includes 7 empirical studies (Curse of Dimensionality, SVM kernel/C parameter, Bias-Variance, Bagging vs. Boosting, Tree Pruning, Clustering, Feature Importance).
- **Statistical Hypothesis Testing:** McNemar's test, paired t-tests on GroupKFold folds, and 1,000-sample bootstrap 95% confidence intervals.
- **Interactive Streamlit Web App:** Real-time prediction gauge, class probability charts, 561-feature batch CSV upload, data exploration, and experimental results tabs.

---

## 📊 Summary Results Table

| Model Pipeline | Feature Mode | Test Accuracy | Test Macro-F1 | 95% Bootstrap CI | Inference Time (ms/sample) | Model Size (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** 🏆 | Full (561) | **96.13%** | **0.9610** | [0.9541 - 0.9676] | 0.015 ms | 52 KB |
| **SVM (Linear)** | Full (561) | **96.06%** | **0.9604** | [0.9535 - 0.9670] | 0.420 ms | 1,240 KB |
| **Logistic Regression** | PCA (102) | **95.82%** | **0.9578** | [0.9502 - 0.9648] | 0.022 ms | 38 KB |
| **SVM (RBF)** | Full (561) | **95.25%** | **0.9521** | [0.9442 - 0.9598] | 1.850 ms | 4,850 KB |
| **XGBoost** | Full (561) | **93.89%** | **0.9381** | [0.9295 - 0.9462] | 0.085 ms | 2,150 KB |
| **Random Forest** | Full (561) | **92.60%** | **0.9248** | [0.9150 - 0.9342] | 0.180 ms | 14,210 KB |
| **KNN** | PCA (102) | **90.23%** | **0.9005** | [0.8895 - 0.9112] | 0.850 ms | 8,420 KB |

---

## 🔒 How Data Leakage Was Prevented

1. **Subject Disjointness:** The dataset consists of recordings from 30 subjects. We enforced a strict subject-level split:
   - **Training:** 21 subjects (`[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`)
   - **Testing:** 9 subjects (`[2, 4, 9, 10, 12, 13, 18, 20, 24]`)
   - **Assertion:** Verified via `assert set(train_subjects).isdisjoint(set(test_subjects))` in `src/data_loader.py`.
2. **Encapsulated Preprocessing:** Feature scaling (`StandardScaler`) and dimensionality reduction (`PCA`) are never fit on the combined dataset. They are fit *only* on training folds inside scikit-learn `Pipeline` objects.
3. **GroupKFold Grouping:** Standard random K-Fold splits consecutive sensor time windows into both train and test folds, artificially inflating validation accuracy. Using `GroupKFold(n_splits=5)` groups data by `subject_id`, guaranteeing out-of-subject validation.

---

## 📁 Repository Structure

```
har-project/
├── config.py                 # Central config (paths, seed=42, activity mappings)
├── requirements.txt          # Pinned python dependencies
├── .gitignore                # Data & cache ignores
├── Dockerfile                # Streamlit app containerization
├── README.md                 # Project documentation
├── data/                     # Raw UCI HAR Dataset (auto-downloaded)
│   └── sample_input.csv      # Sample rows for app demo
├── notebooks/                # Executable Jupyter Notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_and_Leakage_Check.ipynb
│   ├── 03_Model_Training_and_Tuning.ipynb
│   └── 04_Evaluation_and_Statistical_Tests.ipynb
├── src/                      # Core ML source modules
│   ├── data_loader.py        # Download, parse & deduplicate features
│   ├── preprocessing.py      # Sklearn Pipelines & PCA builder
│   ├── models.py             # Model zoo & hyperparameter grids
│   ├── train.py              # CLI training & evaluation runner
│   ├── evaluate.py           # Metrics, ROC curves, confusion matrices
│   ├── stats_tests.py        # McNemar, paired t-test, bootstrap CIs
│   ├── experiments.py        # E1 - E7 syllabus experiments
│   └── utils.py              # Logging, serialization, helpers
├── models/                   # Saved best_model.joblib & metadata.json
├── reports/                  # Generated figures, tables & report draft
│   ├── figures/              # 300 DPI high-res plots (>20 figures)
│   ├── tables/               # CSV & Markdown comparison tables
│   ├── report_draft.md       # ~10-page IEEE academic report draft
│   └── viva_notes.md         # 25 viva examination Q&As
├── app/                      # Streamlit Web Application
│   └── app.py                # Multi-tab responsive UI
└── tests/                    # Pytest test suite
    └── test_pipeline.py      # Shape, leakage, and prediction tests
```

---

## 💻 Quick Start & Setup

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/har-project.git
cd har-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data & Train Models

```bash
# Run end-to-end model training (auto-downloads UCI HAR dataset if missing)
python -m src.train --models all --pca both

# Run all 7 syllabus experiments (generates 300 DPI figures & tables)
python -m src.experiments
```

### 3. Launch Streamlit Web App

```bash
streamlit run app/app.py
```

---

## 🧪 Running Unit Tests

```bash
pytest tests/
```

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New App**, select your repository and branch.
4. Set **Main file path** to `app/app.py`.
5. Click **Deploy!**

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
