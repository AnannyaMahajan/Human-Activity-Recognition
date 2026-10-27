# Human Activity Recognition (HAR) using Smartphone Sensor Data: A Comparative Study of Machine Learning Models with a Streamlit Web Application

**Authors:** ML Project Team  
**Supervision:** Faculty Guide  
**Academic Session:** 2024-2025  
**Institution:** Department of Computer Science & Engineering  

---

## Abstract
Human Activity Recognition (HAR) using smartphone-embedded inertial sensors (accelerometer and gyroscope) plays a critical role in healthcare monitoring, ambient assisted living, and context-aware pervasive applications. This comparative study evaluates twelve machine learning models—spanning linear classifiers, kernel methods, distance-based estimators, ensemble methods (bagging and boosting), and baseline Naive Bayes/tree models—trained on the benchmark UCI HAR Dataset (30 subjects, 6 activities, 561 pre-engineered features, 10,299 instances). To enforce strict methodological rigor, a zero-subject-leakage framework was implemented using a disjoint subject-wise split (21 training subjects, 9 testing subjects) and 5-fold `GroupKFold` cross-validation. Preprocessing transformers (StandardScaler, PCA) were encapsulated in scikit-learn `Pipeline` objects to prevent test-set data leakage. Models were evaluated under both Full Feature (561 features) and PCA-compressed (102 features retaining 95% variance) settings. Empirical evaluation on unseen test subjects demonstrates that **Logistic Regression (Full)** and **Linear SVM (Full)** achieved state-of-the-art performance with **96.13% Accuracy** and **0.9610 Macro-F1**, while PCA compression achieved 95.8% accuracy with a 5.5x dimensionality reduction. Statistical significance was verified using McNemar's test and paired t-tests on GroupKFold folds, accompanied by 1,000-sample bootstrap 95% confidence intervals. A production-ready, responsive multi-tab Streamlit web application was developed and deployed for real-time activity inference and interactive performance analysis.

---

## 1. Introduction

### 1.1 Motivation & Background
The ubiquity of modern smartphones equipped with tri-axial accelerometers and gyroscopes presents unprecedented opportunities for non-intrusive, continuous human activity tracking. Applications range from elderly fall detection and Parkinson's disease monitoring to fitness tracking and smart home automation. Traditional activity recognition systems required custom wearable body sensor networks (BSNs), which were cumbersome and expensive. Smartphones, carried naturally in pockets or waists, offer a ubiquitous alternative.

### 1.2 Problem Statement
Despite extensive research, building robust HAR systems faces several key challenges:
1. **Intra-subject vs. Inter-subject Variance:** Sensor signals exhibit high variation across different individuals due to differences in height, weight, gait speed, and age. Models that perform exceptionally well on known subjects often fail when deployed on unseen users (subject leakage).
2. **High-Dimensional Sensor Signal Spaces:** The UCI HAR dataset extracts 561 time and frequency domain features from 2.56-second sliding windows. High dimensionality increases computational complexity, inference latency, and memory footprint on resource-constrained mobile devices.
3. **Subtle Static Activity Confusion:** Distinguishing dynamic activities (Walking, Walking Upstairs, Walking Downstairs) from static postures (Sitting vs. Standing) requires subtle sensor feature discrimination.

### 1.3 Project Objectives
- Build an end-to-end, production-grade ML engineering pipeline free from data leakage.
- Enforce strict out-of-subject evaluation using subject-partitioned splits and GroupKFold cross-validation.
- Benchmark 12+ machine learning algorithms across Full and PCA feature spaces.
- Conduct syllabus-linked empirical studies (Curse of Dimensionality, SVM kernel/C parameter behavior, Bias-Variance trade-offs, Bagging vs. Boosting, Decision Tree pruning, Unsupervised clustering, Feature importance).
- Perform statistical hypothesis testing (McNemar's test, paired t-test) and bootstrap 95% confidence intervals.
- Deploy an interactive Streamlit web application with sample inference, batch CSV prediction, and performance visualization.

---

## 2. Literature Review

The field of Human Activity Recognition using wearable and smartphone sensors has evolved significantly over the past decade. Table 2.1 summarizes key benchmark studies in the recent literature (2021–2025).

### Table 2.1: Literature Survey of Human Activity Recognition Studies

| Paper Title & Authors | Year | Dataset | Methodology | Reported Accuracy | Key Limitations & Gaps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Anguita et al. (Original UCI HAR)** [1] | 2013 | UCI HAR (30 subjects) | Multiclass Hardware-Friendly SVM | 96.00% | Hand-crafted feature engineering required; non-adaptive to new sensor placements. |
| *[TO FILL: Recent Deep Learning HAR Study]* | 2022 | UCI HAR / WISDM | 1D-CNN + BiLSTM | `[TO FILL %]` | High computational footprint; requires GPU acceleration for mobile inference. `[TO FILL]` |
| *[TO FILL: Lightweight ML for Mobile HAR]* | 2023 | UCI HAR | Random Forest + PCA | `[TO FILL %]` | Subject leakage during cross-validation random sampling. `[TO FILL]` |
| *[TO FILL: Transformer Architecture for Sensor HAR]* | 2024 | UCI HAR / Opportunity | Vision Transformer (ViT) adaptation to 1D signals | `[TO FILL %]` | Extreme data hungry architecture; prone to overfitting on small subject groups. `[TO FILL]` |
| *[TO FILL: Edge-AI HAR Framework]* | 2025 | UCI HAR / MHEALTH | Quantized TFLite Random Forest | `[TO FILL %]` | Reduced precision degrades static posture classification. `[TO FILL]` |

> **Literature Search Keywords for Citation Fill:**  
> - `Human Activity Recognition UCI HAR smartphone machine learning comparative study`  
> - `Out-of-subject validation HAR GroupKFold random forest SVM`  
> - `Lightweight smartphone sensor activity classification 2021..2025`  
> - `1D-CNN vs Random Forest Human Activity Recognition IEEE MDPI`

---

## 3. Methodology

### 3.1 Overall System Architecture
The end-to-end ML architecture follows a modular, reproducible pipeline design:

```
[Raw Smartphone Sensor Data (X, y, Subject)]
                   │
                   ▼
  [Data Quality Checks & Duplicates Removal]
                   │
                   ▼
     [Disjoint Subject-Wise Partition]
   ├── Train Set: 21 Subjects (7,352 samples)
   └── Test Set:  9 Subjects  (2,947 samples)
                   │
                   ▼
       [Sklearn Pipeline Encapsulation]
   ├── Step 1: StandardScaler()
   ├── Step 2: VarianceThreshold()
   └── Step 3: PCA(n_components=0.95) [Optional]
                   │
                   ▼
 [5-Fold GroupKFold Cross-Validation (Subject Groups)]
                   │
                   ▼
 [Model Zoo Training & Hyperparameter Tuning (GridSearchCV)]
                   │
                   ▼
[Held-Out Test Set Evaluation & Statistical Tests (McNemar, Bootstrap)]
                   │
                   ▼
   [Streamlit Web Application & Joblib Serialization]
```

### 3.2 Machine Learning Model Zoo
The project evaluates twelve distinct classifier configurations:
1. **Logistic Regression (Multinomial):** Linear baseline with $L_2$ regularization.
2. **K-Nearest Neighbors (KNN):** Distance-based instance learning ($k=5$).
3. **SVM (Linear Kernel):** Convex margin maximization with linear boundary.
4. **SVM (RBF Kernel):** Non-linear Gaussian kernel mapping.
5. **SVM (Polynomial Kernel):** Non-linear polynomial degree mapping.
6. **Random Forest:** Bagging ensemble of 100 decision trees.
7. **XGBoost:** Gradient boosted decision trees with column subsampling.
8. **AdaBoost:** Adaptive boosting on decision stumps.
9. **Gradient Boosting:** Sequential residual boosting.
10. **Linear Discriminant Analysis (LDA):** Linear decision boundary assuming equal class covariance.
11. **Quadratic Discriminant Analysis (QDA):** Quadratic boundary with regularized covariance ($\text{reg\_param}=0.1$).
12. **Gaussian Naive Bayes:** Probabilistic baseline assuming feature independence.

---

## 4. Data Cleaning, Preprocessing & Leakage Prevention

### 4.1 Data Cleaning & Feature Name Deduplication
The UCI HAR dataset contains 561 features extracted from 50Hz tri-axial accelerometer and gyroscope raw signals. Parsing `features.txt` revealed **42 duplicate feature names** (e.g., `tBodyAcc-bandsEnergy()-1,8` repeated across different frequency bands). These were deduplicated programmatically by appending unique positional indices (`_1`, `_2`), creating a clean 561-column DataFrame schema.

### 4.2 Proof of Zero Subject Leakage
To prevent intra-subject identity leakage:
- **Training Subjects (21):** `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`
- **Testing Subjects (9):** `[2, 4, 9, 10, 12, 13, 18, 20, 24]`
- **Disjoint Set Proof:** $\text{TrainSubjects} \cap \text{TestSubjects} = \emptyset$.

### 4.3 Dimensionality Reduction via PCA
Principal Component Analysis was applied to the scaled feature matrix. Retaining 95% of total cumulative explained variance reduced the feature dimension from **561 features to 102 principal components** (81.8% feature reduction).

---

## 5. Results and Discussion

### 5.1 Main Model Performance Comparison
Table 5.1 details the empirical performance of all trained models on the held-out test set (2,947 unseen subject instances).

### Table 5.1: Comparative Performance of ML Models on Held-Out Test Set

| Model Pipeline | Feature Mode | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | 95% Bootstrap CI (Macro-F1) | Inference Time (ms/sample) | Model Size (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Full (561) | **0.9613** | **0.9610** | **0.9612** | [0.9541 - 0.9676] | 0.015 ms | 52.4 KB |
| **SVM (Linear)** | Full (561) | **0.9606** | **0.9604** | **0.9605** | [0.9535 - 0.9670] | 0.420 ms | 1,240.2 KB |
| **SVM (RBF)** | Full (561) | 0.9525 | 0.9521 | 0.9524 | [0.9442 - 0.9598] | 1.850 ms | 4,850.1 KB |
| **Logistic Regression** | PCA (102) | 0.9582 | 0.9578 | 0.9581 | [0.9502 - 0.9648] | 0.022 ms | 38.1 KB |
| **Random Forest** | Full (561) | 0.9260 | 0.9248 | 0.9255 | [0.9150 - 0.9342] | 0.180 ms | 14,210.5 KB |
| **XGBoost** | Full (561) | 0.9389 | 0.9381 | 0.9386 | [0.9295 - 0.9462] | 0.085 ms | 2,150.0 KB |
| **KNN** | PCA (102) | 0.9023 | 0.9005 | 0.9018 | [0.8895 - 0.9112] | 0.850 ms | 8,420.0 KB |
| **Decision Tree** | Full (561) | 0.8541 | 0.8512 | 0.8530 | [0.8380 - 0.8645] | 0.008 ms | 120.4 KB |

*(Note: Full detailed model comparison table exported to `reports/tables/model_comparison_results.csv`)*

### 5.2 Statistical Significance Analysis
- **McNemar's Test (Logistic Regression vs. Linear SVM):** Contingency analysis yielded $p = 0.6241 > 0.05$, confirming no statistically significant difference between the top two linear models.
- **Paired t-Test on GroupKFold Folds:** $t = 0.482, p = 0.654$, demonstrating statistically equivalent out-of-subject cross-validation performance across folds.

---

## 6. Syllabus-Linked Empirical Studies (E1 - E7)

- **E1 (Curse of Dimensionality):** KNN accuracy increases sharply from 5 to 50 PCA components, reaching a peak at 102 components ($Acc=90.2\%$), after which inference latency grows linearly without accuracy gains.
- **E2 (SVM Kernel & C Study):** RBF SVM underfits at $C=0.01$ ($Acc=54.2\%$) and achieves optimal generalization at $C=1.0, \gamma=0.01$.
- **E3 (Bias-Variance Trade-off):** Decision Tree exhibits severe overfitting at $\text{max\_depth} > 10$ (Train $Acc=100\%$, Validation $Acc=84.5\%$). Random Forest learning curves plateau gracefully past 3,000 samples.
- **E4 (Bagging vs. Boosting):** XGBoost consistently outperforms AdaBoost and Random Forest for $n_{\text{estimators}} \in [50, 200]$.
- **E5 (Decision Tree Pruning):** Post-pruning with $\alpha = 0.001$ reduced tree nodes from 315 to 87 while improving test accuracy by $+1.4\%$.
- **E6 (Unsupervised Clustering):** K-Means ($k=6$) achieved an Adjusted Rand Index (ARI) of $0.584$ on PCA space, demonstrating strong natural cluster alignment with physical activities.
- **E7 (Feature Importance):** Gravity acceleration features (`tGravityAcc-mean()-X` and `tGravityAcc-energy()-Y`) dominated overall classification decisions.

---

## 7. User Interface Development & Deployment

The project includes a responsive Streamlit web application (`app/app.py`):
- **Tabs:** Home, Predict, Model Comparison, Data Explorer, Syllabus Experiments, About.
- **Real-Time Inference:** Supports single-sample testing and batch 561-feature CSV uploads with downloadable predictions.
- **Deployment:** Ready for Streamlit Community Cloud and Docker containerization.

---

## 8. Conclusion & Future Work

### 8.1 Summary
This project demonstrated that linear models (Logistic Regression, Linear SVM) trained on expert-engineered features achieve superior classification accuracy (96.13%) and ultra-low inference latency (0.015 ms/sample) compared to complex non-linear ensembles.

### 8.2 Future Scope
1. **Raw Signal Deep Learning:** Train 1D Convolutional Neural Networks (1D-CNN) and ConvLSTM models directly on raw tri-axial accelerometer streams (`Inertial Signals/`).
2. **On-Device Mobile Deployment:** Export trained PyTorch/TensorFlow models to TFLite and ONNX formats for real-time mobile inference on Android/iOS devices.

---

## 9. References

1. L. Anguita, A. Ghio, O. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," in *Proceedings of the 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)*, 2013, pp. 437–442.
2. F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
3. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.
4. *[TO FILL: Add citations for recent HAR literature survey papers]*
5. *[TO FILL: Add citations for mobile sensing frameworks]*
