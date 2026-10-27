# Viva Voce Examination Prep: Human Activity Recognition (HAR) Project

## 25 Essential Technical Questions & Model Answers

### Q1: What is data leakage in Machine Learning, and how was it prevented in this project?
**Answer:** Data leakage occurs when information from outside the training dataset (e.g., test set or future data) is used to create or tune the model. In HAR, subject biometric signatures (e.g., body mass, gait pattern) leak across random splits if samples from the same subject appear in both train and test sets. We prevented leakage by:
1. Using the official subject-partitioned split (21 train subjects, 9 test subjects) and verifying `set(train_subjects).isdisjoint(set(test_subjects))`.
2. Encapsulating all transformers (`StandardScaler`, `VarianceThreshold`, `PCA`) inside `sklearn.pipeline.Pipeline`, ensuring fitting occurs strictly on training folds during CV.

---

### Q2: Why did you use GroupKFold instead of standard Stratified K-Fold cross-validation?
**Answer:** Standard K-Fold splits samples randomly, which places consecutive time windows from the same subject into both training and validation folds. This causes artificial over-optimism due to intra-subject correlation. `GroupKFold(n_splits=5)` groups data by `subject_id`, ensuring that all samples belonging to a specific subject remain within a single fold, providing realistic out-of-subject generalization estimates.

---

### Q3: Why is Macro-F1 score preferred over simple Accuracy as the primary evaluation metric?
**Answer:** While the UCI HAR dataset is relatively balanced, Macro-F1 calculates the unweighted mean of F1 scores across all 6 activity classes ($\text{Macro-F1} = \frac{1}{C}\sum_{i=1}^{C} F1_i$). This treats each activity class (e.g., WALKING vs LAYING) with equal importance, preventing dominant classes from masking misclassifications in minority or subtle classes (such as SITTING vs STANDING).

---

### Q4: Explain the "Curse of Dimensionality" in the context of KNN on 561 features.
**Answer:** As feature dimensions ($d$) increase, the volume of feature space grows exponentially, making data points extremely sparse. In high-dimensional spaces ($d=561$), Euclidean distance between any pair of points converges to nearly the same value ($d_{\max} - d_{\min} \to 0$), rendering nearest-neighbor distance metrics uninformative. Applying PCA ($d=102$) significantly reduced inference latency while maintaining or improving accuracy.

---

### Q5: How does PCA select the number of principal components?
**Answer:** Principal Component Analysis (PCA) performs eigendecomposition on the feature covariance matrix (or SVD on scaled $X$). We computed the cumulative explained variance ratio $\sum_{i=1}^{k} \lambda_i / \sum_{j=1}^{D} \lambda_j$ and selected $k$ components that retain at least 95% of total variance ($k=102$ components out of 561).

---

### Q6: What is the difference between Linear, RBF, and Polynomial SVM kernels?
**Answer:** 
- **Linear Kernel ($k(x, y) = x^T y$):** Constructs a linear hyperplane in input space; fast and effective when features are linearly separable.
- **RBF Kernel ($k(x, y) = \exp(-\gamma \|x - y\|^2)$):** Maps data into an infinite-dimensional Hilbert space using Gaussian radial basis functions, capturing complex non-linear decision boundaries.
- **Polynomial Kernel ($k(x, y) = (x^T y + c)^d$):** Maps features into non-linear combinations up to degree $d$.

---

### Q7: What is the role of the hyperparameter $C$ in SVM?
**Answer:** $C$ controls the trade-off between maximizing the decision margin and minimizing training classification errors (slack variable penalty). Small $C$ encourages a wider margin (high bias, underfitting), whereas large $C$ strictly penalizes misclassifications (low bias, high variance, risk of overfitting).

---

### Q8: What does the $\gamma$ parameter control in RBF SVM?
**Answer:** $\gamma$ defines the radius of influence of individual support vectors ($\frac{1}{2\sigma^2}$). Small $\gamma$ means a far reach (smooth boundary, high bias), while large $\gamma$ means a tight reach around individual training points (complex boundary, risk of overfitting).

---

### Q9: Explain the difference between Bagging (Random Forest) and Boosting (XGBoost).
**Answer:**
- **Bagging (Bootstrap Aggregating):** Trains multiple deep decision trees independently in parallel on bootstrap subsamples and averages their predictions to reduce **variance**.
- **Boosting:** Trains shallow decision trees sequentially, where each new tree fits the residual errors/gradients of previous trees to reduce **bias**.

---

### Q10: Why do tree-based models like Random Forest not strictly require feature scaling?
**Answer:** Decision trees split nodes based on threshold inequalities ($x_j \le \theta$) applied to single features independently. Monotonic transformations or scaling (like `StandardScaler`) do not alter the ordering of feature values, so optimal split locations remain invariant.

---

### Q11: What is Minimal Cost-Complexity Pruning (`ccp_alpha`) in Decision Trees?
**Answer:** Cost-complexity pruning controls tree size by minimizing the cost function $R_\alpha(T) = R(T) + \alpha |T|$, where $R(T)$ is impurity/misclassification rate and $|T|$ is terminal node count. Increasing $\alpha$ prunes weak subtrees, reducing variance.

---

### Q12: How does McNemar's Test differ from a Paired t-Test?
**Answer:** 
- **McNemar's Test:** Non-parametric test performed on a $2\times 2$ contingency table of discordant predictions ($b$: M1 right/M2 wrong; $c$: M1 wrong/M2 right) on a single test set.
- **Paired t-Test:** Parametric test performed on continuous metric scores (e.g., Macro-F1) across $K$ cross-validation folds to test if mean fold difference is zero.

---

### Q13: What is the null hypothesis of McNemar's Test?
**Answer:** $H_0$: The two algorithms have equal error rates on the test set ($p_b = p_c$). If $p < 0.05$, we reject $H_0$ and conclude one model performs significantly better.

---

### Q14: How does Bootstrap 95% Confidence Interval calculation work?
**Answer:** We draw $N=1,000$ bootstrap resamples (with replacement) of size $N_{\text{test}}$ from the held-out test predictions, calculate accuracy/Macro-F1 for each sample, and extract the 2.5th and 97.5th percentiles of the empirical distribution.

---

### Q15: Why were static activities (Sitting/Standing) harder to separate than dynamic activities?
**Answer:** Dynamic activities (Walking, Walking Upstairs/Downstairs) generate high-frequency body acceleration signals ($tBodyAcc$). Static activities (Sitting, Standing) share near-zero body acceleration and identical gravitational acceleration ($tGravityAcc$) angles relative to the phone's orientation, making them sensorially subtle.

---

### Q16: What features are most important in separating Sitting vs Standing?
**Answer:** Gravitational acceleration components along the Z-axis (`tGravityAcc-mean()-Z`) and signal magnitude area (`tGravityAcc-mag`) provide the primary signal for distinguishing torso angle differences between sitting and standing.

---

### Q17: What is the difference between Adjusted Rand Index (ARI) and Silhouette Score in clustering?
**Answer:**
- **ARI:** Supervised evaluation metric measuring agreement between cluster assignments and true activity labels, corrected for chance (1.0 = perfect match, 0.0 = random).
- **Silhouette Score:** Unsupervised intrinsic metric measuring how dense and well-separated clusters are without knowing ground-truth labels (range [-1, 1]).

---

### Q18: What is Permutation Feature Importance?
**Answer:** It measures the drop in model test score after randomly shuffling values of a single feature. If shuffling a feature degrades performance significantly, the model relies heavily on that feature.

---

### Q19: Why did Linear SVM and Logistic Regression achieve high accuracy (~95-96%) on this dataset?
**Answer:** The domain experts who collected the UCI HAR dataset engineered 561 hand-crafted domain features (FFT frequencies, autoregression coefficients, energy bands), making the feature space highly linearly separable.

---

### Q20: What is XGBoost's `eval_metric='mlogloss'`?
**Answer:** Multi-class log loss (cross-entropy loss), which measures the divergence between predicted probability distributions and true class one-hot vectors: $-\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(p_{i,c})$.

---

### Q21: What is the memory footprint and latency trade-off between SVM RBF and Logistic Regression?
**Answer:** SVM RBF must store support vectors (large model file ~10-20 MB) and computes distances to all support vectors at inference time ($O(N_{sv} \cdot d)$). Logistic Regression stores only weight vector $W \in \mathbb{R}^{C \times d}$ (~50 KB) and runs matrix multiplication $W X^T$ ($O(C \cdot d)$), making it 100x faster and smaller.

---

### Q22: How does Streamlit handle model caching?
**Answer:** Streamlit uses `@st.cache_resource` for unpicklable objects like ML model pipelines (persisted across sessions in RAM) and `@st.cache_data` for DataFrames and JSON metadata.

---

### Q23: What happens if a user uploads a CSV with missing features to the web app?
**Answer:** The web application validates column count ($\text{cols} == 561$) and checks for null values, displaying a user-friendly Streamlit error message (`st.error`) rather than crashing.

---

### Q24: What is the limitation of 561 pre-engineered windowed features compared to raw time-series sensor data?
**Answer:** Pre-engineered features rely on fixed 2.56-second sliding windows (128 readings at 50Hz) and manual signal domain knowledge. They cannot capture long-term temporal dependencies or dynamic sequence patterns that deep learning models (1D-CNN / LSTM) extract directly from raw accelerometer/gyroscope signals.

---

### Q25: How would you extend this project for future real-world deployment?
**Answer:** 
1. Train End-to-End 1D-CNN / ConvLSTM models directly on raw inertial signals (`Inertial Signals/` raw sensor streams).
2. Quantize and convert the PyTorch/TensorFlow model to TFLite / ONNX for real-time, low-power on-device inference on Android/iOS smartphones.
