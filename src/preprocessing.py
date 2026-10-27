"""
Preprocessing module building scikit-learn Pipelines to guarantee no data leakage.
Includes ActivityClassifierPipeline wrapper for seamless string label mapping.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import config

class ActivityClassifierPipeline(BaseEstimator, ClassifierMixin):
    """
    Wrapper around an sklearn Pipeline that transparently handles
    string target encoding ('WALKING', etc.) to integers during .fit()
    and decodes back to string labels during .predict().
    """
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.label_encoder = LabelEncoder()
        self.classes_ = config.ACTIVITY_NAMES

    def fit(self, X, y):
        self.label_encoder.fit(config.ACTIVITY_NAMES)
        y_enc = self.label_encoder.transform(y)
        self.pipeline.fit(X, y_enc)
        return self

    def predict(self, X):
        y_pred_enc = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred_enc)

    def predict_proba(self, X):
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        elif hasattr(self.pipeline, "decision_function"):
            df = self.pipeline.decision_function(X)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            # Softmax to convert decision scores into probability vector
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            # Fallback uniform one-hot probability estimate
            preds = self.predict(X)
            probs = np.zeros((len(preds), len(config.ACTIVITY_NAMES)))
            for i, p in enumerate(preds):
                idx = config.ACTIVITY_NAMES.index(p)
                probs[i, idx] = 1.0
            return probs


def build_preprocessing_pipeline(
    use_scaler: bool = True,
    variance_threshold: float = None,
    use_pca: bool = False,
    n_components: float | int = 0.95,
    random_state: int = config.RANDOM_SEED
) -> Pipeline:
    """
    Constructs a feature preprocessing pipeline.
    All transformations (scaling, variance filtering, PCA) are fit ONLY on training folds.
    """
    steps = []

    if variance_threshold is not None and variance_threshold > 0:
        steps.append(('variance_filter', VarianceThreshold(threshold=variance_threshold)))

    if use_scaler:
        steps.append(('scaler', StandardScaler()))

    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=random_state)))

    if not steps:
        return Pipeline([('passthrough', 'passthrough')])

    return Pipeline(steps)


def get_pca_fitted_model(X_train, n_components: float | int = 0.95):
    """
    Fits standalone PCA on scaled X_train to analyze variance ratios and components.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    pca.fit(X_scaled)
    return scaler, pca
